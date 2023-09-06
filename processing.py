import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

import time
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

from inference import make_requests
from arguments import parse_args_for_post_processing
from template import get_template
from utils import *

similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
df = pd.DataFrame(columns=["rouge score", "sbert score"], data=np.zeros((10, 2)))
if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args_for_post_processing()
    rng = np.random.default_rng(args.seed)
    set_seed(args.seed)
    accelerator = Accelerator()

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    seed_instructions = [
        {
            "instruction": t["instruction"],
            "input": t["instances"][0]["input"],
            "output": t["instances"][0]["output"],
        }
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to(accelerator.device)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # load the data to post process
    machine_instructions = []
    if os.path.exists(args.input_data_path):
        with open(args.input_data_path, "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info)
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")
    # check if there is already a file containing the scores
    start = 0
    if os.path.exists(args.output_data_path):
        with open(args.output_data_path, "r") as fin:
            for line in fin:
                start += 1
    progress_bar = tqdm.tqdm(total=len(machine_instructions))
    progress_bar.update(start)
    # now let's evaluate our new instructions!
    L = []
    header = open("./prompts/output_prompt.txt").read() + "\n###\n"
    template = get_template(args.template_name)
    average_request_duration = 0
    average_process_duration = 0
    for i, inst in enumerate(machine_instructions[start:]):
        values = []
        output_dictionary = {}
        instruction_i = inst["instruction"]
        inst_tokens = scorer._tokenizer.tokenize(instruction_i)
        inst_embedding = similarity_model.encode(instruction_i, convert_to_tensor=True)
        output = inst["output"]
        # repeat for many combinations
        batch_inputs = []
        # create a batch with multiple few-shot samples to predict the same instruction
        progress_bar.update(1)
        for j in range(args.num_trials):
            # Just sample seed instructions from the pool
            seed_indices = rng.choice(
                a=np.arange(len(seed_instructions)),
                size=args.num_prompt_instructions,
                replace=False,
            )
            prompt_instructions = [seed_instructions[p] for p in seed_indices]
            rng.shuffle(prompt_instructions)
            prompt = header
            for idx, instruction in enumerate(prompt_instructions):
                prompt += (
                    template.get_inverse_biprompt(instruction, prefix=f"{idx+1}. ")
                    + "\n###\n"
                )

            prompt += (
                f"{args.num_prompt_instructions+1}. {template.output_token}:\n{machine_instructions[i]['output']}"
                + f"\n\n{args.num_prompt_instructions+1}. {template.instruction_token}:\n"
            )
            batch_inputs.append(prompt)
        request_start = time.time()
        results = make_requests(
            accelerator,
            model,
            tokenizer,
            prompts=batch_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_words=args.stop_words + [f"\n{args.num_prompt_instructions+2}."],
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
        )
        request_duration = time.time() - request_start
        instructions = []
        process_start = time.time()
        for result in results:
            anchor = (
                f"\n\n{args.num_prompt_instructions+1}. {template.instruction_token}:\n"
            )
            end_of_prompt = result["answer"].find(anchor)
            predicted_instruction = result["answer"][
                end_of_prompt + len(anchor) :
            ].strip()
            if predicted_instruction.endswith("###"):
                predicted_instruction = predicted_instruction[::-3]
            print(f"PREDICTION\n{predicted_instruction}")
            instructions.append({"instruction": predicted_instruction})

        process_duration = time.time() - process_start
        total = len(instructions)
        if total == 0:
            output_dictionary[instruction] = []
            L.append(i)
            if accelerator.is_main_process:
                fout.write(json.dumps(output_dictionary) + "\n")
            continue
        for candidate in instructions:
            rouge_score = rouge_scorer._score_lcs(
                inst_tokens, scorer._tokenizer.tokenize(candidate["instruction"])
            )
            rouge_score = rouge_score.fmeasure
            candidate_embedding = similarity_model.encode(
                candidate["instruction"], convert_to_tensor=True
            )
            sbert_score = util.pytorch_cos_sim(
                inst_embedding, candidate_embedding
            ).item()
            values.append((candidate["instruction"], rouge_score, sbert_score))

        # values = sorted(values, key = lambda x : -x[1])
        max_rouge_score = max([b for a, b, c in values])
        max_sbert_score = max([c for a, b, c in values])
        output_dictionary[instruction_i] = values
        # print(values)
        df["rouge score"] += (max_rouge_score >= np.arange(10) / 10).astype("int")
        df["sbert score"] += (max_sbert_score >= np.arange(10) / 10).astype("int")
        average_request_duration += request_duration
        average_process_duration += process_duration
        if (i + 1) % 50 == 0:
            average_request_duration /= 50
            average_process_duration /= 50
            print(
                f"i: {i}, Request duration: {average_request_duration:.2f}s, processing duration: {average_process_duration:.2f}s"
            )
            average_request_duration = 0
            average_process_duration = 0
        if accelerator.is_main_process:
            with open(args.output_data_path, "a") as fout:
                fout.write(json.dumps(output_dictionary) + "\n")
        print(output_dictionary)
    df.to_csv("statistics.csv", index=False)
    np.savetxt("forgotten.txt", np.array(L))

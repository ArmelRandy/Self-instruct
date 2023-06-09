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
from local_api import make_requests

import time
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from templates import INSTRUCTION, INPUT, OUTPUT


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("../prompt.txt").read() + "\n"
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = (
            task_dict["instruction"],
            task_dict["input"],
            task_dict["output"],
        )
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. {INSTRUCTION}: {instruction}\n"
        prompt += f"{idx + 1}. {INPUT}:\n{input}\n"
        prompt += f"{idx + 1}. {OUTPUT}:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. {INSTRUCTION}:"
    return prompt


def sample_machine_instructions(machine_instructions, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def post_process_starcoder_response(num_prompt_instructions, response):
    """
    Response is a  dictionary of the form
    {
        "text" : instruction
        "index" : integer
        "finish_reason" : length or not
    }
    The function takes the first element of the list. And divides the completion ("text") into 1 or more instructions. WHY?
    The model is prompted with 1. ... n. and will complete n+1. ... m. so we try to retrieve the m-n new instructions
    """
    if response is None:
        return []
    end_of_prompt = response["text"].find(
        f"{num_prompt_instructions+1}. {INSTRUCTION}:"
    )
    if end_of_prompt >= 0:
        raw_instructions = response["text"][end_of_prompt:]
    else:
        raw_instructions = response["text"]

    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        splitted_data = re.split(
            f"{idx+num_prompt_instructions+1}\.\s+({INSTRUCTION}|{INPUT}|{OUTPUT}):",
            inst,
        )
        # index 0 : everything that comes before x. Instruction
        # index 1 : x. instruction
        # index 2 : the instruction
        # index 3 : x. Input
        # index 4 : the input
        # index 5 : x. Output
        # index 6 : the output + the rest of the world before (x+1). Instruction:
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program") or inst.startswith("Write a function"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/starcoder_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="The number of instructions to generate.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bigcode/starcoder",
        help="The name or path of the model to use.",
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
        help="The number of requests to send to the model at a time.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="The number of machine generated instructions to use in the prompt.",
    )
    parser.add_argument(
        "--max_length",
        default=4096,
        type=int,
        help="The max_length parameter of generate.",
    )
    parser.add_argument(
        "--temperature",
        default=0.2,
        type=float,
        help="The temperature of the model.",
    )
    parser.add_argument(
        "--top_p",
        default=0.9,
        type=float,
        help="The `top_p` parameter of the model.",
    )
    parser.add_argument(
        "--stop_words",
        default=["\n20", "20.", "20 ."],
        nargs="+",
        help="The `stop_words` that are considered with the generation.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of responses to generate.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="The beam size on the model used in the decoding.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="The repetition penalty parameter to use for the generation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()
    random.seed(args.seed)
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

    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(
        os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")
    ):
        with open(
            os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r"
        ) as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info)
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instructions] + [
        d["instruction"] for d in machine_instructions
    ]
    all_instruction_tokens = [
        scorer._tokenizer.tokenize(inst) for inst in all_instructions
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to(accelerator.device)

    with open(
        os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a"
    ) as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            request_idx += 1
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, n=args.n
                )
                # sample human instructions from the pool
                prompt_instructions += random.sample(
                    seed_instructions,
                    args.num_prompt_instructions - len(prompt_instructions),
                )
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions)
                batch_inputs.append(prompt)
            request_start = time.time()
            results = make_requests(
                accelerator,
                model,
                tokenizer,
                prompts=batch_inputs,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_words=args.stop_words,
                num_return_sequences=args.num_return_sequences,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
            )
            request_duration = time.time() - request_start
            print(f"Request {request_idx} took {request_duration:.2f}s")
            instructions = []
            process_start = time.time()
            for result in results:
                for r in range(len(result["response"])):
                    new_instructions = post_process_starcoder_response(
                        args.num_prompt_instructions, result["response"][r]
                    )
                    instructions += new_instructions

            total = len(instructions)
            keep = 0
            for inst in instructions:
                inst_tokens = scorer._tokenizer.tokenize(inst["instruction"])
                with Pool(4) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, inst_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                if max(rouge_scores) > args.threshold:
                    continue
                keep += 1
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                machine_instructions.append(inst)
                all_instruction_tokens.append(inst_tokens)
                all_instructions.append(inst["instruction"])
                progress_bar.update(1)
                if accelerator.is_main_process:
                    fout.write(
                        json.dumps(
                            dict(
                                inst,
                                **{
                                    "most_similar": most_similar_instructions,
                                    "avg_similarity_score": float(
                                        np.mean(rouge_scores)
                                    ),
                                },
                            )
                        )
                        + "\n"
                    )
            process_duration = time.time() - process_start
            print(
                f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
            )
            print(f"Generated {total} instructions, kept {keep} instructions")

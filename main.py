import os
import json
import logging
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer


import time
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

from inference import make_requests
from arguments import parse_args
from template import get_template
from utils import *
from tgi import run_eval

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.environ.get("HF_TOKEN", "<YOUR TOKEN HERE>")
    api_url = os.environ.get("API_URL", "<YOUR API URL HERE>")
    logger = logging.getLogger(__name__)
    # get input arguments
    args = parse_args()
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
    print(f"Loaded {len(seed_instructions)} human-written seed instructions.")

    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(args.output_data_path):
        with open(args.output_data_path, "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info)
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm(total=args.num_instructions_to_generate)
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
    request_batch_size = args.request_batch_size
    if request_batch_size % accelerator.num_processes != 0:
        request_batch_size = (
            1 + args.request_batch_size // accelerator.num_processes
        ) * accelerator.num_processes
        warnings.warn(
            f"Your request batch size ({args.request_batch_size}) is can not be divided by the number of processes. We'll pad it to {request_batch_size}."
        )
    template = get_template(args.template_name)
    header = open("./prompts/prompt.txt").read() + "\n###\n"
    request_idx = 0
    while len(machine_instructions) < args.num_instructions_to_generate:
        request_idx += 1
        batch_inputs = []
        for _ in range(request_batch_size):
            # sample seed instructions
            seed_indices = rng.choice(
                a=np.arange(len(seed_instructions)),
                size=args.num_prompt_instructions
                - args.num_prompt_synthetic_instructions,
                replace=False,
            )
            if len(machine_instructions) >= args.num_prompt_synthetic_instructions:
                # sample machine generated instructions
                synthetic_indices = rng.choice(
                    a=np.arange(len(machine_instructions)),
                    size=args.num_prompt_synthetic_instructions,
                    replace=False,
                )
                prompt_instructions = [seed_instructions[p] for p in seed_indices] + [
                    machine_instructions[p] for p in synthetic_indices
                ]
            else:
                synthetic_indices = rng.choice(
                    a=np.arange(len(seed_instructions)),
                    size=args.num_prompt_synthetic_instructions,
                    replace=False,
                )
                prompt_instructions = [seed_instructions[p] for p in seed_indices] + [
                    seed_instructions[p] for p in synthetic_indices
                ]

            rng.shuffle(prompt_instructions)
            prompt = header
            if args.format == 2:
                for idx, instruction in enumerate(prompt_instructions):
                    prompt += (
                        template.get_biprompt(instruction, prefix=f"{idx+1}. ")
                        + "\n###\n"
                    )
            else:
                for idx, instruction in enumerate(prompt_instructions):
                    prompt += (
                        template.get_triprompt(instruction, prefix=f"{idx+1}. ")
                        + "\n###\n"
                    )
            prompt += (
                f"{args.num_prompt_instructions+1}. {template.instruction_token}:\n"
            )
            batch_inputs.append(prompt)

        if args.use_tgi:
            inputs = [
                (
                    prompt,
                    hf_token,
                    api_url,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    logger
                )
                for prompt in batch_inputs
            ]
            request_start = time.time()
            with Pool(12) as pool:
                results = list(tqdm(pool.imap(run_eval, inputs), total=len(inputs)))
            request_duration = time.time() - request_start
            print(f"Request {request_idx} took {request_duration:.2f}s")

            instructions = []
            process_start = time.time()
            for result, did_run in results:
                if did_run:
                    new_instructions = post_process_response(
                        num_prompt_instructions=args.num_prompt_instructions,
                        response=result,
                        template=template,
                        format=args.format,
                        keep_programming=args.keep_programming,
                    )
                    instructions += new_instructions
        else:
            request_start = time.time()
            results = make_requests(
                accelerator,
                model,
                tokenizer,
                prompts=batch_inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_words=args.stop_words,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
            )
            request_duration = time.time() - request_start
            print(f"Request {request_idx} took {request_duration:.2f}s")

            instructions = []
            process_start = time.time()
            for result in results:
                new_instructions = post_process_response(
                    num_prompt_instructions=args.num_prompt_instructions,
                    response=result["answer"],
                    template=template,
                    format=args.format,
                    keep_programming=args.keep_programming,
                )
                instructions += new_instructions

        total = len(instructions)
        keep = 0
        with open(args.output_data_path, "a") as fout:
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
        print(f"Request {request_idx}'s processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        if keep == 0 :
            for p, inst in enumerate(instructions) : 
                print(f"p = {p}, inst = {inst['instruction']}")

import os
import json
import logging
import requests
import argparse
from tqdm import tqdm
from multiprocessing.pool import Pool

"""
def run_eval(
    input_prompt: str,
    hf_token: str,
    api_url: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logger: logging.Logger = None,
):
"""


def run_eval(inputs):
    input_prompt, hf_token, api_url, max_new_tokens, temperature, top_p, logger = inputs
    data = {
        "inputs": input_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "best_of" : 1,
            #"stop" :
        },
    }
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(data),
        auth=("hf", hf_token),
        stream=False,
    )
    if response.status_code == 200:
        response_data = response.json()
        generated_text = response_data[0]["generated_text"]
        return generated_text, True
    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        return "", False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=20,
        type=int,
        help="max number of new tokens to generate.",
    )
    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float,
        help="top_p argument for nucleus sampling.",
    )
    parser.add_argument(
        "--temperature",
        default=0.2,
        type=float,
        help="temperature of the generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logger = logging.getLogger(__name__)
    prompts = [
        "What is the capital of Cameroon?",
        "What is the capital of Burundi?",
        "What is the capital of Ethiopia?",
        "What is the capital of Rwanda?",
    ]
    hf_token = os.environ.get("HF_TOKEN", "<YOUR TOKEN HERE>")
    api_url = os.environ.get("API_URL", "https://api-inference.huggingface.co/models/codellama/CodeLlama-13b-hf")
    inputs = [
        (prompt, hf_token, api_url, args.max_new_tokens, args.temperature, args.top_p, logger)
        for prompt in prompts
    ]
    with Pool(12) as pool:
        review_jsons = list(tqdm(pool.imap(run_eval, inputs), total=len(prompts)))
    print(f"REVIEW {review_jsons}")

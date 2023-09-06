import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data. The keys of the dictionary should be `instruction`, `input` and `output`.",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        required=True,
        default="data/output.jsonl",
        help="The path to output data file.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="The number of instructions to generate.",
    )
    parser.add_argument(
        "--template_name",
        default="default",
        help="Name of the template to use for in-context learning.",
    )
    parser.add_argument(
        "--use_tgi",
        action="store_true",  # default False
        help="Whether or not to use text-generation inference. In this case you should have your HF_TOKEN and API_URL stored as env variables.",
    )
    parser.add_argument(
        "--keep_programming",
        action="store_true",  # default False
        help="Whether or not to keep programming tasks.",
    )
    parser.add_argument(
        "--format", choices=[2, 3], type=int, help="biprompt or triprompt."
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
        "--num_prompt_synthetic_instructions",
        type=int,
        default=2,
        help="The number of synthetic (model-generated instructions) to use in the prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=4096,
        type=int,
        help="The max_new_tokens parameter of the generate function. It is the maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        default=0.2,
        type=float,
        help="The temperature of the generation.",
    )
    parser.add_argument(
        "--top_p",
        default=0.9,
        type=float,
        help="The `top_p` parameter for the generation.",
    )
    parser.add_argument(
        "--stop_words",
        default=["\n20", "20.", "20 ."],
        nargs="+",
        help="The `stop_words` that are considered during the generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="The beam size used during the generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="The repetition penalty parameter to use for the generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="The similarity threshold for filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def parse_args_for_post_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data. The keys of the dictionary should be `instruction`, `input` and `output`.",
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        default="data/output.jsonl",
        help="The path to data we want to post-process.",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        required=True,
        default="data/output.jsonl",
        help="Path where we want to store the processed data.",
    )
    parser.add_argument(
        "--template_name",
        default="default",
        help="Name of the template to use for in-context learning.",
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
        "--num_trials",
        type=int,
        default=8,
        help="The number of trials.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=4096,
        type=int,
        help="The max_new_tokens parameter of the generate function. It is the maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        default=0.2,
        type=float,
        help="The temperature of the generation.",
    )
    parser.add_argument(
        "--top_p",
        default=0.9,
        type=float,
        help="The `top_p` parameter for the generation.",
    )
    parser.add_argument(
        "--stop_words",
        default=["\n20", "20.", "20 ."],
        nargs="+",
        help="The `stop_words` that are considered during the generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="The beam size used during the generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="The repetition penalty parameter to use for the generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="The similarity threshold for filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()

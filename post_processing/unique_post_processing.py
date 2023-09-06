import os
import json
import argparse
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="data/bi/starcoder_generations.jsonl",
        help="The path to the data to process.",
    )
    parser.add_argument(
        "--output_data_path",
        type=str,
        default="data/bi/starcoder_generations_processed.jsonl",
        help="The path to where to store the processed data.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold from which 2 instructions are considered too similar to each other",
    )
    return parser.parse_args()


similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    args = parse_args()
    machine_instructions = []
    if os.path.exists(args.input_data_path):
        with open(args.input_data_path, "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info)
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    embeddings = []
    for instruction in machine_instructions:
        embedding = similarity_model.encode(
            instruction["instruction"], convert_to_tensor=True
        )
        embeddings.append(embedding)

    indices_to_keep = []
    for i in tqdm(range(len(machine_instructions))):
        max_similarity_score = float("-inf")
        for j in range(i):
            max_similarity_score = max(
                max_similarity_score,
                util.pytorch_cos_sim(embeddings[i], embeddings[j]).item(),
            )
        if max_similarity_score < args.threshold:
            indices_to_keep.append(i)

    print(
        "We keep "
        + str(len(indices_to_keep))
        + " instructions, which amount for "
        + str(100 * len(indices_to_keep) / len(machine_instructions))
        + "% of the machine generated instructions."
    )

    with open(args.output_data_path, "a") as fout:
        for i in indices_to_keep:
            fout.write(json.dumps(machine_instructions[i]) + "\n")

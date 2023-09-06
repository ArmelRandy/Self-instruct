import re
import string

BLACKLIST = [
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


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def post_process_response(
    num_prompt_instructions, response, template, format, keep_programming=False
):
    """
    The function takes the first element of the list. And divides the completion ("text") into 1 or more instructions. WHY?
    The model is prompted with 1. ... n. and will complete n+1. ... m. so we try to retrieve the m-n new instructions.

    Args :
        num_prompt_instructions : int -> number of instructions in the prompt in-context (number of few-shot examples).
        response : str : -> string to process (prompt + model's generation).
        template : Template : -> template used for the generation
        keep_programming : bool -> whether or not to filter out programming instructions.
    """
    INSTRUCTION, INPUT, OUTPUT = (
        template.instruction_token,
        template.input_token,
        template.output_token,
    )
    if response is None:
        return []
    end_of_prompt = response.find(f"{num_prompt_instructions+1}. {INSTRUCTION}:")
    if end_of_prompt >= 0:
        raw_instructions = response[end_of_prompt:]
    else:
        raw_instructions = response

    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1:
            # Skip the last instruction as it is likely to be cropped
            continue
        if format == 3:
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
        elif format == 2:
            splitted_data = re.split(
                f"{idx+num_prompt_instructions+1}\.\s+({INSTRUCTION}|{OUTPUT}):", inst
            )
            # index 0 : everything that comes before x. Instruction
            # index 1 : x. Instruction
            # index 2 : the instruction
            # index 3 : x. Output
            # index 4 : the output + the rest of the world before (x+1) Instruction.
            if len(splitted_data) != 5:
                continue
            else:
                inst = splitted_data[2].strip()
                input = ""
                output = splitted_data[4].strip()
        else:
            ValueError(
                "The format should be either 2 (instruction and output) or 3 (instruction, input and output)."
            )
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in BLACKLIST):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if not keep_programming:
            if inst.startswith("Write a program") or inst.startswith(
                "Write a function"
            ):
                continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    
    return instructions

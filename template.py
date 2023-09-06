from dataclasses import dataclass


@dataclass
class Template:
    """
    Defines the prompting format used to generate instructions as well as their associated input and output. We advocate for 3 tokens, here is an example
    Instruction :
    {instruction}

    Input :
    {input}

    Output :
    {output}
    """

    instruction_token: str = "Instruction"
    input_token: str = "Input"
    output_token: str = "Output"

    def get_triprompt(self, example, prefix="") -> str:
        """
        takes as input a dictionary, i.e. a seed example with an instruction, its input and its output.
        -> instruction :
        can you debug the following function?
        -> input :
        def fibonacci(n):
            if n <= 1 :
                return n
            else :
                return fibonacci(n-1)+fibonacci(n-1)
        -> output :
        sure, here is the correct implementation of your function
        def fibonacci(n):
            if n <= 1 :
                return n
            else :
                return fibonacci(n-1)+fibonacci(n-2)
        """
        input_ = "" if example["input"] == "<noinput>" else example["input"]
        prompt = f"{prefix}{self.instruction_token}:\n{example['instruction']}\n\n{prefix}{self.input_token}:\n{input_}\n\n{prefix}{self.output_token}:\n{example['output']}"
        return prompt.strip()

    def get_biprompt(self, example, prefix="") -> str:
        """
        takes as input a dictionary, i.e. a seed example with an instruction, its input and its output.
        -> instruction :
        write a function which takes as input a list arr and return its maximum.
        -> output:
        def maximum(arr):
            return max(arr)
        """
        input_ = "" if example["input"] == "<noinput>" else example["input"]
        if len(input_) != 0:
            prompt = f"{prefix}{self.instruction_token}:\n{example['instruction']}\n{input_}\n\n{prefix}{self.output_token}:\n{example['output']}"
        else:
            prompt = f"{prefix}{self.instruction_token}:\n{example['instruction']}\n\n{prefix}{self.output_token}:\n{example['output']}"
        return prompt.strip()

    def get_inverse_biprompt(self, example, prefix="") -> str:
        """
        takes as input a dictionary, i.e. a seed example with an instruction, its input and its output.
        -> instruction :
        write a function which takes as input a list arr and return its maximum.
        -> output:
        def maximum(arr):
            return max(arr)
        """
        input_ = "" if example["input"] == "<noinput>" else example["input"]
        if len(input_) != 0:
            prompt = f"{prefix}{self.output_token}:\n{example['output']}\n\n{prefix}{self.instruction_token}:\n{example['instruction']}\n{input_}"
        else:
            prompt = f"{prefix}{self.output_token}:\n{example['output']}\n\n{prefix}{self.instruction_token}:\n{example['instruction']}"
        return prompt.strip()

    def copy(self):
        return Template(
            instruction_token=self.instruction_token,
            input_token=self.input_token,
            output_token=self.output_token,
        )

    def get_reverse(self):
        return Template(
            instruction_token=self.instruction_token,
            input_token=self.output_token,
            output_token=self.input_token,
        )


default_template = Template()

better_template = Template(
    instruction_token="Instruction", input_token="Input", output_token="Solution"
)

SUPPORTED_TEMPLATES = {"default": default_template, "better": better_template}


def get_template(template: str) -> Template:
    if template not in SUPPORTED_TEMPLATES.keys():
        raise ValueError(f"Template {template} is not supported!")
    return SUPPORTED_TEMPLATES[template].copy()

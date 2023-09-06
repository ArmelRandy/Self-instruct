from dataclasses import dataclass


@dataclass
class Template:
    """
    Defines the prompting format used to generate instructions as well as test cases. We advocate for 3 tokens, here is an example
    Code :
    {code}

    Instruction :
    {instruction}

    Test cases :
    {test case 1}
    {test case 2}
    {test case 3}
    """

    code_token: str = "Code:"
    instruction_token: str = "Instruction:"
    cases_token: str = "Test cases:"

    def get_triprompt(self, example) -> str:
        """
        takes as input a dictionary, i.e. a seed example with a code, an instruction and seed tasks.
        code :
        def maximum(arr):
            return max(arr)
        instruction :
        write a function which takes as input a list arr and return its maximum.
        test cases :
        assert maximum([1, 2, 3]) == 3
        assert maximum([1]) == 1
        """
        prompt = f"{self.code_token}\n{example['code']}\n\n{self.instruction_token}\n{example['instruction']}\n\n{self.cases_token}\n"
        return prompt.strip()
    
    def get_biprompt(self, example) -> str:
        """
        takes as input a dictionary, i.e. a seed example with a code, an instruction and seed tasks.
        code :
        def maximum(arr):
            return max(arr)
        test cases :
        assert maximum([1, 2, 3]) == 3
        assert maximum([1]) == 1
        """
        prompt = f"{self.code_token}\n{example['code']}\n\n{self.cases_token}\n"
        for case in example["cases"]:
            prompt += f"{case}\n"
        return prompt.strip()
    
    def copy(self):
        return Template(
            code_token=self.code_token,
            instruction_token=self.instruction_token,
            cases_token=self.cases_token
        )

default_template = Template()

SUPPORTED_TEMPLATES = {
    "default": default_template,
}


def get_dialogue_template(template: str) -> Template:
    if template not in SUPPORTED_TEMPLATES.keys():
        raise ValueError(f"Template {template} is not supported!")
    return SUPPORTED_TEMPLATES[template].copy()

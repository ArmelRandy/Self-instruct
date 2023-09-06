import tqdm
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset, where the dataset is a list of instructions (str)"""

    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.outputs = self.tokenizer(self.dataset, padding=True, return_tensors="pt")

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield {
                "input_ids": self.outputs.input_ids[i],
                "attention_mask": self.outputs.attention_mask[i],
                "index_prompt": torch.tensor(i, dtype=torch.int32),
            }


def make_requests(
    accelerator,
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    temperature,
    top_p,
    stop_words,
    num_beams,
    repetition_penalty,
):
    results = []
    if isinstance(prompts, list):
        pass
    else:
        # single prompt, i.e str
        prompts = [prompts]
    tokenized_dataset = TokenizedDataset(tokenizer=tokenizer, dataset=prompts)
    dataloader = DataLoader(tokenized_dataset, batch_size=1)
    dataloader = accelerator.prepare(dataloader)
    for step, batch in tqdm.tqdm(enumerate(dataloader)):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            index_prompt = batch["index_prompt"]
            stopping_criteria = StoppingCriteriaList(
                [EndOfFunctionCriteria(attention_mask.sum(), stop_words, tokenizer)]
            )
            response = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )
            padded_indices = accelerator.pad_across_processes(
                index_prompt, dim=1, pad_index=tokenizer.pad_token_id
            )
            padded_responses = accelerator.pad_across_processes(
                response, dim=1, pad_index=tokenizer.pad_token_id
            )
            indices = accelerator.gather(padded_indices)
            answers = accelerator.gather(padded_responses)
            for i in range(accelerator.num_processes):
                results.append(
                    {
                        "prompt": prompts[indices[i]],
                        "answer": tokenizer.decode(
                            answers[i], skip_special_tokens=True
                        ),
                    }
                )
    return results

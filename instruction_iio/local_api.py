import tqdm
from datetime import datetime
 
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
    """Tokenize and preprocess the dataset, where the dataset is a list of instructions (str)
    """
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.outputs = self.tokenizer(self.dataset, padding=True, return_tensors="pt")
    def __iter__(self):
        for i in range(len(self.dataset)):
            yield {
                "input_ids" : self.outputs.input_ids[i],
                "attention_mask" : self.outputs.attention_mask[i],
                "index_prompt" : torch.tensor(i, dtype=torch.int8)
            }

def make_requests(
        accelerator,
        model,
        tokenizer, 
        prompts, 
        max_length, 
        temperature, 
        top_p, 
        stop_words, 
        num_return_sequences, 
        num_beams,
        repetition_penalty 
    ):
    results = []
    if isinstance(prompts, list):
        pass
    else :
        # single prompt, i.e str
        prompts = [prompts]
    state = accelerator.state
    tokenized_dataset = TokenizedDataset(tokenizer=tokenizer, dataset=prompts) 
    dataloader = DataLoader(tokenized_dataset, batch_size=accelerator.num_processes)
    for step, batch_ in tqdm.tqdm(enumerate(dataloader)):
        with state.split_between_processes(batch_) as batch:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(state.device)
                attention_mask = batch["attention_mask"]
                index_prompt = batch["index_prompt"]
                stopping_criteria = StoppingCriteriaList([EndOfFunctionCriteria(attention_mask.sum(), stop_words, tokenizer)])
                try :
                    response = accelerator.unwrap_model(model).generate(
                        input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        num_return_sequences=num_return_sequences,
                        top_p=top_p,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, 
                        stopping_criteria=stopping_criteria
                    )
                except RuntimeError :
                    print("An error occured, please check the size of input_ids compare to the value of max_length")
                    continue
                response = accelerator.pad_across_processes(
                    response, dim=1, pad_index=tokenizer.pad_token_id
                )
                response = accelerator.gather(response)
                outputs = []
                for response_i in response :
                    outputs.append(tokenizer.decode(response_i, skip_special_tokens=True))
                data = {
                    "prompt": prompts[index_prompt],
                    "response": 
                    [
                        {
                            "text" : outputs[i],
                            "index": i,
                            "finish_reason" : "stop" if (len(response[i]) < max_length) else "length"
                        } 
                        for i in range(len(outputs))
                        
                    ] if outputs else None,
                    "created_at": str(datetime.now()),
                }
                results.append(data)
    return results

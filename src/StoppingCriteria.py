from transformers import StoppingCriteria
import torch
from typing import Union, List

class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, stop_sequences, tokenizer):
        self.tokenized_stop_sequences = [tokenizer(stop_seq, add_special_tokens=False)["input_ids"] for stop_seq in stop_sequences]
        self.tokenizer = tokenizer
        print("stop sequence ids", self.tokenized_stop_sequences)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        
        for seq in input_ids:
            seq = seq.tolist()  # Ensure the tensor is moved to CPU and converted to a list
            
            

            for tokenized_stop_seq in self.tokenized_stop_sequences:
                
                if len(seq) >= len(tokenized_stop_seq) and seq[-len(tokenized_stop_seq):] == tokenized_stop_seq:
                    print("stop")
                    return True
                
        
        return False

class BatchStoppingCriteria(StoppingCriteria):

    def __init__(self, eos_token_ids: Union[int, List[int]]):
        
        if isinstance(eos_token_ids, int):
            self.eos_token_ids = [eos_token_ids]
        else:
            self.eos_token_ids = eos_token_ids


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for seq in input_ids:
            seq = seq.tolist()
            if seq[-1] in self.eos_token_ids:
                return True
        return False
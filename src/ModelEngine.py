import torch
import math
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import pandas as pd
from .utils.Timer import Timer
import traceback   
import os
from time import time
from src.StoppingCriteria import BatchStoppingCriteria


from queue import Queue
from typing import List, Tuple, Generator
import torch


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class ModelEngine:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.dry_run = os.getenv("DRY_RUN") == "True"   
        self.model = model
        
        self.tokenizer = tokenizer
        self.queue = Queue()

    def add_to_queue(self, jobs: List[Tuple[any,str]]):
        for id, text, generation_config in jobs:
            self.queue.put((id, text, generation_config))

    def generate(self, batch_size: int) -> Generator[Tuple[any, str], None, None]:
        print("RUNNING INFERENCE ...", "\n", "==="*20, "\n")
        i = 0
        timer = Timer()
        for id, text in self._generate(batch_size):
            timer.add_time()
            i += 1
            if text is None:
                print(f"Row {id} complete","\n", "Error during generation", "\n", "==="*5)
            else:
                print(f"Row {id} complete","\n", "Output:", "\n", "==="*5, "\n", text, "\n", "==="*5)
            
            print(f"Iteration {i} complete")
            timer.print_current_duration()
            yield (id, text)
        print("INFERENCE COMPLETE", "\n", "==="*20, "\n", "==="*20, "\n", "==="*20)
        timer.print_overview()
        

    def _generate(self, batch_size: int) -> Generator[Tuple[any,str], None, None]:
        # Helper function to pad sequences to the same length
        def pad_sequences(sequences, pad_token_id, max_length):
            padded_sequences = []
            for seq in sequences:
                # Calculate the amount of padding needed
                padding_length = max_length - len(seq)
                if padding_length > 0:
                    # Create a padding tensor
                    padding_tensor = torch.tensor([pad_token_id] * padding_length, dtype=seq.dtype, device=seq.device)
                    # Concatenate the sequence with the padding tensor
                    padded_seq = torch.cat((seq, padding_tensor))
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            
            return padded_sequences
        

        pad_token_id = self.tokenizer.pad_token_id
        eos_token_ids = self.tokenizer.eos_token_id
        
        
        if not self.dry_run:
            model_generation_config = self.model.generation_config
            if model_generation_config is not None:
                # overwrite with generation config eos token id as it is the prevailing one 
                eos_token_ids = model_generation_config.eos_token_id 
        
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        
        
        
        batch_stopping_criterion = BatchStoppingCriteria(model_generation_config.eos_token_ids) if  batch_size > 1 else None

        while not self.queue.empty():
            batch_inputs, original_input_lenghts, batch_ids, generation_config = self.get_inputs(batch_size)
            
            
            while batch_inputs:
                try:
                    # Determine the max length for current batch
                    max_length = max(len(tokens) for tokens in batch_inputs)
                    
                    # Pad sequences to the same length
                    padded_inputs = pad_sequences(batch_inputs, pad_token_id, max_length)
                    # count padding tokens at the start of each input 
                    max_new_tokens =  generation_config.get("max_new_tokens", None)
                    
                    if batch_stopping_criterion is not None:
                        if "stopping_criteria" in generation_config and isinstance(generation_config["stopping_criteria"], list):
                            generation_config["stopping_criteria"] += [batch_stopping_criterion]
                            
                        else:
                            generation_config["stopping_criteria"] =  [batch_stopping_criterion]
                           

                    # Convert to tensor and run model
                    if not self.dry_run:
                        input_tensor = torch.stack(padded_inputs).to(self.model.device)
                        attention_mask = (input_tensor != pad_token_id).long()
                        outputs = self.model.generate(input_tensor, attention_mask=attention_mask, **generation_config)
                        print("Generation complete")
                    else:
                        outputs = torch.tensor([padded_inputs[i] + [eos_token_ids][0] for i in range(len(batch_inputs))])

                        print("Dry run, skipping model inference")

                    completed_indices = []
                    completed_texts = []
                    

                    # Check for completion
                    for i, output in enumerate(outputs):
                        existing_eos_token_ids = [ eos_token_id for eos_token_id in eos_token_ids if eos_token_id in output[original_input_lenghts[i]:]] 
                        
                        if len(existing_eos_token_ids) > 0 :
                            eos_position = output[len(padded_inputs[i]):].tolist().index(existing_eos_token_ids[0]) # Find first EOS token in the output
                            eos_position += len(padded_inputs[i])
                            # Decode up from the original inputs end up to eos position, works because attention mask is set and padding tokens are skipped when decoding 
                            generated_text = self.tokenizer.decode(output[original_input_lenghts[i]:eos_position + 1], skip_special_tokens=True) 
                            completed_texts.append((batch_ids[i],generated_text))
                            completed_indices.append(i)
                        elif max_new_tokens is not None and len(output) > original_input_lenghts[i] + max_new_tokens:
                            
                            eos_position = original_input_lenghts[i] + generation_config["max_new_tokens"]
                            generated_text = self.tokenizer.decode(output[original_input_lenghts[i]:eos_position + 1], skip_special_tokens=True) 
                            completed_texts.append((batch_ids[i],generated_text))
                            completed_indices.append(i)  
                        elif "stopping_criteria" in generation_config:
                            
                            for criterion in generation_config["stopping_criteria"]:
                                print("output",output)
                                
                                if criterion([output], []):
                                    print("A custom stopping criterion was responsible for the termination of the generation process")
                                    generated_text = self.tokenizer.decode(output[original_input_lenghts[i]:], skip_special_tokens=True)
                                    completed_texts.append((batch_ids[i], generated_text))
                                    completed_indices.append(i)
                                    break

                        else:
                            # set the outputs as the new batch_inputs 
                            batch_inputs[i] = output

                        
                    # Yield completed texts
                    for id, text in completed_texts:
                        print(id, text)
                        yield (id, text)

                    # Remove completed jobs from batch
                    for i in sorted(completed_indices, reverse=True):
                        batch_inputs.pop(i)
                        batch_ids.pop(i)

                    if len(batch_inputs) < batch_size and not self.queue.empty():
                        batch_inputs,original_input_lenghts,  batch_ids, generation_config = self.get_inputs(batch_size, batch_inputs,batch_ids,original_lenghts=original_input_lenghts, current_config=generation_config)
                        

                    if not self.dry_run:
                        # clear memory
                        del input_tensor
                        del attention_mask
                        del outputs
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error during generation: {str(e)}") 

                    traceback.print_exc()
                    for id in batch_ids:
                        yield (id, "") # Return empty string for failed jobs to let the caller handle it

                    for i in range(len(batch_inputs)):
                        batch_inputs.pop()
                        batch_ids.pop() 

    def get_inputs(self, batch_size: int, current_inputs : list = [],current_ids : list = [], original_lenghts = [], current_config: dict = None)  -> Tuple[List[List[int]], List[any], dict]:
        batch_inputs = current_inputs.copy()
        batch_ids = current_ids.copy()
        generation_config = current_config
        queue_buffer = []
        original_input_lengths = original_lenghts

        while not self.queue.empty() and len(batch_inputs) < batch_size:
            id, text, gen_config = self.queue.get()
            if not generation_config:
                generation_config = gen_config
            else:
                if gen_config != generation_config:
                    queue_buffer.append((id, text, gen_config))
                    continue
            
            tokens = self.tokenizer.encode(text, return_tensors="pt")[0]
            max_length = getattr(self.model.config, 'max_position_embeddings', None)
            print("max length", max_length)


            if max_length is None or len(tokens) <= max_length:
                batch_inputs.append(tokens)
                batch_ids.append(id)
            else: print(f"Skipping sample with id {id}: {len(tokens)} input Tokens exceed max length of {max_length} Tokens")
            

        for id, text, gen_config in queue_buffer:
            self.queue.put((id, text, gen_config))

        new_added = len(batch_inputs) - len(current_inputs)
        for i in range(len(batch_inputs) - new_added, len(batch_inputs)):
            if len(original_input_lengths) <= i:
                original_input_lengths.append(len(batch_inputs[i]))
            else:
                original_input_lengths[i] = len(batch_inputs[i])
        
        return batch_inputs, original_input_lengths, batch_ids, generation_config
    
    


    """
    def _generate(self,df : pd.DataFrame,  input_column : str , output_column : str, generation_config = None, batch_size = 1):
        
        
        
        for i in range(0, len(df), batch_size):
            self.tracker = Timer()
            end_index = min(i + batch_size, len(df))  # Ensure we do not go beyond the DataFrame
            batch = df[i:end_index][input_column].tolist()
            with self.tracker: # track generation time
                print(f"Running batch {i} to {end_index}" if batch_size > 1 else f"Running row {i}")
                outputs, total_tokens = self._generate_batch(batch, generation_config)
                if outputs is None:
                    print("Error during generation, skipping batch")
                    continue
                for j, decoded in enumerate(outputs):
                    print(f"Row {i+j} complete","\n", "Output:", "\n", "==="*5, "\n", decoded, "\n", "==="*5)
                    df.at[i+j, output_column] = decoded    
            print(f"Iteration { (i + batch_size) // batch_size } of {math.ceil(len(df) / batch_size )}  complete")
            if self.dry_run:
                continue
            self.print_reports(total_tokens)
            print("="*20, "\n", "="*50)
                    
        print("INFERENCE COMPLETE", "\n", "==="*20, "\n", "==="*20, "\n", "==="*20)
        return self
    
    def _generate_batch(self, inputs, generation_config):
        try:
            tokenized = self.tokenizer(inputs, return_tensors="pt", padding=True)
            input_length = tokenized["input_ids"].shape[-1] # input length is the same for all rows in the batch
            encodeds = tokenized["input_ids"]
            if not self.dry_run:
                model_inputs = encodeds.to(self.model.device)
                model_outputs = self.model.generate(model_inputs, **generation_config)
                decodeds = [self.tokenizer.decode(output[input_length:], skip_special_tokens=True) for output in model_outputs]
                # calculate the number of generated tokens 
                num_tokens = [len(output[input_length:]) for output in model_outputs]
                total_tokens = sum(num_tokens)
                
            if self.dry_run:
                return [""] * len(inputs), 0
            return decodeds, total_tokens
        except Exception as e:
            print(f"Error during generation: {str(e)}")   
            traceback.print_exc()
            return None, 0
    """
    
        
        
               
        







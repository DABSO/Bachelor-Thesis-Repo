import pandas as pd
from src.output_parsers.base import Validator
from src.preprocessors import Preprocessor
from src.InputFormatter import InputFormatter

class OutputHandler:
    input_df = None
    target_df = None

    id_column = None
    input_column = None

    validation_function = None

    retry_limit = 3

    RETRY = "retry"
    APPEND = "append"

    LIMIT_REACHED = "limit_reached"
    

    def __init__(self, input_df, id_column,  input_column, validation_function,  retry_limit = 3,  ):
        self.input_df = input_df
        self.validation_function = validation_function
        self.retry_counter = {id: 0 for id in input_df[id_column].tolist()}
        self.retry_limit = retry_limit
        self.id_column = id_column
        self.input_column = input_column


    def determine_action(self, output):
        # determines if a message should be appended to the output or if the generation should be retried entirely
        #check if the output contains a high percentage of same words 
        split_output = output.split()
        unique_words = set(split_output)
        if len(unique_words) < 0.3 * len(split_output):
            return self.RETRY
        else:
            return self.APPEND
        
    def get_retry_messages(self, id) -> list[dict]:
        # get the row of the input df that corresponds to the id
        row = self.input_df[self.input_df[self.id_column] == id]
        messages = row.iloc[0][self.input_column]
        return messages
    
    def get_append_messages(self, id, output, error_message = None):
        
        messages = self.get_retry_messages(id)
        messages.append({"role": "assistant", "content": output })
        if error_message:
            messages.append({"role": "user", "content": error_message})
        return messages
    
    def process_output(self, id, output, valid_callback : callable, invalid_callback : callable, retry_callback : callable):
        print("process output id",id)

        
        if self.validation_function(output):
            return valid_callback(id, output)
        else:
            # update the retry counter
            self.retry_counter[id] += 1
            
            if self.retry_counter[id] >= self.retry_limit:
                print("retry limit reached", id)
                return invalid_callback(id, output)
            else:
                # check if the retry limit has been reached
                action = self.determine_action(output)
                if action == self.RETRY:
                    return retry_callback(id, self.get_retry_messages(id))

                else:
                    return retry_callback(id, self.get_append_messages(id, output, error_message = "Please Format your response as specified in the prompt!"))
                    
            

        
    


    
         
import os
from .ModelEngine import ModelEngine

from .preprocessors import Preprocessor
import pandas as pd
from transformers import  AutoModelForCausalLM
from .OutputHandler import OutputHandler
from .InputFormatter import InputFormatter
from .filters import Filter
import torch

class InferencePipeline:

      
    dataset_path : str = None # full path to the dataset

    input_column : str= None  # name of the column that contains the input messages
    output_column : str= None # the output column where the predictions will be stored
    id_column : str = None # the column that contains the unique id for each row

    preprocessor : Preprocessor = None # the preprocessor to be used for the dataset, stores the tokenizer, applies the prompt and chat template 
    input_formatter : InputFormatter = None # the input formatter to be used for the dataset, stores the tokenizer, converts the messages to the correct input format for the model

    model_path : str= None  # the full path to the model to be used / hf repo id if not local
    model_kwargs :dict = None # the model kwargs to be used when loading the model

    generation_config : dict = None # the generation config to be used when running the model
    retry_generation_config : dict = None # the generation config to be used when retrying the generation

    dry_run :bool = False   # if set to true, the model will not be loaded and no predictions will be made, for testing purposes only
    local_files_only: bool = True  # if set to true, the model will only be loaded from local files, if set to false, the model will be loaded from the huggingface repo

    def __init__(self, 
                 dataset_path : str, 
                 output_path : str,
                 output_column : str, 
                 input_column : str,
                 id_column : str,
                 preprocessor : Preprocessor,
                 validation_function : callable,
                 
                 model_path, 
                 model_kwargs : dict = {}, 
                 generation_config : dict = {},
                 retry_generation_config : dict = {},   
                 local_files_only: bool = True,
                 filters : list[Filter] = [],
                 ):
        # administrative settings
        self.local_files_only= local_files_only
        self.dry_run = os.getenv("DRY_RUN") == "True"
        # dynamic settings 
        self.dataset_path = dataset_path
        self.output_path = output_path

        self.output_column = output_column
        self.input_column = input_column
        self.id_column = id_column


        self.preprocessor = preprocessor
        self.validation_function = validation_function
        self.filters = filters
        

        
        self.model_path = model_path
        self.model_kwargs = model_kwargs
        self.generation_config = generation_config    
        self.retry_generation_config = retry_generation_config  
        self.input_formatter = InputFormatter(self.preprocessor.tokenizer)
        

    def load_model(self):
        default_kwargs = {}
        if not "device_map" in self.model_kwargs:
            default_kwargs["device_map"] = "auto"
        if self.local_files_only:
            default_kwargs["local_files_only"] = True
        else:
            default_kwargs["token"] = os.getenv("HUGGINGFACE_TOKEN")

        kv_caching = False
        if "kv_caching_support" in self.model_kwargs:
            kv_caching = self.model_kwargs["kv_caching_support"]
            self.model_kwargs.pop("kv_caching_support")

            

        if not self.dry_run:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **{**default_kwargs, **self.model_kwargs})
            self.print_gpu_utilization()
            print("Using model with config", model.config)
            if kv_caching:
                print("using kv cache")
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

                self.print_gpu_utilization()
            
            return model
        else:
            return None 
        
    def print_gpu_utilization(self):
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            print(f"GPU {i}: {device.name}, Memory: {memory:.1f}GB / {device.total_memory/1024**3:.1f}GB")

    
    def load_input_dataset(self):
        df = pd.read_json(self.dataset_path, orient='records', lines=True, dtype=False)
        print("Loaded dataset: ", self.dataset_path)
        print("Columns: ", df.columns)
        # load the output dataset or create a new one if it does not exist
        assert self.input_column not in df.columns, f"{self.input_column} already in dataframe columns"

        return df
    
    def load_output_dataset(self):
        output_df = None
        if os.path.exists(self.output_path):
            output_df = pd.read_json(self.output_path, orient='records', lines=True, dtype=False, convert_axes=False)
        else:
            output_df = pd.DataFrame()

        return output_df

        
    def store_output_in_df(self,df, id, output):
        df.loc[df[self.id_column] == id, self.output_column] = output
        self.store_data(df)
        return df
        
    

    def generate(self, input_df, output_df, output_handler : OutputHandler, batch_size : int = 1):
        model = self.load_model()
        engine = ModelEngine(model, self.preprocessor.tokenizer)
        
        inputs = zip(input_df[self.id_column], input_df[self.input_column])
        inputs = [(id, self.input_formatter.format(input_text), self.generation_config) for id, input_text in inputs]
        engine.add_to_queue(inputs)
        for id, output in engine.generate(batch_size=batch_size):
            output_handler.process_output(
                id=id,
                output=output,
                
                valid_callback=lambda id, output: self.store_output_in_df(output_df, id, output),
                invalid_callback=lambda id, output: self.store_output_in_df(output_df, id, None),
                retry_callback=lambda id, output: engine.add_to_queue([(id, self.input_formatter.format(output), self.retry_generation_config)])
                )
            self.print_gpu_utilization()
        return output_df
    
    def store_data(self, df: pd.DataFrame):
        """
        Stores the generated data in the dataset, replaces the output column if it exists,
        and saves the dataset to the file.
        """
        try:
            # Ensure the output column exists in the input dataframe
            if self.output_column not in df.columns:
                raise ValueError(f"The output column '{self.output_column}' is not present in the input dataframe.")

            # Load the current version of the file if it exists
            if os.path.exists(self.output_path):
                print("Loading existing dataset")
                existing_df = self.load_output_dataset()

                # Ensure the dataframes have the same number of rows
                if len(existing_df) != len(df):
                    raise ValueError("The input dataframe has a different number of rows than the existing dataset.")

                # Update or add the output column
                existing_df[self.output_column] = df[self.output_column]
            else:
                print("No existing dataset found. Creating a new one.")
                existing_df = df

            # Save the updated dataframe to the output file
            print(f"Saving to {self.output_path}")
            existing_df.to_json(self.output_path, lines=True, orient='records')
            print("Done!")

        except Exception as e:
            print(f"An error occurred during data storing: {str(e)}")
            # Optionally, re-raise the exception if you want it to propagate
            # raise
            raise e

    
    def run(self, batch_size :int = 1):
        assert batch_size > 0, "Batch size must be greater than 0"
        df = self.load_input_dataset()

        output_df = self.load_output_dataset()
        input_df, output_df = self.preprocessor.run(df, output_df)
        for filter in self.filters: 
            input_df = filter.apply(input_df)
            print("Instances after filtering",len(input_df))
        output_handler = OutputHandler(input_df, self.id_column, self.input_column,self.validation_function )
        output_df = self.generate( input_df, output_df, output_handler, batch_size=batch_size)
        self.store_data(output_df)
from typing import Any
from config.model_config import config as model_configs
from config.prompt_config import config as prompt_configs
from config.dataset_config import config as dataset_configs
import pandas as pd 
import os
from .preprocessors import Preprocessor
from transformers import AutoTokenizer
from os import path
from src.filters import ValidFilter
from src.StoppingCriteria import StopSequenceCriteria

class ConfigLoader:
    model_config : dict
    prompt_config : dict
    dataset_config : dict


    local_files_only = True

    dataset_filename : str
    model_name :str= None
    prompt_name : str = None

    input_dataset_path : str = None
    
    model_path : str = None
    
    def __init__(self, model_name : str, prompt_name : str, dataset_filename : str, local_files_only = True, only_failed = False):
        self.local_files_only = local_files_only
        self.validate_config(model_name, prompt_name)
        self.model_config = ConfigLoader.get_model_config(model_name)
        self.prompt_config = ConfigLoader.get_prompt_config(prompt_name)
        self.dataset_config = ConfigLoader.get_dataset_config(dataset_filename)
        self.dataset_filename = dataset_filename
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.only_failed = only_failed
        self.validate_environment()
        
    def load_inference_config(self):
        self.input_dataset_path = self.build_input_dataset_path(self.dataset_filename)
        output_dataset_path = ConfigLoader.build_model_output_dataset_path(self.dataset_filename, self.prompt_name)
        self.model_path = self.build_model_path()    

        tokenizer = self.load_tokenizer(self.model_path)
        input_column = "model_input" # must not be present in the dataframe
        return {
            "dataset_path": self.input_dataset_path,
            "output_path": output_dataset_path,
            "preprocessor": self.load_preprocessor(tokenizer, input_column),
            "model_path": self.model_path,
            "model_kwargs": self.model_config.get("model_kwargs", {}),
            "generation_config": self.load_generation_config(tokenizer),
            "retry_generation_config": self.load_retry_generation_config(tokenizer),
            "input_column": input_column,
            "output_column": self.model_config["output_column"],
            "id_column": self.dataset_config.get("id_column", None),
            "validation_function": self.prompt_config.get("validation_function"),
            "filters": self.load_filters()
        }


    
    def get_num_examples(self,dataset_path : str):
        df = pd.read_json(dataset_path, orient='records', lines=True)
        return len(df)
        
    
    def load_slurm_config(self):
        slurm_model_config = self.model_config.get("slurm_config", {})

        if "seconds-per-example" in slurm_model_config:
            slurm_model_config["time"] = str(int((1.5 * float(slurm_model_config["seconds-per-example"]) * self.get_num_examples(self.input_dataset_path)) // 60))

        return {**{
            "job-name": "inference_"+self.model_name+"_"+self.prompt_name+"_"+self.dataset_filename.replace(".", "-"),
            "output": f"logs/{self.model_name}/{self.prompt_name}_{self.dataset_filename.replace('.', '-')}.out.log",
            },
            **slurm_model_config
            }
        
    
    @staticmethod
    def load_available_models():
        return list(model_configs.keys())
    
    @staticmethod
    def load_available_datasets():
        datasets = [filename.split(".")[0] for filename in os.listdir(ConfigLoader.build_input_dataset_path())]
        
        return datasets
    
    @staticmethod
    def load_available_prompts():
        return list(prompt_configs.keys())

    @staticmethod
    def get_model_config(model_name : str):
        return model_configs[model_name]
    
    @staticmethod
    def get_prompt_config( prompt_name : str):
        return prompt_configs[prompt_name]

    def get_dataset_config(dataset_name : str):
        return dataset_configs[dataset_name]


    def validate_config(self, model : str, prompt: str):
        def validate_model_config(model):
            assert model in model_configs, f"Model not found in model configs: {model}"
            assert "output_column" in model_configs[model], f"Output column not specified for model: {model}"
            assert "model_repo" in model_configs[model], f"Model repo not specified for model: {model}"
            assert "preprocessor" in model_configs[model], f"Preprocessor not specified for model: {model}"
            assert "generation_config" in model_configs[model], f"Generation config not specified for model: {model}"
        validate_model_config(model)

        def validate_prompt_config(prompt):
            assert prompt in prompt_configs, f"Prompt not found in prompt configs: {prompt}"
            assert "prompt_files" in prompt_configs[prompt], f"Prompt files not specified for prompt: {prompt}"
            for prompt_file in prompt_configs[prompt]["prompt_files"]:
                assert "file" in prompt_file, f"File not specified for prompt file: {prompt_file}"
                assert "role" in prompt_file, f"Role not specified for prompt file: {prompt_file}"
        validate_prompt_config(prompt)

    def validate_environment(self):
        assert os.getenv("MODEL_DIR") is not None, "MODEL_DIR environment variable is not set"
        assert os.getenv("DATASET_DIR") is not None, "DATASET_DIR environment variable is not set"
        assert os.getenv("PROMPT_DIR") is not None, "PROMPT_DIR environment variable is not set" 
        if not self.local_files_only:
            assert os.getenv("HUGGINGFACE_TOKEN") is not None, "HUGGINGFACE_TOKEN environment variable is not set"

    def validate_files_exist(self):
        assert path.exists(self.dataset_path), f"Dataset file does not exist: {self.dataset_path}"
        assert path.exists(self.prompt_path), f"Prompt file does not exist: {self.prompt_path}"
        if not self.dry_run:
            assert path.exists(self.model_path), f"Model does not exist: {self.model_path}"

    def build_model_path(self):
        if self.local_files_only: 
            model_repo = self.model_config["model_repo"]
            return os.path.join(os.getenv("MODEL_DIR") ,model_repo)
        else:
            # format is correct if the download_model utility script is used
            model_repo =  self.model_config["model_repo"].split("/")[0] + "/" + self.model_config["model_repo"].split("/")[1]
            return model_repo
        
    def build_classifier_path(model_type : str, train_dataset_name : str, feature_names : list[str]):

        return os.getenv("CLASSIFIER_DIR") + "/"+ model_type + "_"+ train_dataset_name + "_" + "-".join(feature_names) + ".joblib"
    
    def get_classifier_name(model_type : str, train_dataset_name : str, feature_names : list[str]):
        return model_type + "_"+ train_dataset_name + "_" + "-".join(feature_names) 
    @staticmethod
    def build_input_dataset_path(dataset_name : str = None):
        if dataset_name is None:
            joined = os.getenv("DATASET_DIR")+  "/input/"
        else:
            joined = os.getenv("DATASET_DIR")+  "/input/" + dataset_name + ".json"
        return  joined
    
    @staticmethod
    def build_model_output_dataset_path( dataset_name : str, prompt_name : str = None):
        joined = os.getenv("DATASET_DIR")+  "/model_outputs/" + prompt_name + "_"+ dataset_name + ".json"
        return  joined
    
    def build_parsed_model_output_dataset_path(dataset_name : str, prompt_name : str = None):
        joined = os.getenv("DATASET_DIR")+  "/parsed_model_outputs/" + prompt_name + "_"+ dataset_name + ".json"
        return  joined
    
    @staticmethod
    def build_prediction_dataset_path( dataset_name : str, prompt_name : str = None):
        joined = os.getenv("DATASET_DIR")+  "/predictions/" + prompt_name + "_"+ dataset_name + ".json"
        return  joined

    @staticmethod
    def build_optimization_dataset_path(dataset_name : str, optimization_column : str):
        return os.getenv("DATASET_DIR")+  "/optimization/" + dataset_name + "_"+optimization_column + ".json"

    @staticmethod
    def build_evaluation_dataset_path( dataset_name : str, prompt_name : str = None):
        joined = os.getenv("DATASET_DIR")+  "/evaluations/" + prompt_name + "_"+ dataset_name + ".json"
        return  joined
    @staticmethod
    def build_aggregated_predictions_dataset_path(dataset_name):
        joined = os.getenv("DATASET_DIR") +"/aggregated_predictions/" + dataset_name + ".json"
        print(joined)
        return joined    
    @staticmethod
    def build_relabeled_dataset_path(dataset_name : str):
        joined = os.getenv("DATASET_DIR") + "/relabeled/" + dataset_name + ".json"
        print(joined)
        return joined

    @staticmethod
    def build_relabeled_evaluation_dataset_path(dataset_name : str):
        joined = os.getenv("DATASET_DIR") + "/evaluations/relabeled/" + dataset_name + ".json"
        return joined
    
    @staticmethod
    def build_fine_tuning_dataset_path(dataset_name : str, prompt_name : str):
        joined = os.getenv("DATASET_DIR") + "/fine-tuning/" + prompt_name + "_" + dataset_name + ".json"
        return joined
    
    @staticmethod
    def build_prompt_path( prompt_file : str): 
        joined = os.getenv("PROMPT_DIR") + "/" + prompt_file
        return joined

    def load_preprocessor(self, tokenizer,  input_column : str) -> Preprocessor:
        prompt_templates = []
        for prompt_file in self.prompt_config["prompt_files"]:
            prompt_template = {}
            with open(ConfigLoader.build_prompt_path(prompt_file["file"]), "r") as f:
                prompt_template["content"] = f.read()
            prompt_template["role"] = prompt_file["role"]
            prompt_template["variables"] = prompt_file["variables"] if "variables" in prompt_file else None
            prompt_templates.append(prompt_template)

        
        keep_columns = []
        if "keep_columns" in self.dataset_config:
            keep_columns = self.dataset_config["keep_columns"]

        preprocessing_steps = []
        if "preprocessing_steps" in self.prompt_config and self.prompt_config["preprocessing_steps"] is not None:
            preprocessing_steps = self.prompt_config["preprocessing_steps"]
        # instantiate the preprocessor
        return self.model_config["preprocessor"](prompt_templates,  input_column, tokenizer, keep_columns=keep_columns, preprocessing_steps=preprocessing_steps)
    
   
    
    def load_generation_config(self,tokenizer  = None, config_key="generation_config") -> dict:
        gen_config = self.model_config[config_key]
        if gen_config is None:
            return None  # Return None if the model or generation_config does not exist

        # Handle any callable in the generation config (for dynamic attributes)
        for key, value in gen_config.items():
            if callable(value):
                # Call the function if the attribute value is a callable
                gen_config[key] = value(tokenizer)
        stopping_criteria = []
        if "stop_sequences" in self.prompt_config:
            stopping_criteria.append(StopSequenceCriteria(self.prompt_config["stop_sequences"], tokenizer))
        if len(stopping_criteria) > 0:
            gen_config["stopping_criteria"] = stopping_criteria
        
        return gen_config
    
    def load_retry_generation_config(self, tokenizer = None) -> dict:
        return self.load_generation_config(tokenizer, "retry_generation_config")
      
    def load_tokenizer(self, model_path : str):
        kwargs = {}
        if self.local_files_only:
            kwargs["local_files_only"] = True
        else:
            kwargs["token"] = os.getenv("HUGGINGFACE_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer
    
    
    def load_filters(self): 
        filters = []
        if self.only_failed:
            filters.append(ValidFilter(self))
        
        if "filters" in self.prompt_config:
            for filter in self.prompt_config["filters"]:
                filters.append(filter(self))

        return filters
        
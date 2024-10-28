from src.preprocessors import MistralPreprocessor, Llama3Preprocessor
import torch

config = {
    "Mistral8x7B": {
        "groups": ["large_models" , "Mistral8x7B-G", "large_w_o_phi3", "large_w_o_qwen2", "large_w_o_llama3" ],
        "output_column": "Mistral8x7BInstructV01",
        "model_repo": "mistralai/Mixtral-8x7B-Instruct-v0.1/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
        },
        "preprocessor": MistralPreprocessor,
        "generation_config": {
            "max_new_tokens": 4096,
            "temperature": 0.1,
            "top_k": 10,
            "do_sample": True,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "_from_model_config": True
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "slurm_config": {
            "mem": "155G",
            "gpus": "a100:2",
            "cpus-per-task": "32",	
            "seconds-per-example": "20"
        }
    },
    "Llama3_70B": {
        "groups": ["large_models", "Llama3_70B-G", "large_w_o_phi3", "large_w_o_qwen2", "large_w_o_mistral"],
        "output_column": "LLama3_70BInstruct",
        "model_repo": "meta-llama/Meta-Llama-3-70B-Instruct/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/e8cf5276ae3e97cfde8a058e64a636f2cde47820",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attention_implementation": "flash_attention_2",
        },
        "preprocessor": Llama3Preprocessor,
        "generation_config": {
           
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
            "eos_token_id": lambda tokenizer: [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Assumes this token is an empty string
            ]
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
            "eos_token_id": lambda tokenizer: [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Assumes this token is an empty string
            ]
        },
        "slurm_config": {
            "mem": "155G",
            "gpus": "a100:2",
            "cpus-per-task": "32",	
            "seconds-per-example": "30"
        }
    },
    "Phi3_14B": {
        "groups": ["large_models", "Phi3_14B-G", "large_w_o_qwen2", "large_w_o_mistral", "large_w_o_llama3"],
        "output_column": "Phi3_14B",
        "model_repo": "microsoft/Phi-3-medium-128k-instruct/models--microsoft--Phi-3-medium-128k-instruct/snapshots/cae1d42b5577398fd1be9f0746052562ae552886",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "attention_implementation": "flash_attention_2",
        },
        
        "preprocessor": Llama3Preprocessor, #same preprocessing steps as Llama3
        "generation_config": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
        },
        "slurm_config": {
            "mem": "80G",
            "gpus": "a100:1",
            "cpus-per-task": "16",	
            "seconds-per-example": "30"
        }
    },
    "Qwen2_72B": {
        "groups": ["large_models", "Qwen2_72B-G", "large_w_o_phi3", "large_w_o_mistral", "large_w_o_llama3"],
        "output_column": "Qwen2_72BInstruct",
        "model_repo": "Qwen/Qwen2-72B-Instruct/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79",
        "model_kwargs": {
            "torch_dtype": torch.bfloat16,

        },
        "preprocessor": Llama3Preprocessor, #same preprocessing steps as Llama3
        "generation_config": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
        },
        "slurm_config": {
            "mem": "158G",
            "gpus": "a100:2",
            "cpus-per-task": "32",	
            "seconds-per-example": "30"
        }
    },
    "Mistral7B": {
        "groups": ["small_models", "Mistral7B-G"],
        "output_column": "Mistral7BInstructV01",
        "model_repo": "mistralai/Mistral-7B-Instruct-v0.1/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/73068f3702d050a2fd5aa2ca1e612e5036429398",
        "model_kwargs": {},
        "preprocessor": MistralPreprocessor,
        "generation_config": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
            "eos_token_id": lambda tokenizer: [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Assumes this token is an empty string
            ]
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "slurm_config": {
            "mem": "40G",
            "gpus": "a100:1",
            "cpus-per-task": "10",	
            "seconds-per-example": "20",
            "nodes": "1"
        }
    },
    "Qwen2_7B": {
        "groups": ["small_models", "Qwen2_7B-G"],
        "output_column": "Qwen2_7B_Instruct",
        "model_repo": "Qwen/Qwen2-7B-Instruct/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c",
        "model_kwargs": {},
        "preprocessor": Llama3Preprocessor,
        "generation_config": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
           
        },
        "slurm_config": {
            "mem": "40G",
            "gpus": "a100:1",
            "cpus-per-task": "16",	
            "seconds-per-example": "20",
            "nodes": "1"
        }
    },
    "Llama3_8B": {
        "groups": ["small_models", "Llama3_8B-G"],
        "output_column": "LLama3_8BInstruct",
        "model_repo": "meta-llama/Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298",
        "model_kwargs": {},
        "preprocessor": Llama3Preprocessor,
        "generation_config": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
            "eos_token_id": lambda tokenizer: [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Assumes this token is an empty string
            ]
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7, 
            "eos_token_id": lambda tokenizer: [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Assumes this token is an empty string
            ]
        },
        "slurm_config": {
            "mem": "40G",
            "gpus": "a100:1",
            "cpus-per-task": "16",	
            "seconds-per-example": "20",
            "nodes": "1"
        }
    },
    "Phi3_3.8B": {
        "groups": ["small_models", "Phi3_3.8B-G"],
        "output_column": "Phi3_3.8B",
        "model_repo": "microsoft/Phi-3-mini-128k-instruct/models--microsoft--Phi-3-mini-128k-instruct/snapshots/a90b62ae09941edff87a90ced39ba5807e6b2ade",
        "model_kwargs": {
            "trust_remote_code": True,
        },
        "preprocessor": Llama3Preprocessor,
        "generation_config": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "_from_model_config": True,
        },
        "retry_generation_config": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "_from_model_config": True,
            "temperature": 0.2,
            "top_p": 0.7
        },
        "slurm_config": {
            "mem": "40G",
            "gpus": "a100:1",
            "cpus-per-task": "16",	
            "seconds-per-example": "20",
            "nodes": "1"
        }
    },

}

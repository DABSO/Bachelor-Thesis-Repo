from dotenv import load_dotenv
import os
from src.ArgParsers.InferenceArgParser import ArgParser
import subprocess
from src.ConfigLoader import ConfigLoader

load_dotenv()

models = ConfigLoader.load_available_models()
datasets = ConfigLoader.load_available_datasets()
prompts = ConfigLoader.load_available_prompts()

args = ArgParser(models, datasets, prompts).parse_args()

def submit_job(slurm_args, python_args):
        # Define the path to the template
        template_path = 'submit_slurm_job_template.sh'
        
        # Read the template file
        with open(template_path, 'r') as file:
            content = file.readlines()
        
        # Modify by adding sbatch directives after the first line
        new_content = [content[0]]  # Start with the first line

        # Add the slurm arguments as new lines
        for key, value in slurm_args.items():
            new_content.append(f'#SBATCH --{key}={value}\n')

        # Add the rest of the content after the first line
        new_content.extend(content[1:])
        # Add Python arguments to the python command line
        python_command_line = f"python3  -u -m run_inference {' '.join([f'--{k} {v}' for k, v in python_args.items() ])} \n"
        for i, line in enumerate(new_content):
            if line.strip().startswith('python3'):
                new_content[i] = python_command_line
        
        # Determine the new script name from job-name argument
        job_name = slurm_args.get('job-name', 'default_job')
        new_script_name = f"scheduler_scripts/{job_name}.sh"
        print("writing lines")
        for line in new_content:
            print(line)
        # Write the modified content to a new sbatch file
        with open(new_script_name, 'w') as file:
            file.writelines(new_content)
        
        # Submit the new sbatch file
        result = subprocess.run(['sbatch', new_script_name], stdout=subprocess.PIPE, text=True)
        print(result.stdout)

for model in models:
    if args.model != model:
        continue
    for dataset in datasets:
        if args.dataset != dataset:
            continue
        for prompt in prompts:
            if args.prompt != prompt:
                continue
            args.model = model
            args.dataset = dataset
            args.prompt = prompt

            local_files_only = not args.load_from_internet
            config_loader = ConfigLoader(args.model, args.prompt, args.dataset, local_files_only)
            inference_config = config_loader.load_inference_config()
            
            print("Dispatching Slurm Job")

            os.environ["DRY_RUN"] = "False"

            print("Dispatching Slurm Job")
            slurm_config = config_loader.load_slurm_config()
            # Dispatch slurm job 

            # Example usage
            script_path = 'submit_slurm_job.sh'
            # Example of overriding default parameters
            params = {
                "model": args.model ,
                "dataset": args.dataset,
                "prompt": args.prompt ,
                
                "batch_size" : args.batch_size
                }
            if args.failed_only:
                print("Failed only")
                params["failed_only"] = ""

            submit_job( slurm_config, params)
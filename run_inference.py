from dotenv import load_dotenv
import os
from src.ArgParsers.InferenceArgParser import ArgParser
from src.InferencePipeline import InferencePipeline
from src.ConfigLoader import ConfigLoader
load_dotenv()


models = ConfigLoader.load_available_models()
datasets = ConfigLoader.load_available_datasets()
prompts = ConfigLoader.load_available_prompts()

args = ArgParser(models, datasets, prompts).parse_args()
print("Running job: ", args.prompt + "-" +args.dataset )
local_files_only = not args.load_from_internet
dry_run = args.dry_run
if dry_run:
    os.environ["DRY_RUN"] = "True"

config_loader = ConfigLoader(args.model, args.prompt, args.dataset, local_files_only, args.failed_only)
inference_config = config_loader.load_inference_config()
inference_pipeline = InferencePipeline(**inference_config, local_files_only=local_files_only)
inference_pipeline.run(batch_size=args.batch_size or 1)


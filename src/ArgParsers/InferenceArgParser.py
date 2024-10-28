import argparse


class ArgParser(argparse.ArgumentParser):

    def __init__(self, available_models : list[str], available_datasets : list[str], available_prompts : list[str]):
        super().__init__()
        self.add_argument(
            "--dataset", 
            help="The dataset to use", 
            choices=[*available_datasets, "[all]"], 
            required=False,
            default="[all]"
        )
    
        self.add_argument(
            "--prompt", 
            help="The Prompt to use", 
            choices=[*available_prompts, "[all]"], 
            required=False,
            default="[all]"
        )

        self.add_argument(
            "--model", 
            help="The model name to use", 
            choices=available_models, 
            required=True
        )
        self.add_argument(
            "--batch_size", 
            help="The batch size to use", 
            type=int, 
            default=1, 
            required=False
        )

        self.add_argument(
            "--failed_only",
            help="If set, only failed rows will be reprocessed",
            type=self.str2bool,
            nargs='?',
            const=True,
            default=False,
            required=False
        )

    

        self.add_argument(
            "--load_from_internet",
            help="If set, models can be loaded from the internet",
            action='store_true',
            default=False
        )

        self.add_argument(
            "--dry_run", 
            help="If set, the Model is not actually loaded. For tests only.", 
            action='store_true'
        )

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
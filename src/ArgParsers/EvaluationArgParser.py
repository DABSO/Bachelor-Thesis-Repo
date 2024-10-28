import argparse


class EvaluationArgParser(argparse.ArgumentParser):

    def __init__(self, available_datasets : list[str], available_prompts : list[str] = [], feature_columns : list[str] = []):
        super().__init__()
        self.add_argument(
            "--dataset", 
            help="The dataset to use", 
            choices=[*available_datasets, "[all]"], 
            required=False,
            default="[all]"
        )

        if available_prompts:
            self.add_argument(
                "--prompt", 
                help="The Prompt to use", 
                choices=[*available_prompts, "[all]"], 
                required=False,
                default="[all]"
            )

        if feature_columns:
            self.add_argument(
                "--feature_columns",
                help="The columns to use as features, separated by commas",
                type=str,
                required=False,
                
            )
        
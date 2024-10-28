from pandas import DataFrame
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizerBase


class Preprocessor(ABC):
    df: DataFrame
    tokenizer: PreTrainedTokenizerBase
    prompt_templates: list[dict]
    

    def __init__(self, prompt_templates: list[dict], output_column: str, tokenizer: PreTrainedTokenizerBase, keep_columns: list[str] = [], preprocessing_steps: list[callable] = []):
        """
        Initializes the Preprocessor with the provided prompt, variables, output_column and tokenizer
        Parameters:
        prompt_templates : list[dict]
            A list of dictionaries containing the prompt templates for the preprocessing
            each dictionary should contain the following
            "content" : str 
            "role" : str
            "variables": dict or None, A dictionary mapping the variable names in the prompt to the column names in the DataFrame
        output_column : str
            The name of the column to store the results of the preprocessing
        tokenizer : transformers.Tokenizer
            The tokenizer to be used for tokenization
        """
        self.prompt_templates = prompt_templates
        self.output_column = output_column
        self.tokenizer = tokenizer
        self.keep_columns = keep_columns
        self.preprocessing_steps = preprocessing_steps

    def apply_preprocessing_step(self, df: DataFrame, preprocessing_function : callable):
        df = df.apply(preprocessing_function, axis=1)
        return df

    @abstractmethod
    def run(self, input_df: DataFrame, output_df : DataFrame):
        pass

    def apply_prompt(self, example: dict, prompt_template: dict):
        """
        Applies the prompt template based on the provided variables to a single example.
        Parameters:
        example : dict
            A single example from the DataFrame
        prompt_template : dict
            A single prompt template
        Returns:
        str
            The generated prompt
        """
        variables = prompt_template.get("variables", {})
        if variables is None:
                return prompt_template["content"]
        prompt_variables = {key: example[value] for key, value in variables.items()}
        return prompt_template["content"].format(**prompt_variables)

    def create_messages(self, example: dict, prompt_templates: list[dict]):
        """
        Applies multiple chat templates to a single example.
        Parameters:
        example : dict
            A single example from the DataFrame
        prompt_templates : list[dict]
            List of prompt templates
        Returns:
        list
            List of dictionaries with roles and contents
        """
        chat_messages = []
        for template in prompt_templates:
            content = self.apply_prompt(example, template)
            chat_messages.append({"role": template["role"], "content": content})
        return chat_messages
       


class ChatModelPreprocessor(Preprocessor):

    def __init__(self, prompt_templates: list[dict], output_column: str, tokenizer,keep_columns: list[str] = [], preprocessing_steps: list[callable] = []):
        super().__init__(prompt_templates, output_column, tokenizer, keep_columns=keep_columns, preprocessing_steps=preprocessing_steps)

    @abstractmethod
    def run(self, df: DataFrame):
        pass


class MistralPreprocessor(ChatModelPreprocessor):

    def __init__(self, prompt_templates: list[dict], output_column: str, tokenizer, keep_columns: list[str] = [], preprocessing_steps: list[callable] = []):
        super().__init__(prompt_templates, output_column, tokenizer, keep_columns=keep_columns, preprocessing_steps=preprocessing_steps)

    def run(self, input_df: DataFrame, output_df : DataFrame):
        for step in self.preprocessing_steps:
            input_df = self.apply_preprocessing_step(input_df, step)
        results = []
        for index, row in input_df.iterrows():
            chat_messages = self.create_messages(row.to_dict(), self.prompt_templates)
            results.append(chat_messages)
            
        input_df[self.output_column] = results
        for col in self.keep_columns:
            output_df[col] = input_df[col]
        return input_df, output_df


class Llama3Preprocessor(ChatModelPreprocessor):

    def __init__(self, prompt_templates: list[dict], output_column: str, tokenizer, keep_columns: list[str] = [], preprocessing_steps: list[callable] = []):
        super().__init__(prompt_templates, output_column, tokenizer, keep_columns=keep_columns, preprocessing_steps=preprocessing_steps)

    def run(self, input_df: DataFrame, output_df : DataFrame):
        for step in self.preprocessing_steps:
            input_df = self.apply_preprocessing_step(input_df, step)


        results = []
        for index, row in input_df.iterrows():
            chat_messages = self.create_messages(row.to_dict(), self.prompt_templates)
            results.append(chat_messages)
            
        input_df[self.output_column] = results
        for col in self.keep_columns:
            output_df[col] = input_df[col]
        return input_df, output_df
        




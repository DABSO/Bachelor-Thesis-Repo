from transformers.tokenization_utils import PreTrainedTokenizerBase


class InputFormatter:
    tokenizer = None

    def __init__(self, tokenizer : PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer


    def format(self, messages : list[dict]):
        """
        Formats the input messages into a chat sequence that can be fed into the model
        """
        
        return self.tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)

        
        
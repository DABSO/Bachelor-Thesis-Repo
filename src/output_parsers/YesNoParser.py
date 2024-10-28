import re
from .base import Parser, Validator, Evaluator
from collections import Counter


class YesNoParser(Parser, Validator, Evaluator):

    def parse(self, string: str) -> str:
        if not isinstance(string, str): return None
        # find the last match of either "###yes" or "###no"
        matches = re.findall(r'###\s*(yes|no)', string)
        return matches[-1] if matches else None
    
    def validate(self, string: str) -> bool:
        if not isinstance(string, str): return False
        return len(re.findall(r'###\s*(yes|no)', string)) > 0 
    

    def evaluate(self, prediction: str, label: str) -> bool:
        # check if the prediction is either "yes" or "no" else return None
        if prediction not in ["yes", "no"]:
            return None
        
        return prediction == label
        

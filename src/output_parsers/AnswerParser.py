import re
from .base import Parser, Validator, Evaluator
from collections import Counter

class AnswerParser(Parser, Validator, Evaluator):
    pattern = answer_pattern = r'<answer>(.*?)</answer>|\(answer>(.*)<\\/answer>|\(answer\)(.*)\(answer\)|\(answer\)(.*)<\|eot_id\|>|\(Answer\)(.*)<\|eot_id\|>|\(answer>(.*)<\|eot_id\|>|<Solution><Answer>(.*)</Solution>'

    def parse(self, string : str):
        if not isinstance(string, str): return None
        matches = re.finditer(self.pattern, string)

        
        matches = [next(group for group in match.groups() if group is not None) for match in matches]
        
        # Flatten the tuple into a list and filter out None values
        matches = [match for match in matches if match is not None]

        unanswerable_indicators = ["Unknown", "None", "No Information provided", "Not specified"]


        if len(matches) == 0:
            return None
        
        if any(indicator in matches[-1] for indicator in unanswerable_indicators):
            
            return None

        return matches[-1]

    
    def validate(self, string : str):
        print("validating" , string )
        if not isinstance(string, str): return False
        res = len(re.findall(self.pattern, string)) > 0 
        print("is valid?", res)
        return res

    def validate_generation(self, string : str):
        """
        Check if the string contains a valid answer and returns information what to do next
        """
        if self.validate(string):
            return True, "Answer is valid"
        
        return True

    def evaluate(self, string: str, label : str) -> bool:
        if string is None:
            return None

        # split the label and string into 2 lists of answers
        predictions = string.split(";")
        predictions = [prediction.strip() for prediction in predictions]

        labels = label.split(";")
        labels = [l.strip() for l in labels]

        # check if the predictions match the labels exactly
        is_match = self.check_exact_match(predictions, labels)
        return is_match

    @staticmethod
    def check_exact_match(arr1 , arr2):
        if arr1 == arr2:
            return True
        elif arr1 is None or arr2 is None: 
            return False
        elif Counter(arr1) == Counter(arr2):
            return True
        else:
            return False


        

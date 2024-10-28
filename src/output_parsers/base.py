from abc import ABC, abstractmethod


# Define the Validator interface
class Validator(ABC):
    REGENERATE = "REGENERATE"
    REPLY = "REPLY"

    @abstractmethod
    def validate(self, string: str) -> bool:
        pass
    

# Define the Parser interface
class Parser(ABC):
    @abstractmethod
    def parse(self, string: str) -> str:
        pass


class Evaluator(ABC):
    
    @abstractmethod
    def evaluate(self, string: str):
        pass


class BooleanEvaluator(Evaluator):
    @abstractmethod
    def evaluate(self, string: str, label : str ) -> bool:
        pass
        

    

import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Any, List
import json

import json
from typing import Dict, List, Any, Union, Tuple
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Any, List
import json
from transformers import LogitsProcessor



class JSONSchemaEnforcer(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer, schema: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.schema = schema
        self.prompt_text = None
        self.generated_text = ""
        self.queue = []
        self.first_call = True
        self.current_level = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        print("current level", self.current_level)
        
        current_full_text = self.tokenizer.decode(input_ids[0])
        
        
        if self.prompt_text is None or not current_full_text.startswith(self.prompt_text):
            self.prompt_text = current_full_text
            self.generated_text = ""
            self.first_call = True
        else:
            self.generated_text = current_full_text[len(self.prompt_text):].lstrip()


        if self.first_call:
            self.first_call = False
            if not self.generated_text:
                if self.schema["type"] == "object":
                    forced_char = "{"
                    self.current_level = 1
                elif self.schema["type"] == "array":
                    forced_char = "["
                    self.current_level = 1
                else:
                    raise ValueError("Schema type must be either 'object' or 'array'")
                
                forced_token = self.tokenizer.encode(forced_char, add_special_tokens=False)[0]
                scores_processed = torch.full_like(scores, -float("inf"))
                scores_processed[:, forced_token] = 0
                print("inserting", forced_char)
                return scores_processed

        if not self.queue:
            result =generate_json_suffix(self.schema, self.generated_text, self.current_level)
            self.current_level = result["current_level"]
            completed_text = result["completed_text"]
            print("types", type(completed_text), type(self.current_level))
            if completed_text != self.generated_text:
                new_suffix = completed_text[len(self.generated_text):]
                print("inserting:", new_suffix)
                print("completed text",completed_text)
                self.queue = self.tokenizer.encode(new_suffix, add_special_tokens=False)

        if self.queue:
            next_token = self.queue.pop(0)
            scores_processed = torch.full_like(scores, -float("inf"))
            scores_processed[:, next_token] = 0
        else:
            # If no forced tokens, allow normal generation
            scores_processed = scores

        return scores_processed
        
# Assuming these functions are defined elsewhere

def is_in_array(json_string: str, position: int) -> bool:
    """
    Check if the given position in the JSON string is inside an array.
    
    Args:
    json_string (str): The JSON string to check.
    position (int): The position in the string to check.
    
    Returns:
    bool: True if the position is inside an array, False otherwise.
    """
    stack = []
    in_string = False
    escape = False
    
    for i, char in enumerate(json_string[:position]):
        if char == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if char == '[':
                stack.append('[')
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                else:
                    return False  # Mismatched brackets
        
        escape = char == '\\' and not escape
    
    return bool(stack and stack[-1] == '[')

def is_in_incomplete_string(json_string: str, position: int) -> bool:
    """
    Check if the given position in the JSON string is inside an incomplete string.
    
    Args:
    json_string (str): The JSON string to check.
    position (int): The position in the string to check.
    
    Returns:
    bool: True if the position is inside an incomplete string, False otherwise.
    """
    in_string = False
    escape = False
    
    for i, char in enumerate(json_string[:position]):
        if char == '"' and not escape:
            in_string = not in_string
        escape = char == '\\' and not escape
    
    return in_string

_type_map = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
    "array": list,
    "object": dict
}


def get_missing_keys(schema: Dict[str, Any], data: Dict[str, Any], path: str = "", type_path: str = "", level: int = 0) -> List[Tuple[str, str, str, str, int]]:
    missing_keys = []
    
    if "properties" in schema:
        properties = schema["properties"]
    else:
        properties = schema  # For nested objects that don't have a "properties" key
    
    for key, value in properties.items():
        new_path = f"{path}.{key}" if path else key
        new_type_path = f"{type_path}.{value.get('type', 'unknown')}" if type_path else value.get('type', 'unknown')
        
        if key not in data:
            
            missing_keys.append((new_path, value.get("type", "unknown"), "missing", new_type_path, level))
        elif "type" in value:
            if value["type"] == "object" and "properties" in value:
                if isinstance(data[key], dict):
                    missing_keys.extend(get_missing_keys(value, data[key], new_path, new_type_path, level + 1))
                else:
                    missing_keys.append((new_path, "object", "type mismatch", new_type_path, level))
            elif value["type"] == "array" and "items" in value:
                if isinstance(data[key], list):
                    for i, item in enumerate(data[key]):
                        if isinstance(value["items"], dict) and "properties" in value["items"]:
                            array_type_path = f"{new_type_path}.{value['items'].get('type', 'unknown')}"
                            missing_keys.extend(get_missing_keys(value["items"], item, f"{new_path}[{i}]", array_type_path, level + 1))
                else:
                    missing_keys.append((new_path, "array", "type mismatch", new_type_path, level))
            elif not isinstance(data[key], _type_map.get(value["type"], object)):
                
                missing_keys.append((new_path, value["type"], "type mismatch", new_type_path, level))
        
    print("Level: ", level, "keys", missing_keys)
    return missing_keys
def get_existing_keys(schema: Dict[str, Any], data: Dict[str, Any], path: str = "", type_path: str = "", level: int = 0) -> List[Tuple[str, str, str, str, int]]:
    existing_keys = []
    
    if "properties" in schema:
        properties = schema["properties"]
    else:
        properties = schema  # For nested objects that don't have a "properties" key
    
    for key, value in properties.items():
        new_path = f"{path}.{key}" if path else key
        new_type_path = f"{type_path}.{value.get('type', 'unknown')}" if type_path else value.get('type', 'unknown')
        
        if key in data:
            if "type" in value:
                if value["type"] == "object" and "properties" in value:
                    if isinstance(data[key], dict):
                        existing_keys.append((new_path, value["type"], "existing", new_type_path, level))
                        existing_keys.extend(get_existing_keys(value, data[key], new_path, new_type_path, level + 1))
                    else:
                        existing_keys.append((new_path, "object", "type mismatch", new_type_path, level))
                elif value["type"] == "array" and "items" in value:
                    if isinstance(data[key], list):
                        existing_keys.append((new_path, value["type"], "existing", new_type_path, level))
                        for i, item in enumerate(data[key]):
                            if isinstance(value["items"], dict) and "properties" in value["items"]:
                                array_type_path = f"{new_type_path}.{value['items'].get('type', 'unknown')}"
                                existing_keys.extend(get_existing_keys(value["items"], item, f"{new_path}[{i}]", array_type_path, level + 1))
                    else:
                        existing_keys.append((new_path, "array", "type mismatch", new_type_path, level))
                elif isinstance(data[key], _type_map.get(value["type"], object)):
                    existing_keys.append((new_path, value["type"], "existing", new_type_path, level))
                else:
                    existing_keys.append((new_path, value["type"], "type mismatch", new_type_path, level))
            else:
                existing_keys.append((new_path, "unknown", "existing", new_type_path, level))
    
    return existing_keys

def complete_json_schema(incomplete_schema: str) -> str:
    stack = []
    opening_chars = {'{': '}', '[': ']'}
    closing_chars = {'}', ']'}
    in_string = False
    if incomplete_schema[-1] == ",":
        incomplete_schema = incomplete_schema[:-1]
    
    
    for char in incomplete_schema:
        if char == '"' and (not stack or stack[-1] != '\\'):
            if in_string:
                stack.pop()  # Remove the opening quote
            else:
                stack.append('"')
            in_string = not in_string
        elif not in_string:
            if char in opening_chars:
                stack.append(char)
            elif char in closing_chars:
                if stack and opening_chars.get(stack[-1]) == char:
                    stack.pop()
                # Ignore mismatched closing characters
        elif char == '\\':
            stack.append('\\')
        elif stack and stack[-1] == '\\':
            stack.pop()  # Remove the backslash after processing the escaped character
    
    completion = ''
    while stack:
        last_char = stack.pop()
        if last_char == '"':
            completion += '"'
        elif last_char in opening_chars:
            completion += opening_chars[last_char]
        # Ignore any remaining backslashes
    
    return incomplete_schema + completion

def validate_and_complete_json(schema: Dict[str, Any], incomplete_json: str) -> Dict[str, Union[Dict[str, Any], List[str]]]:
    # Complete the JSON if it's incomplete
    completed_json = complete_json_schema(incomplete_json)
    
    try:
        
        # Parse the completed JSON
        data = json.loads(completed_json)
        
        # Get missing keys
        missing_keys = get_missing_keys(schema, data, level=1)
        print(missing_keys)
        existing_keys = get_existing_keys(schema, data, level=1)
        
        return {
            "completed_json": data,
            "missing_keys": missing_keys,
            "existing_keys": existing_keys
        }
    except json.JSONDecodeError as e:
        print("Error during validate and complete json", str(e))
        return {
            "error": f"Invalid JSON: {str(e)}",
            "completed_json": completed_json
        }

import re
from typing import Dict, Any, Tuple, List
def get_current_path_and_level(json_string: str) -> Tuple[List[str], int]:
    stack = []
    current_path = []
    current_key = ""
    in_string = False
    escape_next = False

    for char in json_string:
        if escape_next:
            escape_next = False
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            if not in_string and current_key:
                current_path.append(current_key)
                current_key = ""
        elif in_string:
            if char == '\\':
                escape_next = True
            else:
                current_key += char
        elif char in '{[':
            stack.append(char)
        elif char in '}]':
            if stack:
                stack.pop()
            if current_path:
                current_path.pop()

    return current_path, len(stack)


def generate_json_suffix(schema: Dict[str, Any], json_string: str, level : int) -> Tuple[str, int]:
    """
    Complete the JSON string with the next key based on the schema.

    Args:
    schema (Dict[str, Any]): The JSON schema to validate against.
    json_string (str): The incomplete JSON string.

    Returns:
    str: The JSON string with the next key added, if applicable.
    """
    # Check if the last character is a separator
    curr_path , level = get_current_path_and_level(json_string)
    if json_string[-1] not in '{",]}':
        return {
                "completed_text": json_string,
                "current_level": level
            }

    # Check if we're in an array or incomplete string
    if is_in_array(json_string, len(json_string) ) or is_in_incomplete_string(json_string, len(json_string)):
        return {
                "completed_text": json_string,
                "current_level": level
            }
    # 

    # Validate and get missing keys
    result = validate_and_complete_json(schema, json_string)
    if "error" in result:
        return {
                "completed_text": json_string,
                "current_level": level
            }  # If there's an error, return the original string

    missing_keys = result.get("missing_keys", [])
    if not missing_keys and json_string[-1] not in '{,"':
        completed_json_string = complete_json_schema(json_string)
        if completed_json_string != json_string:
            return {
                "completed_text": completed_json_string,
                "current_level": level
            }
        return {
                "completed_text": json_string,
                "current_level": level
            }  # No missing keys, return the original string
    elif not missing_keys and json_string[-1] in '{,"':
        print("preventing premature closing")
        return {
            "completed_text": json_string,
            "current_level": level
        }

    # determine if the key needs a prefix brace closure 
    existing_keys =  result.get("existing_keys", [])
    
    key_prefix = ""
    current_level = missing_keys[0][4]
    
    
    if len(existing_keys) > 0:
        
        
        print("regular out",current_level, level)
        print("Existing keys", existing_keys)
        type_path =  existing_keys[0][3].split(".")
        while current_level < level:
            level -= 1
            key_prefix += "}" if type_path[level - 1] == "object" else "]" if type_path[level - 1] == "array" else ""
            print("prefixing with:", key_prefix )
        

    # Determine the next key
    if "." in missing_keys[0][0]:
        next_key = missing_keys[0][0].split('.')[-1]  # Get the first part of the first missing key
    else:
        next_key = missing_keys[0][0]
    key_type = missing_keys [0][1]

    type_suffix = ""
    if key_type.lower() == "string":
        type_suffix = '"'
    elif key_type.lower() == "array":
        type_suffix = '['
    elif key_type.lower() == "object":
        type_suffix = "{"
   


    # Prepare the string for the new key
    

    # Add the new key
    json_string += key_prefix 
    json_string += ',' if json_string[-1] in '"]}' else ""
    json_string += f'"{next_key}":'
    json_string += type_suffix
    print("Returning:", json_string, current_level, len(json_string))
    return{
        "completed_text": json_string,
        "current_level": current_level
    }
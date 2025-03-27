import os
import sys
import json
import time
import uuid
import signal
import shutil
import optional
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, request, jsonify, render_template, session
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import random
import openai
from flask import Flask, request, jsonify, render_template, session
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI client for generating questions
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# This is a conceptual implementation of Pynguine
class Pynguine:
    @staticmethod
    def generate_tests(function_signature: str, description: str, examples: List[Dict] = None, num_tests: int = 5) -> List[Dict[str, Any]]:
        """
        Generate test cases based on function signature, description, and examples.
        
        Args:
            function_signature: The Python function signature
            description: Description of what the function should do
            examples: List of example input/output pairs
            num_tests: Number of test cases to generate
            
        Returns:
            List of test cases with inputs and expected outputs
        """
        # Extract function name and parameters
        match = re.match(r'def\s+(\w+)\s*\((.*?)\)', function_signature)
        if not match:
            raise ValueError("Invalid function signature")
            
        function_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if ':' in param:
                    name, type_hint = param.split(':', 1)
                    params.append((name.strip(), type_hint.strip()))
                else:
                    params.append((param, None))
        
        # Start with example test cases if provided
        test_cases = []
        if examples:
            for i, example in enumerate(examples):
                test_cases.append({
                    "test_id": i + 1,
                    "is_example": True,
                    "inputs": example["inputs"],
                    "expected_output": example["output"]
                })
        
        # Generate additional test cases
        for i in range(len(test_cases), num_tests):
            test_case = {
                "test_id": i + 1,
                "is_example": False,
                "inputs": {},
                "expected_output": None
            }
            
            # Generate inputs based on parameter names and types
            for param_name, param_type in params:
                test_case["inputs"][param_name] = Pynguine._generate_sample_input(param_name, param_type, description)
            
            # Generate expected output based on function name and inputs
            test_case["expected_output"] = Pynguine._generate_expected_output(
                function_name, test_case["inputs"], description
            )
            
            test_cases.append(test_case)
            
        return test_cases
    
    @staticmethod
    def _generate_sample_input(param_name: str, param_type: Optional[str], description: str) -> Any:
        """Generate appropriate sample input based on parameter name and type hint"""
        # This is a simplified version - a real implementation would be more sophisticated
        
        # Check type hint first if available
        if param_type:
            if 'int' in param_type:
                return random.randint(-100, 100)
            elif 'float' in param_type:
                return round(random.uniform(-100, 100), 2)
            elif 'str' in param_type:
                return f"test_string_{random.randint(1, 100)}"
            elif 'list' in param_type or 'List' in param_type:
                # Try to determine list type
                if 'int' in param_type:
                    return [random.randint(-10, 10) for _ in range(random.randint(3, 7))]
                elif 'str' in param_type:
                    return [f"item_{i}" for i in range(random.randint(3, 7))]
                else:
                    return [random.randint(-10, 10) for _ in range(random.randint(3, 7))]
            elif 'dict' in param_type or 'Dict' in param_type:
                return {f"key_{i}": random.randint(1, 100) for i in range(random.randint(2, 5))}
            elif 'bool' in param_type:
                return random.choice([True, False])
        
        # If no type hint or unsupported type, infer from parameter name
        if 'num' in param_name or 'count' in param_name or 'index' in param_name:
            return random.randint(1, 100)
        elif 'name' in param_name or 'text' in param_name or 'str' in param_name:
            return f"sample_{param_name}_{random.randint(1, 100)}"
        elif 'list' in param_name or 'array' in param_name:
            return [random.randint(1, 100) for _ in range(random.randint(3, 7))]
        elif 'dict' in param_name or 'map' in param_name:
            return {f"key_{i}": random.randint(1, 100) for i in range(random.randint(2, 5))}
        elif 'flag' in param_name or 'enable' in param_name:
            return random.choice([True, False])
        else:
            # Default to string if we can't determine type
            return f"default_value_for_{param_name}"
    
    @staticmethod
    def _generate_expected_output(function_name: str, inputs: Dict[str, Any], description: str) -> Any:
        """
        Generate expected output based on function name, inputs, and description.
        This implementation includes specific handling for common algorithm types.
        """
        function_name_lower = function_name.lower()
        description_lower = description.lower()
        
        # Handle palindrome functions
        if 'palindrome' in function_name_lower or ('palindrome' in description_lower and 'is_' in function_name_lower):
            for input_val in inputs.values():
                if isinstance(input_val, str):
                    # Clean the string and check if it's a palindrome
                    cleaned = ''.join(char.lower() for char in input_val if char.isalnum())
                    return cleaned == cleaned[::-1]
        
        # Handle anagram functions
        elif 'anagram' in function_name_lower or 'anagram' in description_lower:
            input_values = list(inputs.values())
            if len(input_values) >= 2 and all(isinstance(x, str) for x in input_values[:2]):
                s1, s2 = input_values[0], input_values[1]
                # Check if strings are anagrams (same characters, different order)
                return sorted(s1.lower()) == sorted(s2.lower())
        
        # Sum functions
        elif 'sum' in function_name_lower or 'add' in function_name_lower:
            # If there are numeric inputs, return their sum
            numeric_inputs = [v for v in inputs.values() if isinstance(v, (int, float))]
            if numeric_inputs:
                return sum(numeric_inputs)
            
            # If there's a list of numbers, sum it
            for input_val in inputs.values():
                if isinstance(input_val, list) and all(isinstance(x, (int, float)) for x in input_val):
                    return sum(input_val)
        
        # Average/mean functions
        elif 'average' in function_name_lower or 'mean' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, list) and all(isinstance(x, (int, float)) for x in input_val):
                    return sum(input_val) / len(input_val) if input_val else 0
        
        # Maximum value functions
        elif 'max' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, list) and input_val:
                    try:
                        return max(input_val)
                    except TypeError:
                        pass
        
        # Minimum value functions
        elif 'min' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, list) and input_val:
                    try:
                        return min(input_val)
                    except TypeError:
                        pass
        
        # Count/length functions
        elif any(x in function_name_lower for x in ['count', 'length', 'len']):
            for input_val in inputs.values():
                if isinstance(input_val, (list, str, dict)):
                    return len(input_val)
        
        # Reverse functions
        elif 'reverse' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, list):
                    return input_val[::-1]
                elif isinstance(input_val, str):
                    return input_val[::-1]
        
        # Sort functions
        elif 'sort' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, list):
                    try:
                        return sorted(input_val)
                    except TypeError:
                        pass
        
        # Find missing number
        elif 'find' in function_name_lower and 'missing' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, list) and all(isinstance(x, int) for x in input_val):
                    all_nums = set(range(1, max(input_val) + 2))
                    return min(all_nums - set(input_val))
        
        # Prime number check
        elif 'prime' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, int) and input_val > 1:
                    for i in range(2, int(input_val ** 0.5) + 1):
                        if input_val % i == 0:
                            return False
                    return True
        
        # Fibonacci function
        elif 'fibonacci' in function_name_lower or 'fib' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, int) and input_val >= 0:
                    if input_val <= 1:
                        return input_val
                    a, b = 0, 1
                    for _ in range(2, input_val + 1):
                        a, b = b, a + b
                    return b
        
        # Factorial function
        elif 'factorial' in function_name_lower:
            for input_val in inputs.values():
                if isinstance(input_val, int) and input_val >= 0:
                    result = 1
                    for i in range(2, input_val + 1):
                        result *= i
                    return result
        
        # Common types based on description and return type
        if 'boolean' in description_lower or 'true/false' in description_lower:
            return False  # Default to False for boolean functions
        elif 'list' in description_lower:
            return []  # Empty list as default
        elif 'string' in description_lower:
            return ""  # Empty string as default
        elif 'number' in description_lower or 'integer' in description_lower:
            return 0  # Zero as default
        else:
            # Default fallback - use None instead of a misleading string
            return None


class LLMQuestionGenerator:
    """Class to generate coding questions using OpenAI's API"""
    
    def __init__(self):
        self.client = openai_client
        self.previous_questions = set()
        self.use_local_questions = not bool(OPENAI_API_KEY.strip())  # If no API key, use local questions

    def _get_local_questions(self) -> Dict[str, List[Dict[str, str]]]:
        """Return a dictionary of pre-defined local questions"""
        return {
            "easy": [
                {
                    "problem": """Write a function that finds the second largest element in a list of integers. If the list has fewer than 2 elements, return -1.""",
                    "signature": "def find_second_largest(arr: List[int]) -> int:",
                    "examples": [
                        ("Input: [1, 3, 2, 5, 4]", "Output: 4"),
                        ("Input: [1, 1, 1]", "Output: -1"),
                        ("Input: [7]", "Output: -1")
                    ]
                },
                {
                    "problem": """Write a function that counts the number of vowels in a given string.""",
                    "signature": "def count_vowels(text: str) -> int:",
                    "examples": [
                        ("Input: 'hello'", "Output: 2"),
                        ("Input: 'PYTHON'", "Output: 1"),
                        ("Input: ''", "Output: 0")
                    ]
                }
            ],
            "medium": [
                {
                    "problem": """Write a function that checks if two strings are anagrams of each other, ignoring spaces and case.""",
                    "signature": "def are_anagrams(str1: str, str2: str) -> bool:",
                    "examples": [
                        ("Input: 'listen', 'silent'", "Output: True"),
                        ("Input: 'Hello World', 'World Hello'", "Output: True"),
                        ("Input: 'Python', 'Java'", "Output: False")
                    ]
                },
                {
                    "problem": """Write a function that finds the first non-repeating character in a string.""",
                    "signature": "def first_unique_char(s: str) -> str:",
                    "examples": [
                        ("Input: 'leetcode'", "Output: 'l'"),
                        ("Input: 'hello'", "Output: 'h'"),
                        ("Input: 'aabb'", "Output: ''")
                    ]
                }
            ],
            "hard": [
                {
                    "problem": """Write a function that finds the longest palindromic substring in a given string.""",
                    "signature": "def longest_palindrome(s: str) -> str:",
                    "examples": [
                        ("Input: 'babad'", "Output: 'bab'"),
                        ("Input: 'cbbd'", "Output: 'bb'"),
                        ("Input: 'a'", "Output: 'a'")
                    ]
                },
                {
                    "problem": """Write a function that implements a LRU (Least Recently Used) cache with a given capacity.""",
                    "signature": "class LRUCache:",
                    "examples": [
                        ("cache = LRUCache(2)", "None"),
                        ("cache.put(1, 1)", "None"),
                        ("cache.get(1)", "Returns: 1")
                    ]
                }
            ]
        }

    def generate_question(self, difficulty: str = "medium", topic: str = None) -> Dict[str, Any]:
        """Generate a Python coding question"""
        try:
            if self.use_local_questions:
                return self._get_local_question(difficulty, topic)
            
            try:
                question_text = self._generate_raw_question(difficulty, topic)
                question_info = self._parse_question(question_text)
                return {"success": True, "question": question_info}
            except (openai.RateLimitError, openai.APIError) as e:
                logger.warning(f"OpenAI API error, falling back to local questions: {e}")
                return self._get_local_question(difficulty, topic)
                
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return self._get_local_question("medium", None)  # Default fallback

    def _get_local_question(self, difficulty: str, topic: str = None) -> Dict[str, Any]:
        """Get a random local question based on difficulty"""
        questions = self._get_local_questions()
        difficulty = difficulty.lower() if difficulty else "medium"
        if difficulty not in questions:
            difficulty = "medium"
        
        question = random.choice(questions[difficulty])
        return {
            "success": True,
            "question": {
                "problem_statement": question["problem"],
                "function_signature": question["signature"],
                "examples": question["examples"]
            }
        }

    def _generate_raw_question(self, difficulty: str, topic: str = None) -> str:
        """Generate raw question text using OpenAI's API"""
        try:
            # Construct the prompt
            prompt = self._construct_prompt(difficulty, topic)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are a Python programming teacher creating coding questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract and return the question text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_default_raw_question(difficulty, topic)

    def _construct_prompt(self, difficulty: str, topic: str = None) -> str:
        """Construct the prompt for OpenAI API"""
        topic_str = f" about {topic}" if topic else ""
        
        return f"""Generate a Python coding question{topic_str} of {difficulty} difficulty.
        The response must follow this exact format:
        
        Problem Statement:
        [Write a clear description of what the function should do]

        Function Signature:
        ```python
        from typing import List, Dict  # Include if needed
        def function_name(params) -> return_type:
        ```

        Examples:
        Input: [exact input value(s)]
        Output: [exact expected output]

        [Provide at least 3 test cases with varied inputs]
        """

    def _get_default_raw_question(self, difficulty: str, topic: str = None) -> str:
        """Return a default question text when API fails"""
        default_questions = {
            "easy": """Problem Statement:
Write a function that finds the second largest element in a list of integers. If the list has fewer than 2 elements, return -1.

Function Signature:
def find_second_largest(arr: List[int]) -> int:

Examples:
Input: [1, 3, 2, 5, 4]
Output: 4

Input: [1, 1, 1]
Output: -1

Input: [7]
Output: -1""",
            "medium": """Problem Statement:
Write a function that checks if two strings are anagrams of each other, ignoring spaces and case.

Function Signature:
def are_anagrams(str1: str, str2: str) -> bool:

Examples:
Input: "listen", "silent"
Output: True

Input: "Hello World", "World Hello"
Output: True

Input: "Python", "Java"
Output: False""",
            "hard": """Problem Statement:
Write a function that finds the longest palindromic substring in a given string.

Function Signature:
def longest_palindrome(s: str) -> str:

Examples:
Input: "babad"
Output: "bab"

Input: "cbbd"
Output: "bb"

Input: "a"
Output: "a" """
        }
        
        return default_questions.get(difficulty, default_questions["medium"])
    
    def _parse_question(self, question_text: str) -> Dict[str, Any]:
        """Parse the raw question text into structured format"""
        try:
            # Basic validation of question text
            if not question_text or not isinstance(question_text, str):
                logger.error(f"Invalid question text received: {question_text}")
                return self._get_local_question("medium", None)['question']

            # Extract components
            lines = question_text.strip().split('\n')
            
            problem_statement = ""
            function_signature = ""
            imports = []
            examples = []
            function_name = None
            
            current_section = None
            current_example = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('Problem Statement:'):
                    current_section = "problem"
                    continue
                elif line.startswith('Function Signature:'):
                    current_section = "signature"
                    continue
                elif line.startswith('Examples:'):
                    current_section = "examples"
                    continue
                elif line.startswith('from typing import'):
                    imports.append(line)
                    continue
                
                if current_section == "problem":
                    if line != "Function Signature:":  # Avoid capturing next section header
                        problem_statement += line + " "
                elif current_section == "signature":
                    if line.startswith('def '):
                        function_signature = line
                        # Extract function name from signature
                        match = re.match(r'def\s+(\w+)\s*\(', line)
                        if match:
                            function_name = match.group(1)
                elif current_section == "examples":
                    if line.lower().startswith('input:'):
                        if current_example.get('input'):  # Save previous example if exists
                            examples.append(current_example.copy())
                        current_example = {'input': line[6:].strip()}
                    elif line.lower().startswith('output:') and current_example.get('input'):
                        current_example['output'] = line[7:].strip()
                        examples.append(current_example.copy())
                        current_example = {}

            # Add last example if pending
            if current_example.get('input') and current_example.get('output'):
                examples.append(current_example)

            # Format examples for return
            formatted_examples = []
            for example in examples:
                formatted_examples.extend([
                    f"Input: {example['input']}",
                    f"Output: {example['output']}"
                ])

            # If List is used in signature but import is missing, add it
            if 'List[' in function_signature and not any('typing import List' in imp for imp in imports):
                imports.append('from typing import List')

            # Combine imports and function signature
            full_signature = '\n'.join(imports + [function_signature]) if imports else function_signature
                
            # Ensure we have all required components
            if not problem_statement or not function_signature or not function_name:
                logger.error("Failed to parse question components")
                return self._get_local_question("medium", None)['question']
            
            return {
                "problem_statement": problem_statement.strip(),
                "function_signature": full_signature,
                "function_name": function_name,
                "examples": formatted_examples
            }
            
        except Exception as e:
            logger.exception("Error parsing question")
            return self._get_local_question("medium", None)['question']
    
    def _get_default_question(self, difficulty: str, topic: str) -> Dict[str, Any]:
        """Return a default question in case the API fails"""
        default_questions = {
            "arrays": {
                "easy": {
                    "problem_statement": "Write a function that finds the second largest element in an array. Return -1 if the array has fewer than 2 elements.",
                    "function_signature": "def find_second_largest(arr: List[int]) -> int:\n    pass",
                    "function_name": "find_second_largest",
                    "parameters": [{"name": "arr", "type": "List[int]"}],
                    "return_type": "int",
                    "examples": [
                        {"input": {"arr": [1, 3, 2, 5, 4]}, "output": 4},
                        {"input": {"arr": [1, 1, 1]}, "output": -1},
                        {"input": {"arr": [7]}, "output": -1}
                    ]
                },
                "medium": {
                    "problem_statement": "Write a function that finds the length of the longest subarray where the difference between any two elements is at most k.",
                    "function_signature": "def longest_subarray_with_diff_k(arr: List[int], k: int) -> int:\n    pass",
                    "function_name": "longest_subarray_with_diff_k",
                    "parameters": [{"name": "arr", "type": "List[int]"}, {"name": "k", "type": "int"}],
                    "return_type": "int",
                    "examples": [
                        {"input": {"arr": [1, 5, 3, 2, 6], "k": 2}, "output": 3},
                        {"input": {"arr": [4, 4, 4, 4], "k": 0}, "output": 4},
                        {"input": {"arr": [10, 1, 2, 3], "k": 1}, "output": 3}
                    ]
                },
                "hard": {
                    "problem_statement": "Write a function that returns the minimum number of intervals to remove to make the remaining intervals non-overlapping.",
                    "function_signature": "def min_intervals_to_remove(intervals: List[List[int]]) -> int:\n    pass",
                    "function_name": "min_intervals_to_remove",
                    "parameters": [{"name": "intervals", "type": "List[List[int]]"}],
                    "return_type": "int",
                    "examples": [
                        {"input": {"intervals": [[1,4], [2,3], [3,6]]}, "output": 1},
                        {"input": {"intervals": [[1,2], [2,3]]}, "output": 0},
                        {"input": {"intervals": [[1,2], [1,2], [1,2]]}, "output": 2}
                    ]
                }
            },
            "strings": {
                "easy": {
                    "problem_statement": "Write a function that counts the frequency of each character in a string and returns the first character that appears exactly once. Return None if no such character exists.",
                    "function_signature": "def first_unique_char(s: str) -> Optional[str]:\n    pass",
                    "function_name": "first_unique_char",
                    "parameters": [{"name": "s", "type": "str"}],
                    "return_type": "Optional[str]",
                    "examples": [
                        {"input": {"s": "statistics"}, "output": "a"},
                        {"input": {"s": "aabb"}, "output": None},
                        {"input": {"s": "x"}, "output": "x"}
                    ]
                }
            },
            "trees": {
                "medium": {
                    "problem_statement": "Write a function that finds the sum of all values in a binary tree at a given depth level (root is at level 0).",
                    "function_signature": "def sum_at_level(root: Optional[TreeNode], level: int) -> int:\n    pass",
                    "function_name": "sum_at_level",
                    "parameters": [{"name": "root", "type": "Optional[TreeNode]"}, {"name": "level", "type": "int"}],
                    "return_type": "int",
                    "examples": [
                        {"input": {"root": [1,2,3,4,5,6,7], "level": 2}, "output": 22},
                        {"input": {"root": [1], "level": 0}, "output": 1},
                        {"input": {"root": [], "level": 1}, "output": 0}
                    ]
                }
            }
        }

        if topic in default_questions and difficulty in default_questions[topic]:
            question = default_questions[topic][difficulty].copy()
        else:
            question = default_questions["arrays"]["medium"].copy()

        question["difficulty"] = difficulty
        question["topic"] = topic
        return question


class QuestionInfo:
    """Class to handle question information and validation"""
    
    def __init__(self, raw_info: Dict[str, Any]):
        self.raw_info = raw_info
        self.validate_and_normalize()
    
    def validate_and_normalize(self):
        """Validate and normalize question information"""
        required_fields = ['problem_statement', 'function_signature', 'function_name']
        for field in required_fields:
            if not self.raw_info.get(field):
                raise ValueError(f"Missing required field: {field}")
        
        # Extract parameters from function signature if not already present
        if 'parameters' not in self.raw_info:
            self.raw_info['parameters'] = self._extract_parameters(self.raw_info['function_signature'])
        
        # Normalize parameters
        self.raw_info['parameters'] = self._normalize_parameters(self.raw_info['parameters'])
        
        # Normalize examples
        self.raw_info['examples'] = self._normalize_examples(
            self.raw_info.get('examples', [])
        )

    def _extract_parameters(self, function_signature: str) -> List[Dict[str, str]]:
        """Extract parameters from function signature"""
        # Match everything between parentheses
        match = re.search(r'\((.*?)\)', function_signature)
        if not match:
            return []
            
        params_str = match.group(1)
        if not params_str.strip():
            return []
            
        parameters = []
        for param in params_str.split(','):
            param = param.strip()
            if ':' in param:
                name, type_hint = param.split(':', 1)
                parameters.append({
                    'name': name.strip(),
                    'type': type_hint.strip().rstrip(')')  # Remove trailing ) if present
                })
            else:
                parameters.append({
                    'name': param,
                    'type': 'Any'
                })
                
        return parameters

    def _normalize_parameters(self, parameters: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Normalize parameter information"""
        normalized = []
        for param in parameters:
            if isinstance(param, dict) and 'name' in param:
                normalized.append({
                    'name': param['name'],
                    'type': param.get('type', 'Any')
                })
        return normalized

    def _normalize_examples(self, examples: List[str]) -> List[Dict[str, Any]]:
        """Normalize example information"""
        normalized = []
        current_input = None
        
        for example in examples:
            if example.lower().startswith('input:'):
                current_input = example[6:].strip()
            elif example.lower().startswith('output:') and current_input is not None:
                normalized.append({
                    'inputs': {'s': current_input},  # Assuming single parameter for now
                    'output': example[7:].strip()
                })
                current_input = None
                
        return normalized or [{'inputs': {'s': 'example'}, 'output': False}]

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get information about a specific parameter"""
        for param in self.raw_info['parameters']:
            if param['name'] == param_name:
                return param
        return {'name': param_name, 'type': 'Any', 'optional': False}
    
    def is_optional_parameter(self, param_name: str) -> bool:
        """Check if a parameter is optional"""
        param_info = self.get_parameter_info(param_name)
        return param_info.get('optional', False)


class TestCaseGenerator:
    """Main class for generating test cases from LLM Python questions"""
    
    def __init__(self, pynguine=None):
        self.pynguine = pynguine or Pynguine()
        self.skip_optional_probability = 0.7  # Probability to skip optional parameters
    
    def generate_test_cases(self, question_info: Dict[str, Any], num_tests: int = 8) -> List[Dict[str, Any]]:
        """Generate test cases for a question"""
        # Convert raw question info to QuestionInfo object
        info = QuestionInfo(question_info)
        
        # Start with example test cases if provided
        test_cases = []
        for i, example in enumerate(info.raw_info['examples']):
            test_cases.append({
                "test_id": i + 1,
                "is_example": True,
                "inputs": example["inputs"],
                "expected_output": example["output"]
            })
        
        # Generate additional test cases
        remaining_tests = num_tests - len(test_cases)
        for i in range(remaining_tests):
            test_case = self._generate_single_test_case(info, i + len(test_cases) + 1)
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_single_test_case(self, info: QuestionInfo, test_id: int) -> Dict[str, Any]:
        """Generate a single test case"""
        inputs = {}
        
        # Handle each parameter
        for param in info.raw_info['parameters']:
            param_name = param['name']
            
            # Skip optional parameters based on probability
            if (info.is_optional_parameter(param_name) and 
                random.random() < self.skip_optional_probability):
                continue
            
            # Generate parameter value based on type
            param_type = param['type']
            inputs[param_name] = self._generate_parameter_value(param_type)
        
        return {
            "test_id": test_id,
            "is_example": False,
            "inputs": inputs,
            "expected_output": self._compute_expected_output(info, inputs)
        }
    
    def _generate_parameter_value(self, param_type: str) -> Any:
        """Generate a value for a parameter based on its type"""
        # Add your parameter generation logic here
        # This is a simplified version
        if 'int' in param_type.lower():
            return random.randint(-100, 100)
        elif 'str' in param_type.lower():
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        elif 'list' in param_type.lower():
            return [random.randint(-10, 10) for _ in range(random.randint(1, 5))]
        return None
    
    def _compute_expected_output(self, info: QuestionInfo, inputs: Dict[str, Any]) -> Any:
        """Compute expected output for given inputs"""
        # This should be implemented based on your specific needs
        # For now, return None as placeholder
        return None
    
    def format_test_code(self, question_info: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> str:
        """
        Generate Python code for running the test cases
        
        Args:
            question_info: Parsed question information from LLMQuestionGenerator
            test_cases: Test cases generated by generate_test_cases
            
        Returns:
            Python code that can be executed to test a solution
        """
        function_name = question_info["function_name"]
        
        code = [
            "def run_tests(user_solution):",
            "    test_results = []",
            "    total_tests = 0",
            "    passed_tests = 0",
            ""
        ]
        
        # Add test cases
        for i, test in enumerate(test_cases):
            inputs_str = ", ".join([f"{k}={repr(v)}" for k, v in test["inputs"].items()])
            expected = repr(test["expected_output"])
            
            code.extend([
                f"    # Test case {i+1}",
                f"    try:",
                f"        total_tests += 1",
                f"        result = user_solution({inputs_str})",
                f"        expected = {expected}",
                f"        passed = result == expected",
                f"        if passed:",
                f"            passed_tests += 1",
                f"        test_results.append({{",
                f"            'test_id': {test['test_id']},",
                f"            'inputs': {repr(test['inputs'])},",
                f"            'expected_output': expected,",
                f"            'actual_output': result,",
                f"            'passed': passed,",
                f"            'is_example': {bool(test.get('is_example', False))}",
                f"        }})",
                f"    except Exception as e:",
                f"        test_results.append({{",
                f"            'test_id': {test['test_id']},",
                f"            'inputs': {repr(test['inputs'])},",
                f"            'expected_output': {expected},",
                f"            'error': str(e),",
                f"            'passed': False,",
                f"            'is_example': {bool(test.get('is_example', False))}",
                f"        }})",
                ""
            ])
        
        # Add return statement
        code.extend([
            "    return {",
            "        'total_tests': total_tests,",
            "        'passed_tests': passed_tests,",
            "        'passing_ratio': passed_tests / total_tests if total_tests > 0 else 0,",
            "        'results': test_results",
            "    }"
        ])
        
        return "\n".join(code)


class CodeTester:
    """Class for executing user code and running tests"""
    
    def __init__(self):
        self.temp_dir = Path(os.getcwd()) / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
    def cleanup_old_files(self, max_age_seconds: int = 3600):
        """Clean up old temporary files"""
        current_time = time.time()
        for file_path in self.temp_dir.glob("*"):
            if current_time - file_path.stat().st_mtime > max_age_seconds:
                try:
                    file_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def sanitize_module_name(name: str) -> str:
        """Ensure module name is valid Python identifier"""
        return f"code_mod_{abs(hash(name)) % 10000}"

    def execute_code(self, user_code: str, function_name: str, test_code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute the user's code and run tests with improved security and error handling
        """
        # Clean up old files first
        self.cleanup_old_files()
        
        # Generate unique execution ID
        execution_id = str(uuid.uuid4())
        module_name = self.sanitize_module_name(execution_id)
        
        code_file = self.temp_dir / f"{module_name}.py"
        test_file = self.temp_dir / f"test_{module_name}.py"
        
        try:
            # Validate inputs
            if not isinstance(user_code, str) or not isinstance(function_name, str):
                raise ValueError("Invalid input types")
            
            if not function_name.isidentifier():
                raise ValueError("Invalid function name")
            
            # Write files with proper encoding and error handling
            code_file.write_text(user_code, encoding='utf-8')
            
            # Create test file with proper imports and safety measures
            test_code_content = [
                "import sys",
                "import signal",
                "from contextlib import contextmanager",
                "",
                "# Set up timeout handler",
                "@contextmanager",
                "def timeout(seconds):",
                "    def signal_handler(signum, frame):",
                "        raise TimeoutError('Execution timed out')",
                "    signal.signal(signal.SIGALRM, signal_handler)",
                "    signal.alarm(seconds)",
                "    try:",
                "        yield",
                "    finally:",
                "        signal.alarm(0)",
                "",
                f"sys.path.insert(0, '{self.temp_dir}')",
                f"import {module_name}",
                f"{function_name} = {module_name}.{function_name}",
                "",
                test_code,
                "",
                "import json",
                "with timeout(4):",  # Inner timeout as additional safety
                f"    results = run_tests({function_name})",
                "print(json.dumps(results))"
            ]
            
            test_file.write_text('\n'.join(test_code_content), encoding='utf-8')
            
            # Execute in subprocess with resource limits
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONPATH': str(self.temp_dir)},
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': result.stderr[:500],  # Limit error message size
                    'total_tests': 0,
                    'passed_tests': 0,
                    'passing_ratio': 0,
                    'results': []
                }
            
            try:
                return {
                    'success': True,
                    **json.loads(result.stdout)
                }
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': 'Invalid test output format',
                    'total_tests': 0,
                    'passed_tests': 0,
                    'passing_ratio': 0,
                    'results': []
                }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Execution timed out after {timeout} seconds',
                'total_tests': 0,
                'passed_tests': 0,
                'passing_ratio': 0,
                'results': []
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)[:500],  # Limit error message size
                'total_tests': 0,
                'passed_tests': 0,
                'passing_ratio': 0,
                'results': []
            }
        
        finally:
            # Clean up files
            try:
                code_file.unlink(missing_ok=True)
                test_file.unlink(missing_ok=True)
            except Exception:
                pass


# Initialize Flask app with proper configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max-limit
app.config['MAX_CACHE_SIZE'] = 1000  # 1000 questions in cache
app.config['CACHE_TTL'] = 3600  # 1 hour TTL
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Initialize components
question_generator = LLMQuestionGenerator()
test_case_generator = TestCaseGenerator()
code_tester = CodeTester()

# Use a proper cache with expiration
from cachetools import TTLCache
questions_db = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL

@app.before_request
def validate_request():
    """Validate incoming requests"""
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        if request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'success': False,
                'error': 'Request too large'
            }), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate-question', methods=['POST'])
def generate_question():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
            
        difficulty = data.get('difficulty', 'medium')
        topic = data.get('topic')
        
        if difficulty not in ['easy', 'medium', 'hard']:
            return jsonify({'success': False, 'error': 'Invalid difficulty level'}), 400
        
        try:
            question_info = question_generator.generate_question(difficulty, topic)
            logger.info(f"Generated question info: {json.dumps(question_info)}")
            
            if not question_info.get('success'):
                logger.error(f"Question generation failed: {question_info.get('error', 'Unknown error')}")
                return jsonify(question_info), 400
                
            test_cases = test_case_generator.generate_test_cases(question_info['question'])
            test_code = test_case_generator.format_test_code(question_info['question'], test_cases)
            
            question_id = str(uuid.uuid4())
            questions_db[question_id] = {
                'question_info': question_info['question'],
                'test_cases': test_cases,
                'test_code': test_code,
                'created_at': time.time()
            }
            
            session['current_question_id'] = question_id
            
            return jsonify({
                'success': True,
                'question_id': question_id,
                'question_info': question_info['question'],
                'example_test_cases': [tc for tc in test_cases if tc.get('is_example', False)]
            })
            
        except Exception as e:
            logger.exception("Error in question generation process")
            return jsonify({
                'success': False,
                'error': f'Question generation failed: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.exception("Error in generate_question endpoint")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/submit-solution', methods=['POST'])
def submit_solution():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
            
        question_id = data.get('question_id')
        code = data.get('code', '').strip()
        
        if not code:
            return jsonify({'success': False, 'error': 'No code provided'}), 400
            
        question_data = questions_db.get(question_id)
        if not question_data:
            return jsonify({
                'success': False,
                'error': 'Question not found or expired'
            }), 404
        
        function_name = question_data['question_info']['function_name']
        test_code = question_data['test_code']
        
        results = code_tester.execute_code(code, function_name, test_code)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get-question/<question_id>', methods=['GET'])
def get_question(question_id):
    try:
        question_data = questions_db.get(question_id)
        if not question_data:
            return jsonify({
                'success': False,
                'error': 'Question not found or expired'
            }), 404
        
        return jsonify({
            'success': True,
            'question_info': question_data['question_info'],
            'example_test_cases': [tc for tc in question_data['test_cases'] if tc.get('is_example', False)]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    return jsonify({
        'success': False,
        'error': str(error)
    }), 500

# Add favicon route to handle favicon.ico requests
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return empty response with "No Content" status

if __name__ == '__main__':
    try:
        # Initialize components
        question_generator = LLMQuestionGenerator()
        test_case_generator = TestCaseGenerator()
        code_tester = CodeTester()
        
        # Initialize cache
        from cachetools import TTLCache
        questions_db = TTLCache(
            maxsize=app.config['MAX_CACHE_SIZE'],
            ttl=app.config['CACHE_TTL']
        )
        
        # Start the application
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=False
        )
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)  # Set debug=False in production

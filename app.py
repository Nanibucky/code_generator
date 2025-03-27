import os
import sys
import json
import time
import uuid
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import random
import openai
from flask import Flask, request, jsonify, render_template, session
import logging
from cachetools import TTLCache
import os
from flask import request, jsonify
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max-limit
app.config['MAX_CACHE_SIZE'] = 1000  # 1000 questions in cache
app.config['CACHE_TTL'] = 3600  # 1 hour TTL
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@app.route('/api/chat-completion', methods=['POST'])
def chat_completion():
    """Handle chat completion requests from the Coding Buddy"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request data',
                'message': "I couldn't process your message. Please try again."
            }), 400

        user_message = data.get('user_message', '').strip()
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Empty message',
                'message': "Please type a message first."
            }), 400

        # Check if OpenAI API key is set
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key is not set")
            return jsonify({
                'success': False,
                'error': 'API key not configured',
                'message': "OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
            }), 500

        # Construct the context for the AI
        system_prompt = """You are a helpful coding assistant. Help the user with their programming questions and challenges."""
        user_context = f"USER QUESTION: {user_message}"

        # Add additional context if available
        if data.get('question_info'):
            user_context += f"\n\nCURRENT QUESTION: {data['question_info']}"
        if data.get('code_solution'):
            user_context += f"\n\nUSER'S CODE: {data['code_solution']}"
        if data.get('test_results'):
            user_context += f"\n\nTEST RESULTS: {data['test_results']}"

        logger.info("Calling OpenAI API...")
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            ai_message = response.choices[0].message.content.strip()
            logger.info("Successfully received OpenAI response")
            
            return jsonify({
                'success': True,
                'message': ai_message
            })
            
        except Exception as openai_error:
            logger.error(f"OpenAI API error: {str(openai_error)}")
            return jsonify({
                'success': False,
                'error': str(openai_error),
                'message': "Error connecting to OpenAI API. Please check your API key and try again."
            }), 500

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': "An unexpected error occurred. Please try again."
        }), 500
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
        # Improved regex to handle multiline signatures and more flexible whitespace
        if "def " in function_signature:
            # For function-based questions
            signature_line = [line for line in function_signature.split('\n') if "def " in line][0]
            match = re.match(r'def\s+(\w+)\s*\((.*?)\)', signature_line)
            
            if not match:
                # Try a more flexible regex as fallback
                match = re.search(r'def\s+(\w+)\s*\((.*?)\)(?:\s*->.*?)?:', function_signature, re.DOTALL)
                
            if not match:
                logger.error(f"Failed to parse function signature: {function_signature}")
                # Default parameters to avoid crashing
                function_name = "solution"
                params = [("input", None)]
            else:
                function_name = match.group(1)
                params_str = match.group(2)
                
                # Parse parameters
                params = []
                if params_str:
                    for param in params_str.split(','):
                        param = param.strip()
                        if not param:
                            continue
                        if ':' in param:
                            name, type_hint = param.split(':', 1)
                            params.append((name.strip(), type_hint.strip()))
                        else:
                            params.append((param, None))
        elif "class " in function_signature:
            # Handle class-based questions
            match = re.match(r'class\s+(\w+)', function_signature)
            if match:
                function_name = match.group(1)
                params = [("self", None)]  # Default parameter for class
            else:
                function_name = "Solution"
                params = [("self", None)]
        else:
            # Fallback for unrecognized formats
            logger.warning(f"Unrecognized function signature format: {function_signature}")
            function_name = "solution"
            params = [("input", None)]
        
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
                if param_name != 'self':  # Skip 'self' parameter for class methods
                    test_case["inputs"][param_name] = Pynguine._generate_sample_input(param_name, param_type, description)
            
            # Generate expected output based on function name and inputs
            test_case["expected_output"] = Pynguine._generate_expected_output(
                function_name, test_case["inputs"], description
            )
            
            test_cases.append(test_case)
            
        return test_cases
    
    @staticmethod
    def _generate_sample_input(param_name: str, param_type: Optional[str], description: str) -> Any:
        """Generate simpler sample input based on parameter name and type hint"""
        # Check type hint first if available
        if param_type:
            if 'int' in param_type:
                return random.randint(-10, 10)  # Smaller range than before
            elif 'float' in param_type:
                return round(random.uniform(-10, 10), 1)  # Fewer decimal places
            elif 'str' in param_type:
                return f"test_{random.randint(1, 5)}"  # Shorter strings
            elif 'list' in param_type or 'List' in param_type:
                # Try to determine list type, but keep lists shorter
                if 'int' in param_type:
                    return [random.randint(-5, 5) for _ in range(random.randint(2, 4))]
                elif 'str' in param_type:
                    return [f"item_{i}" for i in range(random.randint(2, 3))]
                else:
                    return [random.randint(-5, 5) for _ in range(random.randint(2, 4))]
            elif 'dict' in param_type or 'Dict' in param_type:
                return {f"k{i}": random.randint(1, 5) for i in range(random.randint(1, 3))}
            elif 'bool' in param_type:
                return random.choice([True, False])
        
        # If no type hint or unsupported type, infer from parameter name
        if 'num' in param_name or 'count' in param_name or 'index' in param_name:
            return random.randint(1, 10)  # Smaller numbers
        elif 'name' in param_name or 'text' in param_name or 'str' in param_name:
            return f"sample_{random.randint(1, 3)}"  # Shorter strings
        elif 'list' in param_name or 'array' in param_name:
            return [random.randint(1, 5) for _ in range(random.randint(2, 4))]  # Shorter lists
        elif 'dict' in param_name or 'map' in param_name:
            return {f"k{i}": random.randint(1, 5) for i in range(random.randint(1, 2))}  # Smaller dicts
        elif 'flag' in param_name or 'enable' in param_name:
            return random.choice([True, False])
        else:
            # Default to a simple string
            return f"input_{random.randint(1, 3)}"
    
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
        self.use_local_questions = False  # Always attempt to use LLM first

    def generate_question(self, difficulty: str = "medium", topic: str = None) -> Dict[str, Any]:
        """Generate a Python coding question using LLM"""
        try:
            # Use OpenAI API to generate the question
            if OPENAI_API_KEY:
                try:
                    question_text = self._generate_raw_question(difficulty, topic)
                    question_info = self._parse_question(question_text)
                    
                    # If we don't have a valid question, fall back to default
                    if not question_info.get('function_name'):
                        logger.warning("LLM generated question didn't have required fields")
                        return self._get_local_question(difficulty, topic)
                    
                    question_info['difficulty'] = difficulty
                    return {"success": True, "question": question_info}
                except Exception as e:
                    logger.error(f"Error generating question with LLM: {e}")
                    return self._get_local_question(difficulty, topic)
            else:
                logger.info("No OpenAI API key provided, using local questions")
                return self._get_local_question(difficulty, topic)
                
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return self._get_local_question("medium", None)  # Default fallback

    def _get_local_questions(self) -> Dict[str, List[Dict[str, str]]]:
        """Return a dictionary of pre-defined local questions for fallback"""
        return {
            "easy": [
                {
                    "problem": """Write a function that finds the second largest element in a list of integers. If the list has fewer than 2 elements, return -1.""",
                    "signature": "def find_second_largest(arr: List[int]) -> int:",
                    "examples": [
                        ("Input: [1, 3, 2, 5, 4]", "Output: 4"),
                        ("Input: [1, 1, 1]", "Output: -1"),
                        ("Input: [7]", "Output: -1")
                    ],
                    "constraints": [
                        "1 <= len(arr) <= 10^5",
                        "-10^9 <= arr[i] <= 10^9"
                    ]
                },
                {
                    "problem": """Write a function that counts the number of vowels in a given string.""",
                    "signature": "def count_vowels(text: str) -> int:",
                    "examples": [
                        ("Input: 'hello'", "Output: 2"),
                        ("Input: 'PYTHON'", "Output: 1"),
                        ("Input: ''", "Output: 0")
                    ],
                    "constraints": [
                        "0 <= len(text) <= 10^4",
                        "text consists of printable ASCII characters"
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
                    ],
                    "constraints": [
                        "0 <= len(str1), len(str2) <= 5 * 10^4",
                        "str1 and str2 consist of printable ASCII characters"
                    ]
                },
                {
                    "problem": """Write a function that finds the first non-repeating character in a string.""",
                    "signature": "def first_unique_char(s: str) -> str:",
                    "examples": [
                        ("Input: 'leetcode'", "Output: 'l'"),
                        ("Input: 'hello'", "Output: 'h'"),
                        ("Input: 'aabb'", "Output: ''")
                    ],
                    "constraints": [
                        "1 <= len(s) <= 10^5",
                        "s consists of only lowercase English letters"
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
                    ],
                    "constraints": [
                        "1 <= len(s) <= 1000",
                        "s consists of only lowercase English letters"
                    ]
                },
                {
                    "problem": """Write a function that implements a LRU (Least Recently Used) cache with a given capacity.""",
                    "signature": "class LRUCache:",
                    "examples": [
                        ("cache = LRUCache(2)", "None"),
                        ("cache.put(1, 1)", "None"),
                        ("cache.get(1)", "Returns: 1")
                    ],
                    "constraints": [
                        "1 <= capacity <= 3000",
                        "0 <= key <= 10^4",
                        "0 <= value <= 10^5",
                        "At most 2 * 10^5 calls will be made to get and put"
                    ]
                }
            ]
        }

    def _get_local_question(self, difficulty: str, topic: str = None) -> Dict[str, Any]:
        """Get a random local question based on difficulty (for fallback)"""
        questions = self._get_local_questions()
        difficulty = difficulty.lower() if difficulty else "medium"
        if difficulty not in questions:
            difficulty = "medium"
        
        question = random.choice(questions[difficulty])
        
        # Format examples for frontend
        formatted_examples = []
        for input_text, output_text in question["examples"]:
            formatted_examples.append({
                "input_text": input_text,
                "output_text": output_text
            })
        
        return {
            "success": True,
            "question": {
                "problem_statement": question["problem"],
                "function_signature": question["signature"],
                "examples": formatted_examples,
                "constraints": question["constraints"],
                "difficulty": difficulty,
                "function_name": self._extract_function_name(question["signature"])
            }
        }
    
    def _extract_function_name(self, signature: str) -> str:
        """Extract function name from signature"""
        if signature.startswith("class"):
            # For class-based questions, return the class name
            match = re.match(r'class\s+(\w+)', signature)
            if match:
                return match.group(1)
            return "Solution"
        else:
            # For function-based questions
            match = re.match(r'def\s+(\w+)', signature)
            if match:
                return match.group(1)
            return "solution"

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
            raise

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
        
        Constraints:
        [List any constraints on input size, value ranges, etc.]
        """

    def _parse_question(self, question_text: str) -> Dict[str, Any]:
        """Parse the raw question text into structured format"""
        try:
            # Basic validation of question text
            if not question_text or not isinstance(question_text, str):
                logger.error(f"Invalid question text received: {question_text}")
                raise ValueError("Invalid question text format")

            # Extract components
            lines = question_text.strip().split('\n')
            
            problem_statement = ""
            function_signature = ""
            imports = []
            examples = []
            constraints = []
            function_name = None
            
            current_section = None
            current_example = {}
            in_code_block = False
            code_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                # Handle code blocks with triple backticks
                if line_stripped == '```' or line_stripped == '```python':
                    in_code_block = not in_code_block
                    if line_stripped == '```python':
                        current_section = "signature"
                    continue
                    
                if line_stripped.startswith('Problem Statement:'):
                    current_section = "problem"
                    continue
                elif line_stripped.startswith('Function Signature:'):
                    current_section = "signature"
                    continue
                elif line_stripped.startswith('Examples:'):
                    current_section = "examples"
                    continue
                elif line_stripped.startswith('Constraints:'):
                    current_section = "constraints"
                    continue
                elif line_stripped.startswith('from typing import'):
                    imports.append(line_stripped)
                    if current_section != "signature":
                        current_section = "signature"
                    continue
                
                if in_code_block and (current_section == "signature" or not current_section):
                    # Collect all code lines within triple backticks
                    code_lines.append(line)
                    if line_stripped.startswith('def '):
                        match = re.match(r'def\s+(\w+)\s*\(', line_stripped)
                        if match:
                            function_name = match.group(1)
                    elif line_stripped.startswith('class '):
                        match = re.match(r'class\s+(\w+)', line_stripped)
                        if match:
                            function_name = match.group(1)
                    continue
                
                if current_section == "problem":
                    problem_statement += line + " "
                elif current_section == "signature" and not in_code_block:
                    if line_stripped.startswith('def ') or line_stripped.startswith('class '):
                        function_signature = line_stripped
                        # Extract function name from signature
                        if line_stripped.startswith('def '):
                            match = re.match(r'def\s+(\w+)\s*\(', line_stripped)
                            if match:
                                function_name = match.group(1)
                        else:
                            match = re.match(r'class\s+(\w+)', line_stripped)
                            if match:
                                function_name = match.group(1)
                elif current_section == "examples":
                    if line_stripped.lower().startswith('input:'):
                        if current_example.get('input_text'):  # Save previous example if exists
                            examples.append(current_example.copy())
                        current_example = {'input_text': line}
                    elif line_stripped.lower().startswith('output:') and current_example.get('input_text'):
                        current_example['output_text'] = line
                        examples.append(current_example.copy())
                        current_example = {}
                elif current_section == "constraints":
                    constraints.append(line_stripped)

            # If we collected code inside triple backticks, use it
            if code_lines and not function_signature:
                function_signature = '\n'.join(code_lines)

            # Add last example if pending
            if current_example.get('input_text') and current_example.get('output_text'):
                examples.append(current_example)

            # If List is used in signature but import is missing, add it
            if 'List[' in function_signature and not any('typing import List' in imp for imp in imports):
                imports.append('from typing import List')

            # Combine imports and function signature
            full_signature = '\n'.join(imports + [function_signature]) if imports else function_signature
            
            # Add default constraints if none were parsed
            if not constraints:
                if 'int' in full_signature.lower():
                    constraints.append('-10^9 <= values <= 10^9')
                if 'list' in full_signature.lower() or 'array' in full_signature.lower():
                    constraints.append('1 <= array length <= 10^5')
                if 'str' in full_signature.lower():
                    constraints.append('1 <= string length <= 10^4')
                
            # Ensure we have all required components
            if not problem_statement.strip():
                logger.error("No problem statement found")
                raise ValueError("Missing problem statement")
                
            if not full_signature.strip():
                logger.error("No function signature found")
                raise ValueError("Missing function signature")
                
            if not function_name:
                logger.warning("No function name extracted, using default")
                function_name = "solution"  # Default name
            
            # Log successful parsing
            logger.info(f"Successfully parsed question with function: {function_name}")
            
            return {
                "problem_statement": problem_statement.strip(),
                "function_signature": full_signature.strip(),
                "function_name": function_name,
                "examples": examples,
                "constraints": constraints
            }
            
        except Exception as e:
            logger.exception("Error parsing question")
            raise


class QuestionInfo:
    """Class to handle question information and validation"""
    
    def __init__(self, raw_info: Dict[str, Any]):
        self.raw_info = raw_info
        self.validate_and_normalize()
    
    def validate_and_normalize(self):
        """Validate and normalize question information"""
        required_fields = ['problem_statement', 'function_signature']
        for field in required_fields:
            if not self.raw_info.get(field):
                raise ValueError(f"Missing required field: {field}")
        
        # Make sure we have a function name
        if 'function_name' not in self.raw_info:
            self.raw_info['function_name'] = self._extract_function_name(self.raw_info['function_signature'])
            
        # Extract parameters from function signature if not already present
        if 'parameters' not in self.raw_info:
            self.raw_info['parameters'] = self._extract_parameters(self.raw_info['function_signature'])
        
        # Normalize parameters
        self.raw_info['parameters'] = self._normalize_parameters(self.raw_info['parameters'])
        
        # Normalize examples for test case generation
        if 'examples' in self.raw_info:
            # Convert examples to format for test case generation
            normalized_examples = []
            for example in self.raw_info['examples']:
                if 'input_text' in example and 'output_text' in example:
                    # Parse input and output text
                    inputs = self._parse_input_text(example['input_text'], self.raw_info['parameters'])
                    output = self._parse_output_text(example['output_text'])
                    
                    normalized_examples.append({
                        "inputs": inputs,
                        "output": output
                    })
            
            self.raw_info['normalized_examples'] = normalized_examples
                
        # Ensure constraints exist
        if 'constraints' not in self.raw_info or not self.raw_info['constraints']:
            self.raw_info['constraints'] = ['1 <= input size <= 10^5']

    def _extract_function_name(self, function_signature: str) -> str:
        """Extract function name from the signature"""
        if function_signature.startswith("class"):
            match = re.match(r'class\s+(\w+)', function_signature)
            if match:
                return match.group(1)
        else:
            match = re.match(r'def\s+(\w+)', function_signature)
            if match:
                return match.group(1)
        return "solution"  # Default name

    def _extract_parameters(self, function_signature: str) -> List[Dict[str, str]]:
        """Extract parameters from function signature"""
        # Check if it's a class
        if function_signature.startswith("class"):
            return []  # For class-based questions, don't extract params
            
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
            if not param or param == 'self':
                continue
                
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
    
    def _parse_input_text(self, input_text: str, parameters: List[Dict[str, str]]) -> Dict[str, Any]:
        """Parse the input text from examples into a dictionary of inputs"""
        # Remove "Input: " prefix
        input_text = input_text.replace("Input:", "").strip()
        
        # Try to parse as Python literal
        try:
            # For multi-parameter functions, try to match params by position
            input_values = eval(input_text)
            
            # If input_values is a tuple or list and we have multiple parameters
            if isinstance(input_values, (list, tuple)) and len(parameters) > 1:
                return {param['name']: val for param, val in zip(parameters, input_values)}
            
            # If we have a single parameter
            if len(parameters) == 1:
                return {parameters[0]['name']: input_values}
                
            # Default case
            if len(parameters) > 0:
                return {parameters[0]['name']: input_values}
            else:
                return {"input": input_values}
                
        except (SyntaxError, ValueError, NameError):
            # If we can't parse it directly, just use a default parameter name
            if len(parameters) > 0:
                return {parameters[0]['name']: input_text}
            else:
                return {"input": input_text}
    
    def _parse_output_text(self, output_text: str) -> Any:
        """Parse the output text from examples"""
        # Remove "Output: " prefix
        output_text = output_text.replace("Output:", "").strip()
        
        # Try to parse as Python literal
        try:
            return eval(output_text)
        except (SyntaxError, ValueError, NameError):
            # If we can't parse it, return as string
            return output_text
class TestCaseGenerator:
    """Main class for generating test cases from LLM Python questions"""
    
    def __init__(self, pynguine=None):
        self.pynguine = pynguine or Pynguine()
        self.skip_optional_probability = 0.7  # Probability to skip optional parameters
    
    def generate_test_cases(self, question_info: Dict[str, Any], num_tests: int = 8) -> List[Dict[str, Any]]:
        """Generate test cases for a question"""
        # Convert raw question info to QuestionInfo object
        info = QuestionInfo(question_info)
        
        # Use normalized examples if available
        examples = info.raw_info.get('normalized_examples', [])
        
        # Generate tests using Pynguine
        function_signature = info.raw_info['function_signature']
        problem_statement = info.raw_info['problem_statement']
        
        test_cases = self.pynguine.generate_tests(
            function_signature=function_signature,
            description=problem_statement,
            examples=examples,
            num_tests=num_tests
        )
        
        return test_cases
    
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
            # Ensure inputs is a dictionary
            if not isinstance(test["inputs"], dict):
                test["inputs"] = {"input": test["inputs"]}
                
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
app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max-limit
app.config['MAX_CACHE_SIZE'] = 1000  # 1000 questions in cache
app.config['CACHE_TTL'] = 3600  # 1 hour TTL
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Initialize components
question_generator = LLMQuestionGenerator()
test_case_generator = TestCaseGenerator(Pynguine())
code_tester = CodeTester()

# Use a proper cache with expiration
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
            # Generate the question using LLM
            question_response = question_generator.generate_question(difficulty, topic)
            logger.info(f"Generated question with difficulty: {difficulty}")
            
            if not question_response.get('success'):
                logger.error(f"Question generation failed: {question_response.get('error', 'Unknown error')}")
                return jsonify(question_response), 400
                
            question_info = question_response['question']
            
            # Ensure we have required fields in question_info
            required_fields = ['problem_statement', 'function_signature', 'function_name']
            for field in required_fields:
                if field not in question_info:
                    return jsonify({
                        'success': False, 
                        'error': f'Generated question missing required field: {field}'
                    }), 500
            
            # Generate test cases using Pynguine
            test_cases = test_case_generator.generate_test_cases(question_info)
            test_code = test_case_generator.format_test_code(question_info, test_cases)
            
            # Store the question and test cases
            question_id = str(uuid.uuid4())
            questions_db[question_id] = {
                'question_info': question_info,
                'test_cases': test_cases,
                'test_code': test_code,
                'created_at': time.time()
            }
            
            session['current_question_id'] = question_id
            
            # Extract only example test cases for the response
            example_test_cases = [tc for tc in test_cases if tc.get('is_example', False)]
            
            return jsonify({
                'success': True,
                'question_id': question_id,
                'question_info': question_info,
                'example_test_cases': example_test_cases
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
        
        # Extract only example test cases for the response
        example_test_cases = [tc for tc in question_data['test_cases'] if tc.get('is_example', False)]
        
        return jsonify({
            'success': True,
            'question_info': question_data['question_info'],
            'example_test_cases': example_test_cases
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
        # Start the application
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=False
        )
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)  # Set debug=False in production

"""
Bfcl postprocessor module for processing function call predictions.
"""
import ast
import json
import logging
import re

from models.model_response import ModelResponse
from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)
logger.propagate = True


class BfclPostprocessor(Postprocessor):
    """
    Postprocessor for bfcl predictions.
    """

    @staticmethod
    def extract_json_from_message(message: str):
        """
        Extracts the JSON object from a message.
        """

        def fix_json_like_string(s):
            # Add double quotes to all keys
            s = re.sub(r'([{,])\s*([a-zA-Z_][\w\.]*)\s*:', r'\1"\2":', s)
            return s

        def decode_json(json_str):
            try:
                json_decode = json.loads(json_str)
                return json_decode
            except:
                try:
                    json_decode = ast.literal_eval(json_str)
                    return json_decode
                except:
                    return None
        decoded_json = decode_json(message)
        if decoded_json is not None:
            return decoded_json
        else:
            pattern = r"```json(.*?)```"
            match = re.search(pattern, message, re.DOTALL)
            if match:
                # Remove leading/trailing whitespace and parse JSON
                json_str = fix_json_like_string(match.group(1).strip())
                return decode_json(json_str)

    def process(
            self,
            dataset: list[dict],
            predictions: ModelResponse,
            metric
    ) -> tuple[list[tuple[str, str]], dict[str, list[str]], list, list] | dict:
        """
        Process and clean model predictions and prepare target-label pairs.
        """
        logger.info("Processing predictions with Bfcl Postprocessor...")

        processed_predictions: dict[str, list[str]] = {}
        for model_name, preds in predictions.items():
            processed = []
            for pred, dataset_row in zip(preds, dataset):
                tool_responses = []
                if isinstance(pred.raw_response, dict):
                    tools = pred.raw_response.get('choices', [])[0]['message']['tool_calls']
                    raw_llm_response = pred.llm_response
                    raw_tool_responses = tools
                else:
                    tools = None
                    tool_responses = None
                    raw_llm_response = None
                    raw_tool_responses = None
                if dataset_row.get('tools', None) is None:
                    # We ran in prediction in prompt mode
                    # Try Python format first, fall back to JSON
                    tool_responses = self._parse_python_function_calls(pred.llm_response.strip())
                    if tool_responses is None:
                        tool_responses = self.extract_json_from_message(pred.llm_response.strip())
                        tool_responses = self._normalize_tool_responses(tool_responses)
                    pred.llm_response = ''
                if tools:
                    for tool in tools:
                        tool_name = tool['function']['name']
                        tool_arguments = json.loads(tool['function']['arguments'])

                        tool_responses.append({tool_name: tool_arguments})

                processed_pred = {"llm_response": pred.llm_response.strip(),
                                  "tool_response": tool_responses,
                                  "raw_tool_response": raw_tool_responses,
                                  "raw_llm_response": raw_llm_response}
                processed.append(processed_pred)
            processed_predictions[model_name] = processed

        output = {
            "instructions": [record.get("instruction", "") for record in dataset],
            "model_targets": [record["model_target"] for record in dataset if "model_target" in record],
            "processed_predictions": processed_predictions,
        }
        self.validate_output(output)
        return output

    @staticmethod
    def _parse_python_function_calls(response: str):
        """
        Parse Python function calls in format: [func_name1(arg1=val1, ...), func_name2(...)]
        Convert to dict format: [{"func_name1": {"arg1": val1, ...}}, ...]
        Uses ast.literal_eval to handle Python literals safely.
        """
        try:
            # Wrap in brackets if needed
            response = response.strip()
            if not response.startswith("["):
                response = "[" + response
            if not response.endswith("]"):
                response = response + "]"

            # Parse the Python AST
            import ast
            tree = ast.parse(response, mode="eval")

            # Extract function calls
            function_calls = []
            if isinstance(tree.body, ast.List):
                for elem in tree.body.elts:
                    if isinstance(elem, ast.Call):
                        call_dict = BfclPostprocessor._resolve_ast_call(elem)
                        if call_dict:
                            function_calls.append(call_dict)

            return function_calls if function_calls else None
        except Exception as e:
            logger.warning(f"Failed to parse Python function calls: {e}")
            return None

    @staticmethod
    def _resolve_ast_call(elem):
        """Convert ast.Call to dict format {"func_name": {"arg": value, ...}}"""
        try:
            # Extract function name
            func_parts = []
            func_part = elem.func
            while isinstance(func_part, ast.Attribute):
                func_parts.append(func_part.attr)
                func_part = func_part.value
            if isinstance(func_part, ast.Name):
                func_parts.append(func_part.id)
            func_name = ".".join(reversed(func_parts))

            # Extract arguments
            args_dict = {}
            for arg in elem.keywords:
                value = BfclPostprocessor._resolve_ast_value(arg.value)
                args_dict[arg.arg] = value

            return {func_name: args_dict}
        except Exception as e:
            logger.warning(f"Failed to resolve AST call: {e}")
            return None

    @staticmethod
    def _resolve_ast_value(value):
        """Recursively resolve AST values to Python literals"""
        if isinstance(value, ast.Constant):
            return value.value
        elif isinstance(value, ast.List):
            return [BfclPostprocessor._resolve_ast_value(v) for v in value.elts]
        elif isinstance(value, ast.Dict):
            return {BfclPostprocessor._resolve_ast_value(k): BfclPostprocessor._resolve_ast_value(v)
                    for k, v in zip(value.keys, value.values)}
        elif isinstance(value, ast.Name):
            return value.id
        elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            return -BfclPostprocessor._resolve_ast_value(value.operand)
        elif isinstance(value, ast.Tuple):
            return tuple(BfclPostprocessor._resolve_ast_value(v) for v in value.elts)
        elif isinstance(value, ast.Call):
            # Nested function call
            call_dict = BfclPostprocessor._resolve_ast_call(value)
            return call_dict if call_dict else None
        else:
            return None

    @staticmethod
    def _normalize_tool_responses(tool_responses):
        """
        Normalize tool responses from various formats to {func_name: {params}} format.
        Handles: [{"func_name": "name", "params": {...}}, ...] -> [{"name": {...}}, ...]
        """
        if not isinstance(tool_responses, list):
            return tool_responses

        normalized = []
        for item in tool_responses:
            if isinstance(item, dict):
                # Check if it's in {"func_name": "...", "params": {...}} format
                if "func_name" in item and "params" in item:
                    func_name = item["func_name"]
                    params = item["params"]
                    normalized.append({func_name: params})
                # Otherwise assume it's already in {func_name: {params}} format
                elif len(item) == 1:
                    normalized.append(item)
                else:
                    # Fallback: keep as is
                    normalized.append(item)
            else:
                normalized.append(item)
        return normalized

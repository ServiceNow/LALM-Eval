"""Multiple Choice Question metrics implementation for GPQA Diamond. 

Evaluates model performance on multiple choice questions by extracting the predicted
answer choice (A-J) and comparing it to the reference answer.
"""
import re
from typing import List, Dict, Optional, Tuple, Any, Union

from metrics.metrics import Metrics
from utils import util
from utils.custom_logging import write_record_log, append_final_score


class MultipleChoiceMetrics(Metrics):
    """Multiple Choice Question evaluation metric.
    
    Computes accuracy for multiple choice questions by extracting the predicted
    answer choice (A-J) and comparing it to the reference answer.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "multiple_choice_accuracy"
        self.instructions = None
        self.model_responses = []
        self.record_level_scores = None
    
    def __call__(
        self, 
        candidates: List[str], 
        references: List[str], 
        instructions: Optional[str] = None, 
        *,
        task_name: Optional[str] = None, 
        model_name: Optional[str] = None, 
        model_responses: Optional[List[Any]] = None
    ) -> Dict[str, float]:
        """Evaluate multiple choice accuracy and optionally log results.
        
        Args:
            candidates: List of model-generated text responses
            references: List of reference answers (single letters A-J)
            instructions: Optional instructions text
            task_name: Task identifier for logging
            model_name: Model identifier for logging
            model_responses: Optional model responses for logging
            
        Returns:
            Dictionary with accuracy percentage under 'multiple_choice_accuracy' key
        """
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        
        scores, normalized_candidates, normalized_references = self.compute_record_level_scores(candidates, references)
        overall = self.get_score(candidates, references)
        
        if task_name and model_name:
            score_list = scores.get(self.name, [])
            write_record_log(
                self, 
                normalized_references, 
                normalized_candidates, 
                score_list, 
                task_name, 
                model_name,
                instructions=self.instructions, 
                model_responses=self.model_responses
            )
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        
        return overall
    
    def _extract_mc_answer(self, prediction: str) -> Optional[str]:
        """
        Extracts the multiple-choice answer letter (A-J) from a prediction string.
        Uses a staged approach: try the primary pattern first, then fallbacks in order.
        Returns the last match from the first successful pattern, or None if nothing found.
        Patterns based on: https://artificialanalysis.ai/methodology/intelligence-benchmarking
        
        Args:
            prediction: The model's prediction string
            
        Returns:
            Uppercase letter (A-J) if found, None otherwise
        """
        if not isinstance(prediction, str):
            return None
            
        patterns = [
            # Primary pattern: Answer: X
            r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-J])(?![a-zA-Z0-9])",
            # LaTeX boxed notation
            r"\\\\boxed\{[^}]*([A-J])[^}]*\}",
            # Natural language
            r"answer is ([a-jA-J])",
            # With parenthesis
            r"answer is\s*\(\s*([a-jA-J])\s*\)",
            # Choice format: "D) ..."
            r"([A-J])\)\s*[^A-J]*",
            # Explicit statement: "E is the correct answer"
            r"([A-J])\s+is\s+the\s+correct\s+answer",
            # Standalone letter at end
            r"([A-J])\s*$",
            # Letter followed by period
            r"([A-J])\s*\\.",
            # Letter followed by non-word character
            r"([A-J])\s*[^\w]",
        ]
        
        for pat in patterns:
            matches = re.findall(pat, prediction, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
                
        return prediction
    
    
    def compute_record_level_scores(
        self, 
        candidates: List[str], 
        references: List[str]
    ) -> Tuple[Dict[str, List[float]], List[str], List[str]]:
        """Compute per-record scores for multiple choice answers.
        
        Args:
            candidates: List of model-generated text responses
            references: List of reference answers (single letters A-J)
            
        Returns:
            Tuple of (scores dict, normalized candidates, normalized references)
        """
        if len(candidates) != len(references):
            raise ValueError(f"Mismatched lengths: {len(candidates)} candidates vs {len(references)} references")
        
        scores = []
        normalized_candidates = []
        normalized_references = []
        
        for candidate, reference in zip(candidates, references):
            pred = self._extract_mc_answer(candidate)
            
            normalized_candidates.append(pred)
            normalized_references.append(reference)
            
            score = 1.0 if (pred is not None and pred == reference) else 0.0
            scores.append(score)
        
        return {self.name: scores}, candidates, references
    
    def get_score(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Compute overall accuracy percentage.
        
        Args:
            candidates: Generated text from the model
            references:  Reference text from the dataset
            
        Returns:
            Dictionary with accuracy percentage under metric name
        """

        if not self.record_level_scores:
            self.record_level_scores, _, _ = self.compute_record_level_scores(candidates, references)

        scores = self.record_level_scores.get(self.name, [])
        accuracy = (sum(scores) / len(scores) * 100.0 if scores else 0.0)
                
        return {self.name: util.smart_round(accuracy, 2)}
"""Dialogue state tracking metrics for task-oriented dialogue evaluation.

This module provides general-purpose evaluation metrics for dialogue systems:
- Joint Goal Accuracy (JGA): Whether all slots match the ground truth
- Slot Accuracy: Per-slot accuracy across all samples
- Slot F1: F1 score for slot value extraction

These metrics can be used by any task-oriented dialogue benchmark.
"""

import json
import logging
import re
from typing import Dict, List, Any

from tqdm import tqdm

from metrics.metrics import Metrics
from utils import util
from utils.custom_logging import write_record_log, append_final_score

logger = logging.getLogger(__name__)


class _BaseDialogueMetric(Metrics):
    """Base class for dialogue state tracking metrics."""
    
    def __init__(self):
        super().__init__()
        self.instructions = None
        self.model_responses = []
        self.ground_truth_slots = []

    def __call__(self, candidates, references, instructions=None, *, 
                 task_name: str | None = None, model_name: str | None = None, 
                 model_responses=None, ground_truth_slots=None):
        """Evaluate model predictions against ground truth.
        
        Args:
            candidates: Model generated responses
            references: Ground truth agent responses
            instructions: Input instructions (contains dialogue context)
            task_name: Name of the task
            model_name: Name of the model
            model_responses: Raw model response objects
            ground_truth_slots: List of ground truth slot dictionaries
        """
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        self.ground_truth_slots = ground_truth_slots if ground_truth_slots else []
        
        # Compute scores
        overall = self.get_score(candidates, references, task_name, model_name)
        
        if task_name and model_name:
            scores = self.record_level_scores
            write_record_log(self, references, candidates, scores, task_name, model_name,
                           instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        
        return overall

    def get_score(self, candidates: list, references: list, 
                  task_name: str = None, model_name: str = None) -> Dict[str, float]:
        """Compute overall scores for the dataset."""
        if not self.record_level_scores:
            self.record_level_scores = self.compute_record_level_scores(
                candidates, references, task_name, model_name
            )
        
        results = {}
        for metric_name, scores in self.record_level_scores.items():
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                results[metric_name] = util.smart_round(sum(valid_scores) / len(valid_scores) * 100, 2)
            else:
                results[metric_name] = 0.0
        
        return results

    def _extract_slots_from_response(self, response: str) -> Dict[str, Dict[str, str]]:
        """Extract slot values from model's natural language response.
        
        Uses pattern matching to identify slot-value pairs mentioned in the response.
        """
        if not response:
            return {}
        
        response_lower = response.lower()
        extracted_slots = {}
        
        # Pattern-based extraction for common slot types
        patterns = {
            'restaurant': {
                'area': r'(?:in|at|around|near)\s+(?:the\s+)?(\w+)\s+(?:area|part|side)',
                'food': r'(?:serving|serves?|offering?|type of food[:\s]+)(\w+(?:\s+\w+)?)\s+(?:food|cuisine)?',
                'pricerange': r'(?:price[d]?\s*range|budget|cost)[:\s]+(\w+)|(\w+)\s+(?:price[d]?\s*range|priced)',
                'name': r'(?:called|named|restaurant[:\s]+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            },
            'hotel': {
                'area': r'(?:in|at|around|near)\s+(?:the\s+)?(\w+)\s+(?:area|part|side)',
                'pricerange': r'(?:price[d]?\s*range|budget|cost)[:\s]+(\w+)|(\w+)\s+(?:price[d]?\s*range|priced)',
                'stars': r'(\d+)\s*(?:star|stars)',
                'parking': r'(?:parking)[:\s]*(yes|no|free)',
                'internet': r'(?:internet|wifi)[:\s]*(yes|no|free)',
                'type': r'(?:type)[:\s]+(\w+)|(\w+)\s+(?:type)',
                'name': r'(?:called|named|hotel[:\s]+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            },
            'train': {
                'departure': r'(?:from|depart(?:ing|ure)?(?:\s+from)?)[:\s]+([A-Za-z]+)',
                'destination': r'(?:to|arrive?(?:\s+at)?|destination)[:\s]+([A-Za-z]+)',
                'day': r'(?:on|day)[:\s]+(\w+day)',
                'leaveat': r'(?:leav(?:e|ing)\s+(?:at)?|departure\s+time)[:\s]+(\d{1,2}[:\.]?\d{2})',
                'arriveby': r'(?:arriv(?:e|ing)\s+(?:by)?|arrival\s+time)[:\s]+(\d{1,2}[:\.]?\d{2})',
            },
            'taxi': {
                'departure': r'(?:from|pick\s*up)[:\s]+([A-Za-z\s]+?)(?:\s+to|\s*$)',
                'destination': r'(?:to|drop\s*off|destination)[:\s]+([A-Za-z\s]+)',
                'leaveat': r'(?:leav(?:e|ing)\s+(?:at)?)[:\s]+(\d{1,2}[:\.]?\d{2})',
                'arriveby': r'(?:arriv(?:e|ing)\s+(?:by)?)[:\s]+(\d{1,2}[:\.]?\d{2})',
            },
            'attraction': {
                'area': r'(?:in|at|around|near)\s+(?:the\s+)?(\w+)\s+(?:area|part|side)',
                'type': r'(?:type)[:\s]+(\w+)|(\w+)\s+(?:attraction|place)',
                'name': r'(?:called|named|attraction[:\s]+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            }
        }
        
        for domain, slot_patterns in patterns.items():
            for slot, pattern in slot_patterns.items():
                matches = re.findall(pattern, response_lower, re.IGNORECASE)
                if matches:
                    value = None
                    for match in matches:
                        if isinstance(match, tuple):
                            value = next((m for m in match if m), None)
                        else:
                            value = match
                        if value:
                            break
                    
                    if value:
                        if domain not in extracted_slots:
                            extracted_slots[domain] = {}
                        extracted_slots[domain][slot] = value.strip()
        
        return extracted_slots

    def _normalize_value(self, value: str) -> str:
        """Normalize slot value for comparison."""
        if not value:
            return ''
        return str(value).lower().strip().replace(' ', '').replace('-', '').replace('_', '')

    def _parse_ground_truth_slots(self, gt_slots) -> Dict:
        """Parse ground truth slots from various formats."""
        if not gt_slots:
            return {}
        if isinstance(gt_slots, str):
            try:
                return json.loads(gt_slots)
            except json.JSONDecodeError:
                return {}
        return gt_slots


class JointGoalAccuracy(_BaseDialogueMetric):
    """Joint Goal Accuracy metric for dialogue state tracking.
    
    JGA is 1 if all predicted slots exactly match ground truth, 0 otherwise.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "joint_goal_accuracy"

    def compute_record_level_scores(self, candidates: list, references: list,
                                    task_name: str = None, model_name: str = None) -> Dict[str, List]:
        """Compute JGA for each sample."""
        scores = []
        
        desc = f"JGA Eval"
        if task_name and model_name:
            desc = f"JGA [{task_name}] [{model_name}]"
        
        for i, (candidate, reference) in enumerate(tqdm(
            zip(candidates, references), desc=desc, total=len(candidates)
        )):
            gt_slots = {}
            if i < len(self.ground_truth_slots):
                gt_slots = self._parse_ground_truth_slots(self.ground_truth_slots[i])
            
            pred_slots = self._extract_slots_from_response(candidate)
            jga = self._compute_jga(pred_slots, gt_slots)
            scores.append(jga)
        
        return {self.name: scores}

    def _compute_jga(self, pred_slots: Dict, gt_slots: Dict) -> float:
        """Compute Joint Goal Accuracy."""
        if not gt_slots:
            return 1.0 if not pred_slots else 0.0
        
        for domain, slots in gt_slots.items():
            if domain not in pred_slots:
                return 0.0
            for slot, value in slots.items():
                pred_value = pred_slots.get(domain, {}).get(slot, '')
                if self._normalize_value(pred_value) != self._normalize_value(value):
                    return 0.0
        
        return 1.0


class SlotAccuracy(_BaseDialogueMetric):
    """Slot Accuracy metric for dialogue state tracking.
    
    Computes the proportion of individual slots correctly predicted.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "slot_accuracy"

    def compute_record_level_scores(self, candidates: list, references: list,
                                    task_name: str = None, model_name: str = None) -> Dict[str, List]:
        """Compute slot accuracy for each sample."""
        scores = []
        
        desc = f"Slot Accuracy Eval"
        if task_name and model_name:
            desc = f"Slot Accuracy [{task_name}] [{model_name}]"
        
        for i, (candidate, reference) in enumerate(tqdm(
            zip(candidates, references), desc=desc, total=len(candidates)
        )):
            gt_slots = {}
            if i < len(self.ground_truth_slots):
                gt_slots = self._parse_ground_truth_slots(self.ground_truth_slots[i])
            
            pred_slots = self._extract_slots_from_response(candidate)
            slot_acc = self._compute_slot_accuracy(pred_slots, gt_slots)
            scores.append(slot_acc)
        
        return {self.name: scores}

    def _compute_slot_accuracy(self, pred_slots: Dict, gt_slots: Dict) -> float:
        """Compute per-slot accuracy."""
        if not gt_slots:
            return 1.0 if not pred_slots else 0.0
        
        total_slots = 0
        correct_slots = 0
        
        for domain, slots in gt_slots.items():
            for slot, value in slots.items():
                total_slots += 1
                pred_value = pred_slots.get(domain, {}).get(slot, '')
                if self._normalize_value(pred_value) == self._normalize_value(value):
                    correct_slots += 1
        
        return correct_slots / total_slots if total_slots > 0 else 1.0


class SlotF1(_BaseDialogueMetric):
    """Slot F1 metric for dialogue state tracking.
    
    Computes F1 score for slot value extraction.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "slot_f1"

    def compute_record_level_scores(self, candidates: list, references: list,
                                    task_name: str = None, model_name: str = None) -> Dict[str, List]:
        """Compute slot F1 for each sample."""
        scores = []
        
        desc = f"Slot F1 Eval"
        if task_name and model_name:
            desc = f"Slot F1 [{task_name}] [{model_name}]"
        
        for i, (candidate, reference) in enumerate(tqdm(
            zip(candidates, references), desc=desc, total=len(candidates)
        )):
            gt_slots = {}
            if i < len(self.ground_truth_slots):
                gt_slots = self._parse_ground_truth_slots(self.ground_truth_slots[i])
            
            pred_slots = self._extract_slots_from_response(candidate)
            slot_f1 = self._compute_slot_f1(pred_slots, gt_slots)
            scores.append(slot_f1)
        
        return {self.name: scores}

    def _compute_slot_f1(self, pred_slots: Dict, gt_slots: Dict) -> float:
        """Compute slot F1 score."""
        gt_set = set()
        pred_set = set()
        
        for domain, slots in gt_slots.items():
            for slot, value in slots.items():
                gt_set.add((domain, slot, self._normalize_value(value)))
        
        for domain, slots in pred_slots.items():
            for slot, value in slots.items():
                pred_set.add((domain, slot, self._normalize_value(value)))
        
        if not gt_set and not pred_set:
            return 1.0
        if not gt_set or not pred_set:
            return 0.0
        
        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)

"""SpokenWoz postprocessor module for task-oriented dialogue evaluation.

This postprocessor extracts ground truth slots from the dataset and passes
them to the SpokenWoz metrics for JGA and slot accuracy evaluation.
"""

import logging

from postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)
logger.propagate = True


class SpokenWozPostprocessor(Postprocessor):
    """Postprocessor for SpokenWoz task-oriented dialogue predictions.
    
    Extracts ground truth slots and passes them along with predictions
    for dialogue state tracking evaluation.
    """

    def process(self, dataset: list[dict], predictions, metric) -> dict:
        """
        Process model predictions and prepare data for SpokenWoz evaluation.
        
        Args:
            dataset (list[dict]): List of preprocessed input samples
            predictions (dict): Dictionary mapping model names to lists of predictions
            metric: Evaluation metric
            
        Returns:
            dict: Dictionary containing processed data for evaluation including ground truth slots
        """

        # Process predictions
        processed_predictions = self.process_predictions(predictions)
        
        # Extract targets (agent responses)
        targets = self.extract_targets(dataset)

        # Extract instructions
        instructions = self.extract_instructions(dataset)

        # Extract ground truth slots for SpokenWoz metrics
        ground_truth_slots = [record.get('ground_truth_slots', {}) for record in dataset]

        # Create output with slots data
        output = {
            "model_targets": targets,
            "processed_predictions": processed_predictions,
            "instructions": instructions,
            "ground_truth_slots": ground_truth_slots
        }

        self.validate_output(output)
        return output

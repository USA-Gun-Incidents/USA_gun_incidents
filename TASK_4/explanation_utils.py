import numpy as np
from aix360.metrics import faithfulness_metric, monotonicity_metric

def evaluate_explanation(model, instance, feature_importances, feature_defaults):
    metrics = {}
    metrics['faithfulness'] = faithfulness_metric(model, instance, feature_importances, feature_defaults)
    metrics['monotonity'] = monotonicity_metric(model, instance, feature_importances, feature_defaults)
    return metrics
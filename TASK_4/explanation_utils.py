import numpy as np
from aix360.metrics import faithfulness_metric, monotonicity_metric # TODO: provarne altre?

def evaluate_explanation(model, instance, feature_importances): 
    metrics = {}
    base = np.zeros(instance.shape[0]) # TODO: valutare
    metrics['faithfulness'] = faithfulness_metric(model, instance, feature_importances, base)
    metrics['monotonity'] = monotonicity_metric(model, instance, feature_importances, base)
    return metrics
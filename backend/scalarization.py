import numpy as np

def weighted_sum_scalarization(objectives, weights):
    """
    Compute the weighted sum scalarization score.

    Parameters:
        objectives (dict): Dictionary of objective values, e.g.,
                           {'return': 0.12, 'risk': 0.08, 'liquidity': 0.6}
        weights (dict): Dictionary of weights for each objective, e.g.,
                        {'return': 1.0, 'risk': -0.5, 'liquidity': 0.2}

    Returns:
        float: Weighted sum score (higher is better).
    """
    if set(objectives.keys()) != set(weights.keys()):
        raise ValueError("Objective and weight keys must match.")
    return sum(objectives[k] * weights[k] for k in objectives)
    

def tchebycheff_scalarization(objectives, weights, ideal_point):
    """
    Implements the Tchebycheff scalarization method.

    Parameters:
        objectives (dict): A dictionary of current objective values.
                           Example: {'return': 0.12, 'risk': 0.08, 'liquidity': 0.6}
        weights (dict): A dictionary of positive weights for each objective.
                        Example: {'return': 1.0, 'risk': 0.5, 'liquidity': 0.3}
        ideal_point (dict): A dictionary of ideal/target values for each objective.
                            Example: {'return': 0.15, 'risk': 0.03, 'liquidity': 0.8}

    Returns:
        float: The Tchebycheff scalarization score (lower is better).
    """
    if set(objectives.keys()) != set(weights.keys()) or set(objectives.keys()) != set(ideal_point.keys()):
        raise ValueError("Objective, weight, and ideal point keys must match.")

    max_term = float('-inf')
    for key in objectives:
        term = weights[key] * abs(objectives[key] - ideal_point[key])
        if term > max_term:
            max_term = term

    return max_term


if __name__ == "__main__":
    # Test for weighted sum
    obj = {'return': 0.10, 'risk': 0.05, 'liquidity': 0.6}
    wts = {'return': 0.4, 'risk': 0.4, 'liquidity': 0.2}
    print("Weighted Sum Score:", weighted_sum_scalarization(obj, wts))

    # Test for Tchebycheff
    ideal = {'return': 0.15, 'risk': 0.03, 'liquidity': 0.8}
    print("Tchebycheff Score:", tchebycheff_scalarization(obj, wts, ideal))
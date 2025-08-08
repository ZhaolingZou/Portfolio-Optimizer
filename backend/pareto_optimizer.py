# backend/pareto_optimizer.py
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.variable import Real
from config import Config

# Internal preference learning system (hidden from external interface)
class _PreferenceLearner:
    """
    Internal preference learning system - not exposed to external modules
    """
    
    def __init__(self):
        self.preference_weights = {'return': 0.33, 'risk': 0.33, 'liquidity': 0.34}
        self.learning_rate = 0.5  # Increased base learning rate
        self.user_selections = []
        self.selection_count = 0
        self.min_weight = 0.05  # Minimum weight threshold (5%)
        
    def update_preferences(self, selected_solution, all_solutions):
        """Updated preference learning with better weight calculation"""
        if len(all_solutions) < 2:
            print("Debug: Not enough solutions to learn preferences")
            return
            
        self.user_selections.append({
            'selected': selected_solution,
            'alternatives': all_solutions
        })
        self.selection_count += 1
        
        print(f"\n=== Preference Learning Debug - Selection {self.selection_count} ===")
        
        # Extract and validate objective values
        objectives_data = []
        selected_idx = -1
        
        for i, sol in enumerate(all_solutions):
            obj_values = [
                sol['objectives']['return'], 
                sol['objectives']['risk'], 
                sol['objectives']['liquidity']
            ]
            objectives_data.append(obj_values)
            if sol['id'] == selected_solution['id']:
                selected_idx = i
            
            print(f"Debug: Solution {i} ({'SELECTED' if i == selected_idx else 'alternative'}): "
                  f"return={obj_values[0]:.4f}, risk={obj_values[1]:.4f}, liquidity={obj_values[2]:.4f}")
        
        if selected_idx == -1:
            print("Debug: Selected solution not found in alternatives!")
            return
            
        objectives_matrix = np.array(objectives_data)
        
        # Calculate preference weights using improved method
        new_weights = self._calculate_preference_weights_v2(
            selected_idx, objectives_matrix
        )
        
        # Use adaptive learning rate - more aggressive for early selections
        base_lr = 0.7  # Increased base learning rate
        adaptive_lr = base_lr * (1.0 - 0.03 * min(self.selection_count, 10))
        adaptive_lr = max(adaptive_lr, 0.3)  # Higher minimum learning rate
        
        print(f"Debug: Learning rate: {adaptive_lr}")
        print(f"Debug: Old weights: {self.preference_weights}")
        print(f"Debug: New weights from calculation: {dict(zip(['return', 'risk', 'liquidity'], new_weights))}")
        
        # Store old weights for comparison
        old_weights = self.preference_weights.copy()
        
        # Update weights with adaptive learning
        objectives = ['return', 'risk', 'liquidity']
        for i, obj in enumerate(objectives):
            self.preference_weights[obj] = (1-adaptive_lr) * self.preference_weights[obj] + adaptive_lr * new_weights[i]
        
        # Apply minimum weight constraint and normalize
        self._apply_constraints_and_normalize()
        
        print(f"Debug: Updated weights: {self.preference_weights}")
        print(f"Debug: Weight changes: return={self.preference_weights['return'] - old_weights['return']:.4f}, "
              f"risk={self.preference_weights['risk'] - old_weights['risk']:.4f}, "
              f"liquidity={self.preference_weights['liquidity'] - old_weights['liquidity']:.4f}")
        print("=" * 60)
    
    def _calculate_preference_weights_v2(self, selected_idx, objectives_matrix):
        """Improved preference weight calculation method"""
        n_solutions, n_objectives = objectives_matrix.shape
        preference_scores = np.zeros(n_objectives)
        
        selected_obj = objectives_matrix[selected_idx]
        print(f"Debug: Selected solution raw values: {selected_obj}")
        
        # For each objective, calculate how well the selected solution performs
        for obj_idx in range(n_objectives):
            selected_value = selected_obj[obj_idx]
            all_values = objectives_matrix[:, obj_idx]
            
            if obj_idx == 1:  # Risk objective - lower is better
                # Count how many solutions have higher risk (worse performance)
                better_count = np.sum(all_values > selected_value)
                # If selected solution has low risk, it should get high preference
                rank_score = better_count / (n_solutions - 1) if n_solutions > 1 else 0.5
            else:  # Return and Liquidity - higher is better
                # Count how many solutions have lower return/liquidity (worse performance)
                better_count = np.sum(all_values < selected_value)
                # If selected solution has high return/liquidity, it should get high preference
                rank_score = better_count / (n_solutions - 1) if n_solutions > 1 else 0.5
            
            preference_scores[obj_idx] = rank_score
            
            print(f"Debug: Objective {obj_idx} ({'risk' if obj_idx == 1 else ['return', 'risk', 'liquidity'][obj_idx]}), "
                  f"selected_value: {selected_value:.4f}, better_count: {better_count}, "
                  f"rank_score: {rank_score:.4f}")
        
        # Enhance differences - give more weight to objectives where selected solution excels
        enhanced_scores = np.power(preference_scores, 1.5)  # Moderate enhancement
        
        # Add base weight to prevent zero weights
        base_weight = 0.15  # Increased base weight
        enhanced_scores = enhanced_scores + base_weight
        
        # Give extra boost to the best performing objective
        max_idx = np.argmax(enhanced_scores)
        enhanced_scores[max_idx] *= 1.3
        
        # Normalize to sum to 1
        total_score = np.sum(enhanced_scores)
        if total_score > 0:
            normalized_scores = enhanced_scores / total_score
        else:
            normalized_scores = np.ones(n_objectives) / n_objectives
        
        print(f"Debug: Raw preference scores: {preference_scores}")
        print(f"Debug: Enhanced scores: {enhanced_scores}")
        print(f"Debug: Final normalized scores: {normalized_scores}")
        
        return normalized_scores
    
    def _apply_constraints_and_normalize(self):
        """Apply minimum weight constraints and normalize"""
        # Ensure minimum weights
        for obj in self.preference_weights:
            if self.preference_weights[obj] < self.min_weight:
                print(f"Debug: Adjusting {obj} weight from {self.preference_weights[obj]:.4f} to minimum {self.min_weight}")
                self.preference_weights[obj] = self.min_weight
        
        # Normalize to ensure sum equals 1
        total = sum(self.preference_weights.values())
        if total > 0:
            self.preference_weights = {k: v/total for k, v in self.preference_weights.items()}
        
        print(f"Debug: After constraint application: {self.preference_weights}")
    
    def calculate_utility_score(self, solution):
        """Calculate utility score using learned preferences with improved utility functions"""
        ret = solution['objectives']['return']
        risk = solution['objectives']['risk'] 
        liquidity = solution['objectives']['liquidity']
        
        # Improved utility functions with better scaling
        return_utility = np.log(1 + max(ret * 100, 0)) if ret > 0 else -2 * np.log(1 + abs(ret * 100))
        risk_utility = -np.log(1 + risk * 100)  # Lower risk = higher utility
        liquidity_utility = np.sqrt(max(liquidity, 0))
        
        # Weighted combination
        total_utility = (
            self.preference_weights['return'] * return_utility +
            self.preference_weights['risk'] * risk_utility +
            self.preference_weights['liquidity'] * liquidity_utility
        )
        
        return total_utility
    
    def has_learned_preferences(self):
        """Check if we have learned any preferences"""
        return len(self.user_selections) > 0
    
    def reset_preferences(self):
        """Reset preferences to default values"""
        self.preference_weights = {'return': 0.33, 'risk': 0.33, 'liquidity': 0.34}
        self.user_selections = []
        self.selection_count = 0
        print("Debug: Preferences reset to default values")

# Global internal preference learner
_preference_learner = _PreferenceLearner()

class PortfolioProblem(Problem):
    """
    Multi-objective portfolio optimization problem for NSGA-II
    Objectives: Maximize return, Minimize risk, Maximize liquidity
    """
    
    def __init__(self, expected_returns, cov_matrix, liquidity_scores):
        """
        Initialize the portfolio optimization problem.
        
        Parameters:
            expected_returns (pd.Series): Expected returns for each asset
            cov_matrix (pd.DataFrame): Covariance matrix of returns
            liquidity_scores (pd.Series): Liquidity scores for each asset
        """
        self.expected_returns = expected_returns.values
        self.cov_matrix = cov_matrix.values
        self.liquidity_scores = liquidity_scores.values
        self.n_assets = len(expected_returns)
        
        # Get configuration parameters with defaults
        min_weight = getattr(Config, 'MIN_ASSET_WEIGHT', 0.0)
        max_weight = getattr(Config, 'MAX_SINGLE_ASSET_WEIGHT', 1.0)
        
        # Define the problem: n_var variables, 3 objectives, 1 constraint
        super().__init__(
            n_var=self.n_assets,
            n_obj=3,  # return (maximize), risk (minimize), liquidity (maximize)
            n_ieq_constr=1,  # Use n_ieq_constr instead of n_constr
            xl=np.full(self.n_assets, min_weight),  # lower bounds
            xu=np.full(self.n_assets, max_weight)   # upper bounds
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the objectives and constraints for given portfolio weights.
        
        Parameters:
            X (np.array): Portfolio weights matrix (n_solutions x n_assets)
            out (dict): Output dictionary to store objectives and constraints
        """
        n_solutions = X.shape[0]
        
        # Initialize output arrays
        f1 = np.zeros(n_solutions)  # Return (to maximize, so we negate)
        f2 = np.zeros(n_solutions)  # Risk (to minimize)
        f3 = np.zeros(n_solutions)  # Liquidity (to maximize, so we negate)
        g1 = np.zeros(n_solutions)  # Constraint: sum of weights = 1
        
        for i in range(n_solutions):
            weights = X[i, :]
            
            # Normalize weights to sum to 1
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(self.n_assets) / self.n_assets
            
            # Calculate portfolio return (negative because we want to maximize)
            portfolio_return = np.dot(weights, self.expected_returns)
            f1[i] = -portfolio_return
            
            # Calculate portfolio risk (variance) - add numerical stability
            portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
            portfolio_risk = np.sqrt(max(portfolio_variance, 1e-8))
            f2[i] = portfolio_risk
            
            # Calculate portfolio liquidity (negative because we want to maximize)
            portfolio_liquidity = np.dot(weights, self.liquidity_scores)
            f3[i] = -portfolio_liquidity
            
            # Constraint: sum of weights should be 1 (use inequality constraint format)
            g1[i] = abs(np.sum(weights) - 1.0) - 0.01  # Allow small tolerance
        
        # Store results
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = g1.reshape(-1, 1)

def generate_pareto_solutions(expected_returns, cov_matrix, liquidity_scores):
    """
    Generate Pareto optimal solutions using NSGA-II algorithm.
    
    Parameters:
        expected_returns (pd.Series): Expected returns for each asset
        cov_matrix (pd.DataFrame): Covariance matrix of returns
        liquidity_scores (pd.Series): Liquidity scores for each asset
        
    Returns:
        list: List of dictionaries containing Pareto optimal solutions
    """
    
    # Create the problem
    problem = PortfolioProblem(expected_returns, cov_matrix, liquidity_scores)
    
    # Get configuration parameters with defaults
    pop_size = getattr(Config, 'PARETO_POPULATION_SIZE', 100)
    generations = getattr(Config, 'PARETO_GENERATIONS', 200)
    
    # Configure the algorithm - modified parameters for increased diversity
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=10),  # Lower eta for more diversity
        mutation=PM(prob=0.2, eta=15),    # Increased mutation probability
        eliminate_duplicates=True
    )
    
    # Run optimization - remove fixed seed for more randomness
    res = minimize(
        problem,
        algorithm,
        ('n_gen', generations),
        verbose=False
    )
    
    # Extract Pareto optimal solutions
    pareto_solutions = []
    
    if res.X is not None and len(res.X) > 0:
        for i, (weights, objectives) in enumerate(zip(res.X, res.F)):
            # Normalize weights
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                normalized_weights = weights / weight_sum
            else:
                normalized_weights = np.ones(len(weights)) / len(weights)
            
            # Create weight dictionary - include all weights, don't filter small ones
            weight_dict = {}
            for j, ticker in enumerate(expected_returns.index):
                weight_dict[ticker] = float(normalized_weights[j])
            
            # Calculate actual metrics (convert back from optimization format)
            portfolio_return = -objectives[0]  # Was negated for maximization
            portfolio_risk = objectives[1]
            portfolio_liquidity = -objectives[2]  # Was negated for maximization
            
            # Calculate Sharpe ratio - add safety check
            risk_free_rate = getattr(Config, 'RISK_FREE_RATE', 0.02)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            solution = {
                'id': i,
                'weights': weight_dict,
                'metrics': {
                    'return': float(portfolio_return),
                    'risk': float(portfolio_risk),
                    'liquidity': float(portfolio_liquidity),
                    'sharpe_ratio': float(sharpe_ratio)
                },
                'objectives': {
                    'return': float(portfolio_return),
                    'risk': float(portfolio_risk),
                    'liquidity': float(portfolio_liquidity)
                }
            }
            
            pareto_solutions.append(solution)
    
    # Use better diversity selection
    solutions_to_show = getattr(Config, 'PARETO_SOLUTIONS_TO_SHOW', 5)
    diverse_solutions = select_diverse_solutions(pareto_solutions, solutions_to_show)
    
    return diverse_solutions

def select_diverse_solutions(pareto_solutions, n_solutions=5):
    """
    Select diverse solutions from Pareto front to present to user.
    Enhanced version with better diversity selection and internal preference learning.
    
    Parameters:
        pareto_solutions (list): List of Pareto optimal solutions
        n_solutions (int): Number of solutions to select
        
    Returns:
        list: Selected diverse solutions
    """
    if len(pareto_solutions) <= n_solutions:
        # Add labels and calculate utility scores for internal ranking
        for i, sol in enumerate(pareto_solutions):
            sol['label'] = f'Portfolio {i+1}'
            sol['_utility_score'] = _preference_learner.calculate_utility_score(sol)  # Internal use only
        return pareto_solutions
    
    selected = []
    remaining = pareto_solutions.copy()
    
    # Calculate internal utility scores for all solutions
    for sol in pareto_solutions:
        sol['_utility_score'] = _preference_learner.calculate_utility_score(sol)  # Internal use only
    
    # 1. If we have learned preferences, prioritize AI recommended solution
    if _preference_learner.has_learned_preferences():
        best_utility_sol = max(remaining, key=lambda x: x['_utility_score'])
        best_utility_sol['label'] = 'AI Recommended'
        selected.append(best_utility_sol)
        remaining.remove(best_utility_sol)
    
    # 2. Select extreme solutions
    # Highest return
    if remaining:
        max_return_sol = max(remaining, key=lambda x: x['objectives']['return'])
        max_return_sol['label'] = 'High Return Strategy'
        selected.append(max_return_sol)
        remaining.remove(max_return_sol)
    
    # Lowest risk
    if remaining:
        min_risk_sol = min(remaining, key=lambda x: x['objectives']['risk'])
        min_risk_sol['label'] = 'Conservative Strategy'
        selected.append(min_risk_sol)
        remaining.remove(min_risk_sol)
    
    # Highest liquidity
    if remaining:
        max_liquidity_sol = max(remaining, key=lambda x: x['objectives']['liquidity'])
        max_liquidity_sol['label'] = 'High Liquidity Strategy'
        selected.append(max_liquidity_sol)
        remaining.remove(max_liquidity_sol)
    
    # Best Sharpe ratio
    if remaining and len(selected) < n_solutions:
        best_sharpe_sol = max(remaining, key=lambda x: x['metrics']['sharpe_ratio'])
        best_sharpe_sol['label'] = 'Balanced Strategy'
        selected.append(best_sharpe_sol)
        remaining.remove(best_sharpe_sol)
    
    # 3. Fill remaining positions - use improved distance calculation
    while len(selected) < n_solutions and remaining:
        # Normalize objective values for distance calculation
        all_returns = [sol['objectives']['return'] for sol in pareto_solutions]
        all_risks = [sol['objectives']['risk'] for sol in pareto_solutions]
        all_liquidities = [sol['objectives']['liquidity'] for sol in pareto_solutions]
        
        return_range = max(all_returns) - min(all_returns) + 1e-6
        risk_range = max(all_risks) - min(all_risks) + 1e-6
        liquidity_range = max(all_liquidities) - min(all_liquidities) + 1e-6
        
        max_min_distance = -1
        best_candidate = None
        
        for candidate in remaining:
            # Calculate minimum distance to selected solutions
            min_distance = float('inf')
            for selected_sol in selected:
                # Normalize objective values
                norm_candidate_return = (candidate['objectives']['return'] - min(all_returns)) / return_range
                norm_candidate_risk = (candidate['objectives']['risk'] - min(all_risks)) / risk_range
                norm_candidate_liquidity = (candidate['objectives']['liquidity'] - min(all_liquidities)) / liquidity_range
                
                norm_selected_return = (selected_sol['objectives']['return'] - min(all_returns)) / return_range
                norm_selected_risk = (selected_sol['objectives']['risk'] - min(all_risks)) / risk_range
                norm_selected_liquidity = (selected_sol['objectives']['liquidity'] - min(all_liquidities)) / liquidity_range
                
                # Calculate Euclidean distance
                dist = np.sqrt(
                    (norm_candidate_return - norm_selected_return)**2 +
                    (norm_candidate_risk - norm_selected_risk)**2 +
                    (norm_candidate_liquidity - norm_selected_liquidity)**2
                )
                min_distance = min(min_distance, dist)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        
        if best_candidate:
            best_candidate['label'] = f'Alternative Strategy {len(selected)}'
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    # Sort by utility score if preferences learned, otherwise by return
    if _preference_learner.has_learned_preferences():
        selected.sort(key=lambda x: x['_utility_score'], reverse=True)
    else:
        selected.sort(key=lambda x: x['objectives']['return'], reverse=True)
    
    # Clean up internal utility scores before returning
    for sol in selected:
        if '_utility_score' in sol:
            del sol['_utility_score']
    
    return selected

# New functions for external modules to interact with preference learning (optional usage)
def record_user_selection(selected_solution, all_solutions):
    """
    Record user selection for preference learning with validation
    
    Parameters:
        selected_solution: User's selected portfolio
        all_solutions: All available portfolio options
    """
    print(f"\n=== User Selection Validation ===")
    print(f"Selected solution ID: {selected_solution.get('id', 'Unknown')}")
    print(f"Selected objectives: {selected_solution.get('objectives', {})}")
    
    # Find the solution with highest return for comparison
    max_return_sol = max(all_solutions, key=lambda x: x['objectives']['return'])
    print(f"Highest return solution ID: {max_return_sol['id']}")
    print(f"Highest return objectives: {max_return_sol['objectives']}")
    
    if selected_solution['id'] == max_return_sol['id']:
        print("✓ User selected the highest return solution")
    else:
        print("✗ User did NOT select the highest return solution")
    
    # Find the solution with lowest risk for comparison
    min_risk_sol = min(all_solutions, key=lambda x: x['objectives']['risk'])
    print(f"Lowest risk solution ID: {min_risk_sol['id']}")
    print(f"Lowest risk objectives: {min_risk_sol['objectives']}")
    
    if selected_solution['id'] == min_risk_sol['id']:
        print("✓ User selected the lowest risk solution")
    else:
        print("✗ User did NOT select the lowest risk solution")
    
    # Find the solution with highest liquidity for comparison
    max_liquidity_sol = max(all_solutions, key=lambda x: x['objectives']['liquidity'])
    print(f"Highest liquidity solution ID: {max_liquidity_sol['id']}")
    print(f"Highest liquidity objectives: {max_liquidity_sol['objectives']}")
    
    if selected_solution['id'] == max_liquidity_sol['id']:
        print("✓ User selected the highest liquidity solution")
    else:
        print("✗ User did NOT select the highest liquidity solution")
    
    print("=" * 50)
    
    _preference_learner.update_preferences(selected_solution, all_solutions)

def get_learned_preferences():
    """
    Get current learned preference weights (optional function for external use)
    
    Returns:
        dict: Current preference weights for return, risk, and liquidity
    """
    return _preference_learner.preference_weights.copy()

def reset_preferences():
    """
    Reset learned preferences to default values
    """
    _preference_learner.reset_preferences()

def get_preference_learning_stats():
    """
    Get statistics about preference learning
    
    Returns:
        dict: Statistics about the learning process
    """
    return {
        'selection_count': _preference_learner.selection_count,
        'has_learned_preferences': _preference_learner.has_learned_preferences(),
        'current_weights': _preference_learner.preference_weights.copy(),
        'min_weight_threshold': _preference_learner.min_weight
    }

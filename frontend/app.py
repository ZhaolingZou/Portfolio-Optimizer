# frontend/app.py
from flask import Flask, render_template, request, jsonify, session
import json
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from portfolio_engine import PortfolioEngine
from config import Config

def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)
    app.secret_key = Config.SECRET_KEY
    return app

app = create_app()

# Global portfolio engine instance
portfolio_engine = None

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system initialization status."""
    global portfolio_engine
    
    return jsonify({
        'initialized': portfolio_engine is not None and portfolio_engine.data_loaded,
        'current_preferences': session.get('preferences', Config.DEFAULT_WEIGHTS),
        'has_pareto_solutions': portfolio_engine.current_pareto_solutions is not None if portfolio_engine else False,
        'optimization_stage': session.get('optimization_stage', 'initial')  # initial, pareto_selection, preference_refinement
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the portfolio optimization system."""
    global portfolio_engine
    
    try:
        data = request.get_json()
        tickers = data.get('tickers', Config.DEFAULT_TICKERS)
        start_date = data.get('start_date', Config.DEFAULT_START_DATE)
        end_date = data.get('end_date', Config.DEFAULT_END_DATE)
        
        # Create portfolio engine
        portfolio_engine = PortfolioEngine(tickers=tickers)
        
        # Load data
        success = portfolio_engine.load_data(start_date=start_date, end_date=end_date)
        
        if success:
            # Initialize session
            session['preferences'] = Config.DEFAULT_WEIGHTS.copy()
            session['optimization_stage'] = 'initial'
            session['iteration_count'] = 0
            
            data_info = portfolio_engine.get_data_info()
            
            return jsonify({
                'success': True,
                'message': f'System initialized with {data_info["n_tickers"]} stocks',
                'data_info': data_info
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load market data'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Initialization failed: {str(e)}'
        }), 500

@app.route('/api/generate_pareto', methods=['POST'])
def generate_pareto_frontier():
    """Generate Pareto optimal solutions for user selection."""
    global portfolio_engine
    
    if not portfolio_engine or not portfolio_engine.data_loaded:
        return jsonify({
            'success': False,
            'message': 'System not initialized'
        }), 400
    
    try:
        data = request.get_json()
        target_date = data.get('target_date', None)
        
        # Generate Pareto frontier
        result = portfolio_engine.generate_pareto_frontier(target_date=target_date)
        
        if result['success']:
            session['optimization_stage'] = 'pareto_selection'
            session['pareto_solutions'] = result['pareto_solutions']
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to generate Pareto solutions: {str(e)}'
        }), 500

@app.route('/api/select_pareto_solution', methods=['POST'])
def select_pareto_solution():
    """Handle user selection of a Pareto optimal solution."""
    global portfolio_engine
    
    if not portfolio_engine or not portfolio_engine.current_pareto_solutions:
        return jsonify({
            'success': False,
            'message': 'No Pareto solutions available'
        }), 400
    
    try:
        data = request.get_json()
        selected_id = data.get('solution_id')
        
        if selected_id is None:
            return jsonify({
                'success': False,
                'message': 'No solution ID provided'
            }), 400
        
        # Extract preferences from selection
        preferences = portfolio_engine.extract_preferences_from_selection(selected_id)
        
        # Update session
        session['preferences'] = preferences
        session['optimization_stage'] = 'preference_refinement'
        session['selected_pareto_id'] = selected_id
        
        # Get the selected solution details
        selected_solution = None
        for solution in portfolio_engine.current_pareto_solutions:
            if solution['id'] == selected_id:
                selected_solution = solution
                break
        
        return jsonify({
            'success': True,
            'message': 'Selection recorded. Preferences extracted from your choice.',
            'extracted_preferences': preferences,
            'selected_solution': selected_solution
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to process selection: {str(e)}'
        }), 500

@app.route('/api/optimize_with_preferences', methods=['POST'])
def optimize_with_preferences():
    """Optimize portfolio using extracted or refined preferences."""
    global portfolio_engine
    
    if not portfolio_engine or not portfolio_engine.data_loaded:
        return jsonify({
            'success': False,
            'message': 'System not initialized'
        }), 400
    
    try:
        data = request.get_json()
        preferences = data.get('preferences', session.get('preferences', Config.DEFAULT_WEIGHTS))
        method = data.get('method', 'tchebycheff')
        target_date = data.get('target_date', None)
        
        # Validate preferences
        if not isinstance(preferences, dict) or not all(k in preferences for k in ['return', 'risk', 'liquidity']):
            return jsonify({
                'success': False,
                'message': 'Invalid preferences format'
            }), 400
        
        # Normalize preferences to sum to 1
        total_weight = sum(preferences.values())
        if total_weight > 0:
            preferences = {k: v/total_weight for k, v in preferences.items()}
        
        # Optimize portfolio
        result = portfolio_engine.optimize_portfolio(
            preferences=preferences,
            target_date=target_date,
            method=method
        )
        
        # Update session
        session['preferences'] = preferences
        session['last_optimization'] = result
        session['iteration_count'] = session.get('iteration_count', 0) + 1
        
        return jsonify({
            'success': True,
            'message': 'Portfolio optimized successfully',
            'result': result,
            'iteration': session['iteration_count']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Optimization failed: {str(e)}'
        }), 500

@app.route('/api/refine_preferences', methods=['POST'])
def refine_preferences():
    """Refine user preferences based on feedback."""
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        current_preferences = session.get('preferences', Config.DEFAULT_WEIGHTS)
        
        if not portfolio_engine:
            return jsonify({
                'success': False,
                'message': 'System not initialized'
            }), 400
        
        # Use GPT to extract refined preferences
        extraction_result = portfolio_engine.preference_extractor.extract_preferences(
            user_input=user_input,
            current_preferences=current_preferences
        )
        
        if extraction_result['success']:
            refined_preferences = extraction_result['preferences']
            
            # Update session
            session['preferences'] = refined_preferences
            
            return jsonify({
                'success': True,
                'message': 'Preferences refined based on your feedback',
                'refined_preferences': refined_preferences,
                'explanation': extraction_result.get('explanation', '')
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to refine preferences',
                'error': extraction_result.get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to refine preferences: {str(e)}'
        }), 500

@app.route('/api/reset_optimization', methods=['POST'])
def reset_optimization():
    """Reset the optimization process to start over."""
    try:
        # Reset session state
        session['optimization_stage'] = 'initial'
        session['preferences'] = Config.DEFAULT_WEIGHTS.copy()
        session['iteration_count'] = 0
        
        # Clear Pareto solutions from engine
        if portfolio_engine:
            portfolio_engine.current_pareto_solutions = None
        
        # Clear session data
        session.pop('pareto_solutions', None)
        session.pop('selected_pareto_id', None)
        session.pop('last_optimization', None)
        
        return jsonify({
            'success': True,
            'message': 'Optimization process reset successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to reset: {str(e)}'
        }), 500

@app.route('/api/get_data_info')
def get_data_info():
    """Get information about loaded data."""
    global portfolio_engine
    
    if not portfolio_engine or not portfolio_engine.data_loaded:
        return jsonify({
            'success': False,
            'message': 'No data loaded'
        }), 400
    
    try:
        data_info = portfolio_engine.get_data_info()
        return jsonify({
            'success': True,
            'data_info': data_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to get data info: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

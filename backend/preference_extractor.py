import openai
import json
import re
from config import Config

class PreferenceExtractor:
    def __init__(self):
        """Initialize the preference extractor with OpenAI API key from config."""
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
        self.system_prompt = """
        You are an assistant that extracts investment preferences from user input.
        Your response must be ONLY a valid JSON object with the following keys:

        - "return": a number between 0 and 1 (higher means more focus on returns)
        - "risk": a number between 0 and 1 (higher means more risk tolerance)
        - "liquidity": a number between 0 and 1 (higher means more focus on liquidity)

        The values should sum to approximately 1.0 and reflect the user's preferences.
        
        Example responses:
        {"return": 0.6, "risk": 0.2, "liquidity": 0.2}
        {"return": 0.3, "risk": 0.4, "liquidity": 0.3}
        """

    def extract_preferences(self, user_input, current_preferences=None):
        """
        Extract structured investment preferences from natural language input using GPT.

        Args:
            user_input (str): User's textual input expressing preferences.
            current_preferences (dict): Current preference weights (optional).

        Returns:
            dict: A dictionary containing extracted preferences and metadata.
        """
        if current_preferences is None:
            current_preferences = Config.DEFAULT_WEIGHTS

        # Create context-aware prompt
        context = f"Current preferences: {current_preferences}\nUser input: {user_input}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context},
        ]

        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS
            )

            reply = response.choices[0].message.content.strip()

            # Try to extract JSON content from the reply
            json_match = re.search(r"\{.*\}", reply, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in GPT response.")

            preferences = json.loads(json_match.group())

            # Validate and normalize preferences
            preferences = self._validate_and_normalize_preferences(preferences)

            # Generate interpretation text
            interpretation = self._generate_interpretation(user_input, preferences, current_preferences)

            return {
                "preferences": preferences,
                "suggested_weights": preferences,  # Frontend expects this key
                "interpretation": interpretation,  # Frontend expects this key
                "raw_response": reply,
                "success": True,
                "user_input": user_input,
                "confidence": 0.8  # Add confidence score
            }

        except Exception as e:
            print(f"Error extracting preferences: {e}")
            # Return default preferences on error
            return {
                "preferences": current_preferences,
                "suggested_weights": current_preferences,
                "interpretation": f"I couldn't fully understand your input '{user_input}', so I'm keeping your current preferences.",
                "raw_response": "",
                "success": False,
                "error": str(e),
                "user_input": user_input,
                "confidence": 0.0
            }

    def _validate_and_normalize_preferences(self, preferences):
        """Validate and normalize preference values."""
        required_keys = ["return", "risk", "liquidity"]
        
        # Check for required keys
        for key in required_keys:
            if key not in preferences:
                raise KeyError(f"Missing key: {key}")
            
            value = preferences[key]
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                raise ValueError(f"Invalid value for '{key}': {value}")

        # Normalize to sum to 1.0
        total = sum(preferences.values())
        if total > 0:
            preferences = {k: v/total for k, v in preferences.items()}
        else:
            # Fallback to equal weights
            preferences = {k: 1/len(required_keys) for k in required_keys}

        return preferences

    def _generate_interpretation(self, user_input, new_preferences, old_preferences):
        """Generate human-readable interpretation of preference changes."""
        try:
            interpretation = f"Based on your input '{user_input}', I've updated your preferences: "
            
            changes = []
            for key in ["return", "risk", "liquidity"]:
                old_val = old_preferences.get(key, 0)
                new_val = new_preferences.get(key, 0)
                diff = new_val - old_val
                
                if abs(diff) > 0.05:  # Significant change threshold
                    direction = "increased" if diff > 0 else "decreased"
                    changes.append(f"{key} focus {direction} to {new_val*100:.1f}%")
            
            if changes:
                interpretation += ", ".join(changes) + "."
            else:
                interpretation += "your preferences remain similar to before."
                
            return interpretation
            
        except Exception as e:
            return f"Updated preferences based on your input: {user_input}"

    def generate_explanation(self, weights, metrics):
        """
        Generate human-readable explanation of optimization results.

        Parameters:
            weights (dict): Portfolio weights
            metrics (dict): Portfolio metrics

        Returns:
            str: Human-readable explanation
        """
        try:
            # Find top holdings
            top_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            
            explanation = f"This portfolio allocates your investments across {len(weights)} assets. "
            explanation += f"Your largest positions are: "
            
            for i, (ticker, weight) in enumerate(top_holdings):
                if i > 0:
                    explanation += ", "
                explanation += f"{ticker} ({weight*100:.1f}%)"
            
            explanation += f". The portfolio has an expected annual return of {metrics.get('return', 0)*100:.2f}% "
            explanation += f"with a risk level of {metrics.get('risk', 0)*100:.2f}%. "
            
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 0:
                explanation += f"The Sharpe ratio is {sharpe:.3f}, indicating good risk-adjusted performance."
            
            return explanation
            
        except Exception as e:
            return f"Portfolio optimized with {len(weights)} assets based on your preferences."


# Legacy function for backward compatibility
def extract_preferences(user_input):
    """
    Legacy function for backward compatibility.
    """
    extractor = PreferenceExtractor()
    result = extractor.extract_preferences(user_input)
    return result["preferences"] if result["success"] else None

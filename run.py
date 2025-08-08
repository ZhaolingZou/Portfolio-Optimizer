# run.py
"""
Portfolio Optimizer Application Launcher
Main entry point for the Flask web application
"""

import os
import sys
from flask import Flask

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import Config

def main():
    """Main function to run the Flask application"""
    try:
        # Import the app from frontend
        from frontend.app import app
        
        # Print startup information
        print("=" * 50)
        print("Portfolio Optimizer Starting...")
        print("=" * 50)
        print(f"Application: Smart Portfolio Optimizer")
        print(f"Host: {Config.HOST}")
        print(f"Port: {Config.PORT}")
        print(f"Debug Mode: {Config.DEBUG}")
        print(f"Data Source: {Config.DATA_SOURCE}")
        print(f"AI Model: {Config.OPENAI_MODEL}")
        print("=" * 50)
        print(f"Access the application at: http://{Config.HOST}:{Config.PORT}")
        print("=" * 50)
        
        
        # Run the Flask application
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Application stopped by user")
        print("=" * 50)
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"Error starting application: {str(e)}")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()

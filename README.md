# Portfolio Optimizer (Multi-Objective)

## Demo
![Demo](docs/demo.gif)
**Live Demo:** [Portfolio Optimizer on Render](https://portfolio-optimizer-7n1i.onrender.com)  
*(Free tier — may take ~30s to load on first request)*

A Python-based portfolio optimization tool that supports **three objectives** — return, risk, and liquidity — with a simple Flask web interface for visualization and interaction.

## Features
- Generate a set of Pareto-optimal portfolios using multi-objective optimization (`pymoo`).
- Visualize risk/return/liquidity trade-offs in the browser.
- Allow users to express preferences, which the system uses to adjust optimization weights.
- Rolling strategy support for backtesting.
- Configurable via `.env` for API keys and secrets.

## Project Structure
```
portfolio_optimizer/
├── backend/                  # Core optimization logic
│   ├── data_loader.py         # Load financial data
│   ├── scalarization.py       # Scalarization methods
│   ├── optimizer.py           # Optimization engine
│   ├── rolling_strategy.py    # Rolling backtest logic
│   ├── preference_extractor.py# GPT-based preference extraction
│   ├── pareto_optimizer.py    # Pareto optimal set generation
│   └── portfolio_engine.py    # Main portfolio engine
├── frontend/                  # Flask web app
│   ├── app.py                 # Entry point for web interface
│   ├── templates/
│   │   └── index.html         # HTML template
│   └── static/
│       ├── style.css          # CSS styling
│       └── script.js          # Client-side JS
├── config.py                  # Configuration (loads from .env)
├── requirements.txt           # Python dependencies
├── run.py                     # Script to launch the app
├── .env.example               # Example environment variables
├── .gitignore                 # Ignore patterns
└── README.md                  # Project documentation
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/portfolio_optimizer.git
cd portfolio_optimizer
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the example file and edit it:
```bash
cp .env.example .env
```
Fill in your API keys in `.env`:
```
OPENAI_API_KEY="your-openai-api-key-here"

# Flask Secret Key (dev can be any string, prod should be random & secure)
SECRET_KEY="your-secret-key-here"
```

> **Note:** For development, `SECRET_KEY` can be any string.

## Usage

Run the application:
```bash
python run.py
```
or
```bash
python frontend/app.py
```
Then open your browser at `http://127.0.0.1:5000`.

## Development Notes
- `.env` contains sensitive credentials and **should not** be committed to version control.
- Use `.env.example` to document the required environment variables without exposing real keys.
- The backend logic is modular; you can replace or extend `scalarization.py`, `optimizer.py`, etc.
- Rolling strategy logic is in `rolling_strategy.py`, and can be customized for different backtest windows.
- Preference extraction uses the OpenAI API — make sure your API key has sufficient quota.

## License
This project is open-source under the MIT License.

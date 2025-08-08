// frontend/static/script.js
class PortfolioOptimizer {
    constructor() {
        this.initialized = false;
        this.currentPreferences = null;
        this.portfolioChart = null;
        this.paretoChart = null;
        this.optimizationStage = 'initial';
        this.paretoSolutions = null;
        this.initializeEventListeners();
        this.checkSystemStatus();
    }

    initializeEventListeners() {
        // Initialize button
        document.getElementById('init-btn').addEventListener('click', () => {
            this.initializeSystem();
        });

        // Generate Pareto frontier button
        document.getElementById('generate-pareto-btn').addEventListener('click', () => {
            this.generateParetoFrontier();
        });

        // Optimize with preferences button
        document.getElementById('optimize-btn').addEventListener('click', () => {
            this.optimizeWithPreferences();
        });

        // Reset optimization button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetOptimization();
        });

        // Send message button
        document.getElementById('send-btn').addEventListener('click', () => {
            this.sendMessage();
        });

        // Enter key in textarea
        document.getElementById('user-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        document.getElementById('user-input').addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = e.target.scrollHeight + 'px';
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            this.initialized = data.initialized;
            this.currentPreferences = data.current_preferences;
            this.optimizationStage = data.optimization_stage;
            
            if (this.initialized) {
                this.showPanel('main-interface');
                this.hidePanel('init-panel');
                this.updateInterfaceForStage();
                this.addSystemMessage("System is ready! You can start by generating Pareto optimal solutions or directly optimize with preferences.");
            } else {
                this.showPanel('init-panel');
                this.hidePanel('main-interface');
            }
        } catch (error) {
            console.error('Error checking system status:', error);
        }
    }

    updateInterfaceForStage() {
        const generateBtn = document.getElementById('generate-pareto-btn');
        const optimizeBtn = document.getElementById('optimize-btn');
        const paretoSection = document.getElementById('pareto-solutions');
        
        switch (this.optimizationStage) {
            case 'initial':
                generateBtn.style.display = 'inline-block';
                optimizeBtn.style.display = 'inline-block';
                paretoSection.style.display = 'none';
                break;
            case 'pareto_selection':
                generateBtn.style.display = 'none';
                optimizeBtn.style.display = 'inline-block';
                paretoSection.style.display = 'block';
                break;
            case 'preference_refinement':
                generateBtn.style.display = 'inline-block';
                optimizeBtn.style.display = 'inline-block';
                paretoSection.style.display = 'none';
                break;
        }
    }

    async initializeSystem() {
        const initBtn = document.getElementById('init-btn');
        const statusDiv = document.getElementById('init-status');
        
        try {
            initBtn.disabled = true;
            initBtn.textContent = 'Initializing...';
            this.showLoading();

            const tickers = document.getElementById('tickers').value
                .split(',')
                .map(t => t.trim().toUpperCase())
                .filter(t => t.length > 0);
            
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;

            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    tickers: tickers,
                    start_date: startDate,
                    end_date: endDate
                })
            });

            const data = await response.json();

            if (data.success) {
                this.initialized = true;
                this.hidePanel('init-panel');
                this.showPanel('main-interface');
                this.addSystemMessage(data.message);
                this.addSystemMessage(`Data loaded: ${data.data_info.n_tickers} stocks from ${data.data_info.date_range.start} to ${data.data_info.date_range.end}`);
                this.optimizationStage = 'initial';
                this.updateInterfaceForStage();
            } else {
                statusDiv.innerHTML = `<div class="error">Error: ${data.message}</div>`;
            }

        } catch (error) {
            statusDiv.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
        } finally {
            this.hideLoading();
            initBtn.disabled = false;
            initBtn.textContent = 'Initialize System';
        }
    }

    async generateParetoFrontier() {
        if (!this.initialized) {
            this.addSystemMessage("Please initialize the system first.");
            return;
        }

        try {
            this.showLoading();
            this.addSystemMessage("Generating Pareto optimal solutions... This may take a moment.");

            const response = await fetch('/api/generate_pareto', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            });

            const data = await response.json();

            if (data.success) {
                this.paretoSolutions = data.pareto_solutions;
                this.optimizationStage = 'pareto_selection';
                this.updateInterfaceForStage();
                
                this.addSystemMessage(data.message);
                this.displayParetoSolutions(data.pareto_solutions);
                this.addSystemMessage("Please select one of the solutions above that best matches your preferences, or you can directly optimize with custom preferences.");
            } else {
                this.addSystemMessage(`Error: ${data.message}`);
            }

        } catch (error) {
            this.addSystemMessage(`Network error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayParetoSolutions(solutions) {
        const paretoSection = document.getElementById('pareto-solutions');
        const solutionsContainer = document.getElementById('solutions-container');
        
        solutionsContainer.innerHTML = '';
        
        solutions.forEach((solution, index) => {
            const solutionDiv = document.createElement('div');
            solutionDiv.className = 'pareto-solution';
            solutionDiv.innerHTML = `
                <div class="solution-header">
                    <h4>Solution ${index + 1}</h4>
                    <button class="select-solution-btn" onclick="portfolioOptimizer.selectParetoSolution(${solution.id})">
                        Select This Portfolio
                    </button>
                </div>
                <div class="solution-metrics">
                    <div class="metric">
                        <span class="metric-label">Expected Return:</span>
                        <span class="metric-value">${(solution.metrics.return * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Risk (Volatility):</span>
                        <span class="metric-value">${(solution.metrics.risk * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Liquidity Score:</span>
                        <span class="metric-value">${solution.metrics.liquidity.toFixed(3)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sharpe Ratio:</span>
                        <span class="metric-value">${solution.metrics.sharpe_ratio.toFixed(3)}</span>
                    </div>
                </div>
                <div class="solution-weights">
                    <h5>Portfolio Allocation:</h5>
                    <div class="weights-grid">
                        ${Object.entries(solution.weights)
                            .sort((a, b) => b[1] - a[1])
                            .map(([ticker, weight]) => `
                                <div class="weight-item">
                                    <span class="ticker">${ticker}</span>
                                    <span class="weight">${(weight * 100).toFixed(1)}%</span>
                                </div>
                            `).join('')}
                    </div>
                </div>
            `;
            solutionsContainer.appendChild(solutionDiv);
        });
        
        paretoSection.style.display = 'block';
        
        // Create Pareto frontier visualization
        this.createParetoChart(solutions);
    }

    createParetoChart(solutions) {
        const ctx = document.getElementById('pareto-chart');
        if (!ctx) return;

        if (this.paretoChart) {
            this.paretoChart.destroy();
        }

        const data = solutions.map((sol, index) => ({
            x: sol.metrics.risk * 100,
            y: sol.metrics.return * 100,
            label: `Solution ${index + 1}`,
            liquidity: sol.metrics.liquidity,
            sharpe: sol.metrics.sharpe_ratio,
            id: sol.id
        }));

        this.paretoChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Pareto Optimal Solutions',
                    data: data,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 12
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Pareto Frontier: Risk vs Return'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return [
                                    `${point.label}`,
                                    `Return: ${point.y.toFixed(2)}%`,
                                    `Risk: ${point.x.toFixed(2)}%`,
                                    `Liquidity: ${point.liquidity.toFixed(3)}`,
                                    `Sharpe: ${point.sharpe.toFixed(3)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Risk (Volatility %)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Expected Return (%)'
                        }
                    }
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        const index = elements[0].index;
                        const solutionId = data[index].id;
                        this.selectParetoSolution(solutionId);
                    }
                }
            }
        });
    }

    async selectParetoSolution(solutionId) {
        try {
            this.showLoading();
            this.addSystemMessage(`Analyzing your selection...`);

            const response = await fetch('/api/select_pareto_solution', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    solution_id: solutionId
                })
            });

            const data = await response.json();

            if (data.success) {
                this.currentPreferences = data.extracted_preferences;
                this.optimizationStage = 'preference_refinement';
                this.updateInterfaceForStage();
                
                this.addSystemMessage(data.message);
                this.displayPreferences(data.extracted_preferences);
                
                // Automatically optimize with extracted preferences
                await this.optimizeWithPreferences();
                
                this.addSystemMessage("Based on your selection, I've optimized a portfolio. Are you satisfied with this result? If not, please tell me how you'd like to adjust your preferences.");
            } else {
                this.addSystemMessage(`Error: ${data.message}`);
            }

        } catch (error) {
            this.addSystemMessage(`Network error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async optimizeWithPreferences() {
        if (!this.initialized) {
            this.addSystemMessage("Please initialize the system first.");
            return;
        }

        try {
            this.showLoading();
            this.addSystemMessage("Optimizing portfolio with current preferences...");

            const response = await fetch('/api/optimize_with_preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    preferences: this.currentPreferences,
                    method: 'tchebycheff'
                })
            });

            const data = await response.json();

            if (data.success) {
                this.addSystemMessage(`${data.message} (Iteration ${data.iteration})`);
                this.displayOptimizationResult(data.result);
            } else {
                this.addSystemMessage(`Error: ${data.message}`);
            }

        } catch (error) {
            this.addSystemMessage(`Network error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async sendMessage() {
        const userInput = document.getElementById('user-input');
        const message = userInput.value.trim();
        
        if (!message) return;

        this.addUserMessage(message);
        userInput.value = '';
        userInput.style.height = 'auto';

        try {
            this.showLoading();

            const response = await fetch('/api/refine_preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_input: message
                })
            });

            const data = await response.json();

            if (data.success) {
                this.currentPreferences = data.refined_preferences;
                this.addSystemMessage(data.message);
                this.displayPreferences(data.refined_preferences);
                
                if (data.explanation) {
                    this.addSystemMessage(data.explanation);
                }
                
                // Automatically optimize with refined preferences
                await this.optimizeWithPreferences();
            } else {
                this.addSystemMessage(`Error: ${data.message}`);
            }

        } catch (error) {
            this.addSystemMessage(`Network error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async resetOptimization() {
        try {
            const response = await fetch('/api/reset_optimization', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (data.success) {
                this.optimizationStage = 'initial';
                this.paretoSolutions = null;
                this.currentPreferences = null;
                
                // Clear UI
                document.getElementById('pareto-solutions').style.display = 'none';
                document.getElementById('results-section').style.display = 'none';
                document.getElementById('chat-messages').innerHTML = '';
                
                this.updateInterfaceForStage();
                this.addSystemMessage(data.message);
                this.addSystemMessage("You can now start fresh by generating Pareto solutions or directly optimizing with preferences.");
            } else {
                this.addSystemMessage(`Error: ${data.message}`);
            }

        } catch (error) {
            this.addSystemMessage(`Network error: ${error.message}`);
        }
    }

    displayPreferences(preferences) {
        const preferencesDiv = document.getElementById('current-preferences');
        if (!preferencesDiv) return;

        preferencesDiv.innerHTML = `
            <h4>Current Preferences:</h4>
            <div class="preferences-display">
                <div class="preference-item">
                    <span class="preference-label">Return Focus:</span>
                    <span class="preference-value">${(preferences.return * 100).toFixed(1)}%</span>
                </div>
                <div class="preference-item">
                    <span class="preference-label">Risk Tolerance:</span>
                    <span class="preference-value">${(preferences.risk * 100).toFixed(1)}%</span>
                </div>
                <div class="preference-item">
                    <span class="preference-label">Liquidity Focus:</span>
                    <span class="preference-value">${(preferences.liquidity * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
        preferencesDiv.style.display = 'block';
    }

    displayOptimizationResult(result) {
        const resultsSection = document.getElementById('results-section');
        const weightsContainer = document.getElementById('weights-container');
        const metricsContainer = document.getElementById('metrics-container');
        
        // Display portfolio weights
        weightsContainer.innerHTML = Object.entries(result.weights)
            .sort((a, b) => b[1] - a[1])
            .map(([ticker, weight]) => `
                <div class="weight-item">
                    <div class="weight-ticker">${ticker}</div>
                    <div class="weight-value">${(weight * 100).toFixed(1)}%</div>
                    <div class="weight-bar">
                        <div class="weight-fill" style="width: ${weight * 100}%"></div>
                    </div>
                </div>
            `).join('');

        // Display portfolio metrics
        metricsContainer.innerHTML = `
            <div class="metric-item">
                <div class="metric-label">Expected Return</div>
                <div class="metric-value">${(result.metrics.return * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Risk (Volatility)</div>
                <div class="metric-value">${(result.metrics.risk * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Liquidity Score</div>
                <div class="metric-value">${result.metrics.liquidity.toFixed(3)}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">${result.metrics.sharpe_ratio.toFixed(3)}</div>
            </div>
        `;

        resultsSection.style.display = 'block';
        
        // Create portfolio chart
        this.createPortfolioChart(result.weights);
    }

    createPortfolioChart(weights) {
        const ctx = document.getElementById('portfolio-chart');
        if (!ctx) return;

        if (this.portfolioChart) {
            this.portfolioChart.destroy();
        }

        const sortedWeights = Object.entries(weights)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10); // Show top 10 holdings

        this.portfolioChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: sortedWeights.map(([ticker]) => ticker),
                datasets: [{
                    data: sortedWeights.map(([, weight]) => weight * 100),
                    backgroundColor: [
                        '#667eea', '#764ba2', '#f093fb', '#f5576c',
                        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
                        '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Portfolio Allocation'
                    },
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    addUserMessage(message) {
        this.addMessage(message, 'user');
    }

    addSystemMessage(message) {
        this.addMessage(message, 'system');
    }

    addMessage(message, sender) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString();
        messageDiv.innerHTML = `
            <div class="message-content">${message}</div>
            <div class="message-time">${timestamp}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showLoading() {
        document.getElementById('loading').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showPanel(panelId) {
        document.getElementById(panelId).style.display = 'block';
    }

    hidePanel(panelId) {
        document.getElementById(panelId).style.display = 'none';
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', function() {
    window.portfolioOptimizer = new PortfolioOptimizer();
});


// Performance optimization: Lazy loading and debouncing
let loadingStates = new Set();
let chartLoadQueue = [];

// Global configuration for Plotly dark theme
const DARK_THEME_CONFIG = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e6edf3' },
    colorway: ['#58a6ff', '#3fb950', '#d2a8ff', '#ff7b72', '#f0e68c', '#79c0ff', '#a5f3fc', '#fbbf24'],
    responsive: true
};

// Utility functions with performance optimization
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i> ${message}
            </div>
        `;
    }
}

function showLoading(elementId, message = "Loading...") {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="loading">
                <i class="fas fa-spinner fa-spin pulse"></i>
                <p class="mt-2">${message}</p>
            </div>
        `;
    }
}

// Debounced fetch function for better performance
function debouncedFetch(url, delay = 100) {
    return new Promise((resolve) => {
        setTimeout(() => {
            fetch(url).then(resolve);
        }, delay);
    });
}

// Enhanced system status check
function checkSystemStatus() {
    const statusElement = document.getElementById('systemStatus');
    
    fetch('/api/test')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const regimeStatus = data.regime_detector_available ? '🧠 AI Regime Detection' : '⚠️ Basic Analysis';
                statusElement.innerHTML = `
                    <i class="fas fa-check-circle text-success"></i> 
                    System Ready: ${data.data_shape} | ${data.sectors} sectors | ${regimeStatus}
                `;
                statusElement.className = 'system-status success fade-in-up';
                
                // Update KPIs with animation
                updateKPIWithAnimation('totalCompanies', data.data_shape.split(' ')[0]);
                updateKPIWithAnimation('totalSectors', data.sectors);
                updateKPIWithAnimation('topCAGR', data.top_cagr + '%');
                
                // Load dashboard components
                loadDashboardComponents();
            } else {
                statusElement.innerHTML = `
                    <i class="fas fa-exclamation-triangle text-danger"></i> 
                    System Error: ${data.error}
                `;
                statusElement.className = 'system-status error';
            }
        })
        .catch(error => {
            console.error('System status check failed:', error);
            statusElement.innerHTML = `
                <i class="fas fa-exclamation-triangle text-danger"></i> 
                Connection Error: Cannot reach server
            `;
            statusElement.className = 'system-status error';
        });
}

// Animated KPI updates
function updateKPIWithAnimation(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.opacity = '0.5';
        setTimeout(() => {
            element.textContent = value;
            element.style.opacity = '1';
        }, 150);
    }
}

// Load dashboard components with staggered timing
function loadDashboardComponents() {
    const loadingSequence = [
        { fn: loadTopPerformers, delay: 100 },
        { fn: loadSectorAnalysis, delay: 300 },
        { fn: loadMonthlyPredictions, delay: 500 },
        { fn: loadMarketRegime, delay: 700 },
        { fn: loadRegimeHistory, delay: 900 },
        { fn: loadSectorRegimeImpact, delay: 1100 },
        { fn: loadPerformanceChart, delay: 1300 },
        { fn: loadSectorChart, delay: 1500 }
    ];

    loadingSequence.forEach(({ fn, delay }) => {
        setTimeout(fn, delay);
    });
}

// Load market regime analysis
function loadMarketRegime() {
    fetch('/api/market_regime')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('currentRegimeCard', 'Regime detection not available');
                return;
            }
            
            renderCurrentRegime(data);
            updateMarketIndicators(data.market_indicators);
        })
        .catch(error => {
            console.error('Error loading market regime:', error);
            showError('currentRegimeCard', 'Regime detection not available');
        });
}

function renderCurrentRegime(data) {
    const regimeColor = data.characteristics.color || '#58a6ff';
    const regimeIcon = data.characteristics.icon || 'fa-chart-line';
    
    document.getElementById('currentRegimeCard').innerHTML = `
        <div class="fade-in-up">
            <i class="fas ${regimeIcon} fa-3x mb-3" style="color: ${regimeColor}"></i>
            <h4 style="color: ${regimeColor}">${data.current_regime}</h4>
            <p class="mb-2">${data.characteristics.description || 'Current market regime'}</p>
            <div class="badge bg-primary">
                ${data.confidence}% Confidence
            </div>
            <div class="mt-3">
                <small class="text-muted">Investment Strategy:</small>
                <p class="small mt-1">${data.characteristics.investment_strategy || 'General market approach'}</p>
            </div>
        </div>
    `;
}

function updateMarketIndicators(indicators) {
    document.getElementById('marketVolatility').textContent = indicators.market_volatility + '%';
    document.getElementById('marketReturns').textContent = indicators.market_returns + '%';
    document.getElementById('sectorCorrelation').textContent = indicators.sector_correlation;
    document.getElementById('momentumScore').textContent = indicators.momentum_score;
}

function loadRegimeHistory() {
    fetch('/api/regime_history')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('regimeHistoryTable', data.error);
                return;
            }
            
            let tableHtml = `
                <div class="table-responsive">
                    <table class="table table-dark table-sm">
                        <thead>
                            <tr>
                                <th>Period</th>
                                <th>Regime</th>
                                <th>Duration</th>
                                <th>Impact</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            data.forEach((regime, index) => {
                const regimeColor = getRegimeColor(regime.regime);
                tableHtml += `
                    <tr class="fade-in-up" style="animation-delay: ${index * 0.1}s">
                        <td><strong>${regime.period}</strong></td>
                        <td><span class="badge" style="background: ${regimeColor}">${regime.regime}</span></td>
                        <td>${regime.duration}</td>
                        <td><small>${regime.impact}</small></td>
                    </tr>
                `;
            });
            
            tableHtml += '</tbody></table></div>';
            document.getElementById('regimeHistoryTable').innerHTML = tableHtml;
        })
        .catch(error => {
            console.error('Error loading regime history:', error);
            showError('regimeHistoryTable', 'Failed to load regime history');
        });
}

function loadSectorRegimeImpact() {
    fetch('/api/regime_sector_impact')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('sectorRegimeImpact', data.error);
                return;
            }
            
            let html = '';
            data.forEach((sector, index) => {
                const performanceColor = getPerformanceColor(sector.performance_type);
                html += `
                    <div class="border-bottom border-secondary py-2 fade-in-up" style="animation-delay: ${index * 0.05}s">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <strong style="color: ${getSectorColor(sector.sector)}">${sector.sector}</strong>
                                <br>
                                <small class="text-muted">${sector.performance_type}</small>
                            </div>
                            <div class="text-end">
                                <span class="badge" style="background: ${performanceColor}">
                                    ${sector.best_regime}
                                </span>
                                <br>
                                <small>${sector.companies} companies</small>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            document.getElementById('sectorRegimeImpact').innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading sector regime impact:', error);
            showError('sectorRegimeImpact', 'Failed to load sector impact data');
        });
}

// Top performers with caching
let topPerformersCache = new Map();

function loadTopPerformers(metric = 'CAGR') {
    const cacheKey = `${metric}_10`;
    
    if (topPerformersCache.has(cacheKey)) {
        renderTopPerformers(topPerformersCache.get(cacheKey));
        return;
    }

    debouncedFetch(`/api/top_performers?metric=${metric}&limit=10`, 100)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('topPerformersTable', data.error);
                return;
            }
            
            topPerformersCache.set(cacheKey, data);
            renderTopPerformers(data);
        })
        .catch(error => {
            console.error('Error loading top performers:', error);
            showError('topPerformersTable', 'Failed to load data');
        });
}

function renderTopPerformers(data) {
    let tableHtml = '';
    data.forEach((company, index) => {
        const rankBadge = index < 3 ? 'bg-warning' : 'bg-primary';
        const sectorColor = getSectorColor(company.Sector);
        
        tableHtml += `
            <tr class="fade-in-up" style="animation-delay: ${index * 0.1}s">
                <td><span class="badge ${rankBadge}">${index + 1}</span></td>
                <td><strong>${company.Company}</strong></td>
                <td><span class="badge" style="background: ${sectorColor}">${company.Sector}</span></td>
                <td><span class="badge bg-success">${company.CAGR}%</span></td>
                <td><span class="badge bg-info">${company.Sharpe_Ratio || 'N/A'}</span></td>
            </tr>
        `;
    });
    
    document.getElementById('topPerformersTable').innerHTML = tableHtml;
    
    // Update KPIs
    if (data.length > 0) {
        updateKPIWithAnimation('topCAGRCompany', data[0].Company);
        
        const bestSharpe = data.reduce((prev, current) => 
            (prev.Sharpe_Ratio > current.Sharpe_Ratio) ? prev : current);
        updateKPIWithAnimation('bestSharpe', bestSharpe.Sharpe_Ratio || 'N/A');
        updateKPIWithAnimation('bestSharpeCompany', bestSharpe.Company);
    }
}

// Load sector analysis
function loadSectorAnalysis() {
    fetch('/api/sector_analysis')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('sectorAnalysisTable', data.error);
                return;
            }
            
            let tableHtml = '';
            data.forEach((sector, index) => {
                const riskLevel = (sector.Avg_Volatility || 0) > 30 ? 'High' : 
                                (sector.Avg_Volatility || 0) > 20 ? 'Medium' : 'Low';
                const riskBadge = riskLevel === 'High' ? 'bg-danger' : 
                                 riskLevel === 'Medium' ? 'bg-warning' : 'bg-success';
                
                tableHtml += `
                    <tr class="fade-in-up" style="animation-delay: ${index * 0.05}s">
                        <td><strong style="color: ${getSectorColor(sector.Sector)}">${sector.Sector}</strong></td>
                        <td><span class="badge bg-primary">${sector.Avg_CAGR || 'N/A'}%</span></td>
                        <td><span class="badge bg-secondary">${sector.Avg_Volatility || 'N/A'}%</span></td>
                        <td><span class="badge bg-info">${sector.Avg_Sharpe_Ratio || 'N/A'}</span></td>
                        <td><span class="badge bg-dark">${sector.Companies || 'N/A'}</span></td>
                        <td><span class="badge ${riskBadge}">${riskLevel}</span></td>
                    </tr>
                `;
            });
            
            document.getElementById('sectorAnalysisTable').innerHTML = tableHtml;
        })
        .catch(error => {
            console.error('Error loading sector analysis:', error);
            showError('sectorAnalysisTable', 'Failed to load sector data');
        });
}

// Load monthly predictions
function loadMonthlyPredictions() {
    fetch('/api/monthly_predictions')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('monthlyPredictions', data.error);
                return;
            }
            
            let html = '';
            const gradients = ['tertiary', 'quaternary', 'primary', 'secondary'];
            
            data.forEach((prediction, index) => {
                const gradientClass = gradients[index % gradients.length];
                html += `
                    <div class="prediction-card-small fade-in-up" style="animation-delay: ${index * 0.1}s">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="mb-1">
                                    <i class="fas fa-star"></i> #${index + 1} Sector Leader
                                </h6>
                                <p class="mb-2">
                                    <strong>${prediction.company}</strong><br>
                                    <small>${prediction.sector} Sector</small>
                                </p>
                                <span class="badge bg-light text-dark">
                                    CAGR: ${prediction.expected_cagr}%
                                </span>
                            </div>
                            <i class="fas fa-chart-trend-up fa-2x opacity-50"></i>
                        </div>
                    </div>
                `;
            });
            
            document.getElementById('monthlyPredictions').innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading monthly predictions:', error);
            showError('monthlyPredictions', 'Failed to load predictions');
        });
}

// Optimized chart loading with lazy rendering
function loadPerformanceChart() {
    showLoading('performanceChart', 'Generating risk vs return analysis...');
    
    fetch('/api/charts/performance_scatter')
        .then(response => response.json())
        .then(fig => {
            if (fig.error) {
                showError('performanceChart', fig.error);
                return;
            }
            
            // Apply dark theme configuration
            fig.layout = { 
                ...fig.layout, 
                ...DARK_THEME_CONFIG,
                height: 500,
                margin: { l: 60, r: 60, t: 60, b: 60 }
            };
            
            // Use requestAnimationFrame for smooth rendering
            requestAnimationFrame(() => {
                Plotly.newPlot('performanceChart', fig.data, fig.layout, {
                    responsive: true,
                    displayModeBar: false
                });
            });
        })
        .catch(error => {
            console.error('Error loading performance chart:', error);
            showError('performanceChart', 'Failed to load chart');
        });
}

function loadSectorChart() {
    showLoading('sectorChart', 'Comparing sector performance...');
    
    fetch('/api/charts/sector_comparison')
        .then(response => response.json())
        .then(fig => {
            if (fig.error) {
                showError('sectorChart', fig.error);
                return;
            }
            
            // Apply dark theme
            fig.layout = { 
                ...fig.layout, 
                ...DARK_THEME_CONFIG,
                height: 400,
                margin: { l: 60, r: 60, t: 60, b: 100 }
            };
            
            requestAnimationFrame(() => {
                Plotly.newPlot('sectorChart', fig.data, fig.layout, {
                    responsive: true,
                    displayModeBar: false
                });
            });
        })
        .catch(error => {
            console.error('Error loading sector chart:', error);
            showError('sectorChart', 'Failed to load chart');
        });
}

// Color coding for sectors
function getSectorColor(sector) {
    const colors = {
        'IT': '#58a6ff',
        'Banking': '#3fb950',
        'Energy': '#ff7b72',
        'FMCG': '#d2a8ff',
        'Pharmaceuticals': '#f0e68c',
        'Automotive': '#79c0ff',
        'Metals': '#fbbf24',
        'Construction': '#a5f3fc',
        'Oil & Gas': '#ff7b72',
        'Financial Services': '#3fb950',
        'Healthcare': '#f0e68c',
        'Media': '#d2a8ff',
        'Retail': '#79c0ff',
        'Chemicals': '#fbbf24',
        'Real Estate': '#a5f3fc',
        'Textiles': '#58a6ff'
    };
    return colors[sector] || '#7d8590';
}

function getRegimeColor(regime) {
    const colors = {
        'Bear Market': '#ff7b72',
        'Recovery': '#f0e68c', 
        'Bull Market': '#3fb950',
        'High Volatility': '#d2a8ff'
    };
    return colors[regime] || '#7d8590';
}

function getPerformanceColor(performance) {
    if (performance.includes('Defensive')) return '#3fb950';
    if (performance.includes('Cyclical')) return '#58a6ff';
    if (performance.includes('Volatile')) return '#ff7b72';
    return '#7d8590';
}

// Enhanced prediction form with better UX
document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = {
                volatility: parseFloat(document.getElementById('volatility').value),
                sharpe_ratio: parseFloat(document.getElementById('sharpe_ratio').value),
                max_drawdown: parseFloat(document.getElementById('max_drawdown').value),
                beta: parseFloat(document.getElementById('beta').value),
                sector: document.getElementById('sector').value,
                sortino_ratio: parseFloat(document.getElementById('sharpe_ratio').value) * 1.1,
                skewness: 0,
                kurtosis: 0,
                momentum_3m: 0,
                momentum_6m: 0,
                current_rsi: 50,
                volume_volatility: 0.5
            };

            // Enhanced loading state
            document.getElementById('predictionResult').innerHTML = `
                <div class="prediction-result">
                    <div class="text-center">
                        <i class="fas fa-brain fa-2x pulse text-primary mb-3"></i>
                        <p>AI model calculating prediction...</p>
                        <div class="progress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                                 style="width: 100%"></div>
                        </div>
                    </div>
                </div>
            `;

            fetch('/api/predict_cagr', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const confidenceColor = data.confidence > 80 ? 'text-success' : 
                                           data.confidence > 60 ? 'text-warning' : 'text-danger';
                    
                    document.getElementById('predictionResult').innerHTML = `
                        <div class="prediction-result fade-in-up">
                            <h5 class="text-primary mb-3">
                                <i class="fas fa-chart-line"></i> Prediction Result
                            </h5>
                            <div class="row text-center">
                                <div class="col-6">
                                    <div class="border-end border-secondary">
                                        <h2 class="text-primary mb-1">${data.predicted_cagr}%</h2>
                                        <small class="text-muted">Predicted CAGR</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <h2 class="${confidenceColor} mb-1">${data.confidence}%</h2>
                                    <small class="text-muted">Confidence Level</small>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    document.getElementById('predictionResult').innerHTML = `
                        <div class="error-message">
                            <i class="fas fa-exclamation-triangle"></i> Error: ${data.error}
                        </div>
                    `;
                }
            })
            .catch(error => {
                console.error('Prediction error:', error);
                document.getElementById('predictionResult').innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-triangle"></i> Network error occurred
                    </div>
                `;
            });
        });
    }

    // Metric selector with smooth transitions
    const metricSelector = document.getElementById('metricSelector');
    if (metricSelector) {
        metricSelector.addEventListener('change', function() {
            const table = document.getElementById('topPerformersTable');
            table.style.opacity = '0.5';
            
            setTimeout(() => {
                loadTopPerformers(this.value);
                table.style.opacity = '1';
            }, 200);
        });
    }

    // Initialize dashboard
    console.log('🚀 NIFTY 500 Dark Mode Dashboard initializing...');
    
    // Add loading performance tracking
    const startTime = performance.now();
    
    checkSystemStatus();
    
    // Monitor loading performance
    window.addEventListener('load', function() {
        const loadTime = performance.now() - startTime;
        console.log(`📊 Dashboard loaded in ${loadTime.toFixed(2)}ms`);
    });
    
    // Add resize observer for responsive charts
    if (window.ResizeObserver) {
        const resizeObserver = new ResizeObserver(entries => {
            entries.forEach(entry => {
                if (entry.target.id === 'performanceChart' || entry.target.id === 'sectorChart') {
                    Plotly.Plots.resize(entry.target.id);
                }
            });
        });
        
        const chartElements = ['performanceChart', 'sectorChart'];
        chartElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) resizeObserver.observe(element);
        });
    }
});

// Add error boundary
window.addEventListener('error', function(e) {
    console.error('🚨 Dashboard error:', e.error);
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    topPerformersCache.clear();
});

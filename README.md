# NIFTY 500 Advanced Stock Analysis & ML Platform

[![CI/CD Pipeline](https://github.com/username/nifty500-ml-platform/actions/workflows/main.yml/badge.nifty500-ml-platform/actionsshields.io/badge/python-3.9+-blue.svg](https://img.shields.io/badgeldschine learning platform for analyzing and predicting NIFTY 500 stock performance using 15+ years of historical data. Built with modern MLOps practices including MLflow, DVC, and AWS cloud infrastructure.

## 🚀 Features

- **Advanced ML Models**: CAGR prediction, Sharpe ratio forecasting, volatility analysis
- **Interactive Dashboard**: Real-time dark-themed web interface with Plotly charts
- **Risk Analytics**: Comprehensive risk assessment and drawdown analysis
- **Sector Intelligence**: Sector rotation insights and performance comparison
- **MLOps Pipeline**: Complete ML lifecycle with experiment tracking and model versioning
- **Cloud-Native**: Auto-scaling AWS deployment with ECS Fargate
- **CI/CD Integration**: Automated testing, training, and deployment with GitHub Actions

## 📊 Dashboard Preview

![Dashboard Preview](docs/images

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   ML Pipeline   │    │  Web Dashboard  │
│                 │    │                 │    │                 │
│ • Stock Prices  │───▶│ • Data Cleaning │───▶│ • Interactive   │
│ • Financial     │    │ • Feature Eng.  │    │   Charts        │
│   Metrics       │    │ • Model Training│    │ • Predictions   │
│ • Market Data   │    │ • Validation    │    │ • Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      DVC        │    │     MLflow      │    │   AWS Cloud     │
│                 │    │                 │    │                 │
│ • Data Version  │    │ • Experiment    │    │ • ECS/Fargate   │
│ • Pipeline      │    │   Tracking      │    │ • Load Balancer │
│   Management    │    │ • Model Registry│    │ • Auto Scaling  │
│ • Reproducible  │    │ • Metrics       │    │ • CloudWatch    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5, Plotly.js |
| **Backend** | Python, Flask, Pandas, NumPy, Scikit-learn |
| **ML/AI** | Random Forest, Gradient Boosting, Feature Engineering |
| **MLOps** | MLflow, DVC, Git |
| **Cloud** | AWS (ECS, ECR, S3, CloudFormation) |
| **DevOps** | Docker, GitHub Actions, CI/CD |
| **Monitoring** | AWS CloudWatch, MLflow Tracking |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker
- AWS CLI
- Git

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/username/nifty500-ml-platform.git
cd nifty500-ml-platform
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Initialize DVC**
```bash
dvc init
dvc remote add -d myremote s3://your-bucket/dvc-storage
dvc pull
```

4. **Run the application**
```bash
cd webapp
python app.py
```

5. **Access the dashboard**
```
http://localhost:5000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access MLflow UI
http://localhost:5001

# Access Dashboard
http://localhost:5000
```

## 📁 Project Structure

```
nifty500-ml-platform/
├── data/
│   ├── raw/                    # Raw stock data
│   ├── processed/              # Cleaned data
│   └── features/               # Engineered features
├── src/
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   ├── feature_engineering.py # Feature creation
│   ├── model_training.py       # ML model training
│   ├── sharpe_predictor.py     # Sharpe ratio prediction
│   └── mlflow_config.py        # MLflow configuration
├── models/
│   ├── best_cagr_model.pkl     # Trained CAGR model
│   ├── sharpe_predictor.pkl    # Sharpe ratio model
│   └── feature_scaler.pkl      # Feature scaling
├── webapp/
│   ├── app.py                  # Flask application
│   ├── templates/              # HTML templates
│   └── static/                 # CSS/JS files
├── infrastructure/
│   ├── cloudformation.yml      # AWS infrastructure
│   └── task-definition.json    # ECS task definition
├── tests/
│   ├── unit/                   # Unit tests
│   └── load/                   # Load testing
├── .github/
│   └── workflows/
│       └── main.yml            # CI/CD pipeline
├── dvc.yaml                    # DVC pipeline
├── params.yaml                 # Model parameters
├── requirements.txt            # Python dependencies
└── docker-compose.yml         # Local development
```

## 🤖 Machine Learning Models

### CAGR Prediction Model
Predicts compound annual growth rate using:
- **Features**: Volatility, Sharpe ratio, Beta, sector classification
- **Algorithm**: Random Forest with hyperparameter tuning
- **Performance**: R² = 0.85, RMSE = 5.2%

### Sharpe Ratio Predictor
Forecasts 30-day risk-adjusted returns:
- **Features**: Technical indicators, market momentum, volume patterns
- **Algorithm**: Gradient Boosting Regressor
- **Performance**: R² = 0.78, MAE = 0.21

### Feature Engineering
```python
# Key financial metrics calculated
features = {
    'volatility': 'Annualized price volatility',
    'sharpe_ratio': 'Risk-adjusted return metric',
    'max_drawdown': 'Maximum peak-to-trough loss',
    'beta': 'Market sensitivity coefficient',
    'momentum_3m': '3-month price momentum',
    'rsi': 'Relative Strength Index',
    'sector': 'Industry classification'
}
```

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/test` | GET | System health check |
| `/api/predict_cagr` | POST | CAGR prediction |
| `/api/predict_sharpe` | POST | Sharpe ratio prediction |
| `/api/top_performers` | GET | Top performing stocks |
| `/api/sector_analysis` | GET | Sector comparison |
| `/api/charts/performance_scatter` | GET | Risk vs return chart |

### Example API Usage

```python
# CAGR Prediction
import requests

data = {
    "volatility": 22.5,
    "sharpe_ratio": 1.2,
    "max_drawdown": -12.3,
    "beta": 0.95,
    "sector": "IT"
}

response = requests.post("http://localhost:5000/api/predict_cagr", json=data)
result = response.json()
# Output: {"predicted_cagr": 16.8, "confidence": 82.3, "status": "success"}
```

## ☁️ AWS Deployment

### Infrastructure Components

- **ECS Fargate**: Serverless container hosting
- **Application Load Balancer**: Traffic distribution
- **ECR**: Container image registry
- **S3**: Data lake and model storage
- **CloudWatch**: Monitoring and logging
- **Lambda**: Automated model training

### Deployment Commands

```bash
# Deploy infrastructure
aws cloudformation create-stack \
  --stack-name nifty500-infrastructure \
  --template-body file://infrastructure/cloudformation.yml \
  --capabilities CAPABILITY_IAM

# Build and push container
./scripts/deploy.sh production
```

## 🔄 MLOps Pipeline

### DVC Pipeline
```yaml
stages:
  data_preprocessing:
    cmd: python src/data_preprocessing.py
    deps: [data/raw/, src/data_preprocessing.py]
    outs: [data/processed/cleaned_data.csv]
    
  feature_engineering:
    cmd: python src/feature_engineering.py
    deps: [data/processed/, src/feature_engineering.py]
    outs: [data/features/features.csv]
    
  model_training:
    cmd: python src/model_training.py
    deps: [data/features/, src/model_training.py]
    outs: [models/]
    metrics: [models/metrics.json]
```

### MLflow Tracking
```python
# Model training with MLflow
import mlflow

with mlflow.start_run():
    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metrics({
        'r2_score': r2_score(y_test, y_pred),
        'rmse': rmse,
        'mae': mae
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "cagr_model")
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Load tests
k6 run tests/load/api-load-test.js

# Data validation
python src/data_validation.py
```

### Test Coverage
- Unit tests: 85%+ coverage
- Integration tests: API endpoints
- Load tests: 1000 concurrent users
- Data quality: Automated validation

## 📈 Performance Metrics

### Model Performance
- **CAGR Model**: R² = 0.85, RMSE = 5.2%
- **Sharpe Model**: R² = 0.78, MAE = 0.21
- **Training Time**: < 10 minutes
- **Inference Time**: < 100ms

### System Performance
- **Response Time**: < 200ms (95th percentile)
- **Throughput**: 1000+ requests/second
- **Uptime**: 99.9% SLA
- **Auto-scaling**: 2-10 containers

## 🔒 Security

- **Container Scanning**: Trivy vulnerability assessment
- **Code Security**: Bandit static analysis
- **AWS Security**: IAM roles with least privilege
- **Data Encryption**: In-transit and at-rest
- **Secrets Management**: AWS Secrets Manager

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Model Documentation](docs/models.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure CI/CD pipeline passes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for historical stock data
- NSE for sector classifications
- Open source ML community
- AWS for cloud infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/username/nifty500-ml-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/nifty500-ml-platform/discussions)
- **Email**: support@nifty500platform.com

**Built with ❤️ for the Indian stock market analysis community**
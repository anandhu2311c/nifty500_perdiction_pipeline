# src/model_training.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class StockMLTraining:
    def __init__(self):
        # Set MLflow tracking to local directory
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("NIFTY500_Stock_Analysis")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
    def prepare_data(self):
        """Prepare training data with comprehensive features"""
        df = pd.read_csv("data/features/comprehensive_features.csv")
        
        # Remove rows with missing values
        df = df.dropna()
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Sectors: {df['Sector'].nunique()}")
        
        # Check sector distribution
        sector_counts = df['Sector'].value_counts()
        print(f"📊 Sector distribution:")
        print(sector_counts)
        
        # Remove sectors with only 1 company (can't stratify)
        sectors_to_keep = sector_counts[sector_counts >= 2].index
        df = df[df['Sector'].isin(sectors_to_keep)]
        
        print(f"📊 After filtering: {df.shape}")
        print(f"📊 Sectors kept: {df['Sector'].nunique()}")
        
        # Encode categorical variables
        le_sector = LabelEncoder()
        df['Sector_Encoded'] = le_sector.fit_transform(df['Sector'])
        
        # Feature selection for model training
        feature_columns = [
            'Volatility', 'Max_Drawdown', 'Sharpe_Ratio', 'Sortino_Ratio',
            'Beta', 'Skewness', 'Kurtosis', 'Momentum_3M', 'Momentum_6M',
            'Current_RSI', 'Volume_Volatility', 'Sector_Encoded'
        ]
        
        # Check if all feature columns exist
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) != len(feature_columns):
            missing_features = set(feature_columns) - set(available_features)
            print(f"⚠️ Missing features: {missing_features}")
            feature_columns = available_features
        
        X = df[feature_columns]
        y = df['CAGR']
        
        # Split data - remove stratify if it causes issues, or use random split
        try:
            # Try stratified split first
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=df['Sector']
            )
            print("✅ Used stratified split by sector")
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}")
            print("🔄 Using random split instead...")
            # Fall back to random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print("✅ Used random split")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📊 Training set: {X_train_scaled.shape}")
        print(f"📊 Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_sector, feature_columns
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (avoid division by zero)
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = 0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with hyperparameter tuning"""
        with mlflow.start_run(run_name="RandomForest_CAGR_Prediction"):
            
            # Model parameters
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "RandomForest")
            
            # Train model
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test, "Random Forest")
            
            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log feature importance
            feature_importance = model.feature_importances_
            mlflow.log_dict({f"feature_{i}": imp for i, imp in enumerate(feature_importance)}, "feature_importance.json")
            
            # Log model
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            return model, metrics
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        """Train Gradient Boosting model"""
        with mlflow.start_run(run_name="GradientBoosting_CAGR_Prediction"):
            
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 5,
                'subsample': 0.8,
                'random_state': 42
            }
            
            mlflow.log_params(params)
            mlflow.log_param("model_type", "GradientBoosting")
            
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model(model, X_test, y_test, "Gradient Boosting")
            
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            mlflow.sklearn.log_model(model, "gradient_boosting_model")
            
            return model, metrics
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model"""
        with mlflow.start_run(run_name="LinearRegression_CAGR_Prediction"):
            
            params = {
                'fit_intercept': True
            }
            
            mlflow.log_params(params)
            mlflow.log_param("model_type", "LinearRegression")
            
            model = LinearRegression(**params)
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model(model, X_test, y_test, "Linear Regression")
            
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            mlflow.sklearn.log_model(model, "linear_regression_model")
            
            return model, metrics
    
    def train_all_models(self):
        """Train all models and select the best one"""
        print("🚀 Starting comprehensive model training with MLflow...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, le_sector, feature_columns = self.prepare_data()
        
        # Train models
        models_performance = {}
        
        print("\n🌲 Training Random Forest...")
        rf_model, rf_metrics = self.train_random_forest(X_train, X_test, y_train, y_test)
        models_performance['RandomForest'] = (rf_model, rf_metrics)
        
        print("\n🚀 Training Gradient Boosting...")
        gb_model, gb_metrics = self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        models_performance['GradientBoosting'] = (gb_model, gb_metrics)
        
        print("\n📈 Training Linear Regression...")
        lr_model, lr_metrics = self.train_linear_regression(X_train, X_test, y_train, y_test)
        models_performance['LinearRegression'] = (lr_model, lr_metrics)
        
        # Select best model based on R² score
        best_model_name = max(models_performance.keys(), 
                             key=lambda k: models_performance[k][1]['R2_Score'])
        best_model, best_metrics = models_performance[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"🎯 Best R² Score: {best_metrics['R2_Score']:.4f}")
        print(f"📉 Best RMSE: {best_metrics['RMSE']:.4f}")
        
        # Save best model and artifacts
        joblib.dump(best_model, "models/best_cagr_model.pkl")
        joblib.dump(scaler, "models/feature_scaler.pkl")
        joblib.dump(le_sector, "models/sector_encoder.pkl")
        
        # Save feature columns
        with open("models/feature_columns.txt", "w") as f:
            f.write("\n".join(feature_columns))
        
        # Save model performance summary
        performance_summary = pd.DataFrame.from_dict(
            {name: metrics for name, (model, metrics) in models_performance.items()},
            orient='index'
        )
        performance_summary.to_csv("models/model_performance_summary.csv")
        
        print("✅ Model training completed!")
        print("📁 Models saved to 'models/' directory")
        print("📊 View MLflow UI with: mlflow ui --port 5001")
        
        return best_model, scaler, le_sector, feature_columns, best_metrics

if __name__ == "__main__":
    trainer = StockMLTraining()
    trainer.train_all_models()

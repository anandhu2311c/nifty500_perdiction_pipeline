# src/feature_engineering.py
import pandas as pd
import numpy as np
import os
from glob import glob

class FeatureEngineer:
    def __init__(self):
        self.processed_folder = "data/processed"
        self.features_folder = "data/features"
        os.makedirs(self.features_folder, exist_ok=True)
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive financial metrics for all companies"""
        processed_files = glob(f"{self.processed_folder}/*.csv")
        all_features = []
        
        for file in processed_files:
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                company = os.path.basename(file).replace('.csv', '')
                
                if len(df) < 500:  # Need minimum data for reliable calculations
                    continue
                
                # Basic metrics
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                years = (df.index[-1] - df.index[0]).days / 365.25
                
                # CAGR
                cagr = ((end_price / start_price) ** (1/years) - 1) * 100
                
                # Returns analysis
                daily_returns = df['Daily_Return'].dropna()
                
                # Volatility (annualized)
                volatility = daily_returns.std() * np.sqrt(252) * 100
                
                # Sharpe Ratio (assuming 6% risk-free rate)
                excess_return = daily_returns.mean() * 252 - 0.06
                sharpe_ratio = excess_return / (daily_returns.std() * np.sqrt(252))
                
                # Sortino Ratio
                downside_returns = daily_returns[daily_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(252)
                sortino_ratio = excess_return / downside_volatility if len(downside_returns) > 0 else 0
                
                # Maximum Drawdown
                rolling_max = df['Close'].cummax()
                drawdown = (df['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                
                # Calmar Ratio
                calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
                
                # Beta (simplified - using correlation with overall market trend)
                market_return = 0.12  # Assumed market return
                beta = daily_returns.corr(pd.Series([market_return/252] * len(daily_returns)))
                if pd.isna(beta):
                    beta = 1.0
                
                # Treynor Ratio
                treynor_ratio = excess_return / beta if beta != 0 else 0
                
                # Information Ratio
                tracking_error = daily_returns.std() * np.sqrt(252)
                information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
                
                # VaR (Value at Risk) - 5th percentile
                var_5 = np.percentile(daily_returns, 5) * 100
                
                # Expected Shortfall (Conditional VaR)
                expected_shortfall = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
                
                # Skewness and Kurtosis
                skewness = daily_returns.skew()
                kurtosis = daily_returns.kurtosis()
                
                # Price momentum
                price_momentum_3m = (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100 if len(df) >= 63 else 0
                price_momentum_6m = (df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1) * 100 if len(df) >= 126 else 0
                price_momentum_1y = (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else 0
                
                # Volume analysis
                avg_volume = df['Volume'].mean()
                volume_volatility = df['Volume'].std() / avg_volume if avg_volume > 0 else 0
                
                # Technical indicators
                current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
                
                features = {
                    'Company': company,
                    'Sector': df['Sector'].iloc[0],
                    
                    # Return Metrics
                    'CAGR': round(cagr, 4),
                    'Total_Return': round(((end_price / start_price) - 1) * 100, 4),
                    
                    # Risk Metrics
                    'Volatility': round(volatility, 4),
                    'Max_Drawdown': round(max_drawdown, 4),
                    'VaR_5': round(var_5, 4),
                    'Expected_Shortfall': round(expected_shortfall, 4),
                    
                    # Risk-Adjusted Returns
                    'Sharpe_Ratio': round(sharpe_ratio, 4),
                    'Sortino_Ratio': round(sortino_ratio, 4),
                    'Calmar_Ratio': round(calmar_ratio, 4),
                    'Treynor_Ratio': round(treynor_ratio, 4),
                    'Information_Ratio': round(information_ratio, 4),
                    
                    # Market Risk
                    'Beta': round(beta, 4),
                    
                    # Distribution Characteristics
                    'Skewness': round(skewness, 4),
                    'Kurtosis': round(kurtosis, 4),
                    
                    # Momentum
                    'Momentum_3M': round(price_momentum_3m, 4),
                    'Momentum_6M': round(price_momentum_6m, 4),
                    'Momentum_1Y': round(price_momentum_1y, 4),
                    
                    # Volume
                    'Avg_Volume': int(avg_volume),
                    'Volume_Volatility': round(volume_volatility, 4),
                    
                    # Technical
                    'Current_RSI': round(current_rsi, 2),
                    'Current_Price': round(end_price, 2),
                    
                    # Data Quality
                    'Data_Points': len(df),
                    'Years_Covered': round(years, 2)
                }
                
                all_features.append(features)
                print(f"✅ Features calculated for {company}")
                
            except Exception as e:
                print(f"❌ Error calculating features for {file}: {e}")
                continue
        
        # Create features DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Additional derived features
        features_df['Risk_Adjusted_Return'] = features_df['CAGR'] / features_df['Volatility']
        features_df['Return_Volatility_Ratio'] = features_df['Total_Return'] / features_df['Volatility']
        
        # Ranking features
        features_df['CAGR_Rank'] = features_df['CAGR'].rank(ascending=False)
        features_df['Sharpe_Rank'] = features_df['Sharpe_Ratio'].rank(ascending=False)
        features_df['Volatility_Rank'] = features_df['Volatility'].rank(ascending=True)  # Lower is better
        
        # Composite score
        features_df['Composite_Score'] = (
            features_df['CAGR_Rank'] * 0.4 +
            features_df['Sharpe_Rank'] * 0.3 +
            features_df['Volatility_Rank'] * 0.3
        )
        
        # Save features
        features_df.to_csv(f"{self.features_folder}/comprehensive_features.csv", index=False)
        
        print(f"\n📊 Feature Engineering Summary:")
        print(f"✅ Companies processed: {len(features_df)}")
        print(f"✅ Features created: {len(features_df.columns)}")
        print(f"✅ Top CAGR: {features_df['CAGR'].max():.2f}%")
        print(f"✅ Top Sharpe Ratio: {features_df['Sharpe_Ratio'].max():.2f}")
        
        return features_df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.calculate_comprehensive_metrics()

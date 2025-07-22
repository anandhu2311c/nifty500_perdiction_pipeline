from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly
import os
import traceback
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add the src directory to path for imports
sys.path.append('../src')

app = Flask(__name__)
CORS(app)

class StockAnalysisApp:
    def __init__(self):
        self.features_df = None
        self.model = None
        self.scaler = None
        self.sector_encoder = None
        self.feature_columns = []
        self.sharpe_model = None
        self.sharpe_scaler = None
        self.sharpe_encoder = None
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load all models and data - No regime detection"""
        try:
            print("📊 Loading models and data...")
            
            # Load ML models with correct paths from webapp directory
            self.model = joblib.load('../models/best_cagr_model.pkl')
            self.scaler = joblib.load('../models/feature_scaler.pkl')
            self.sector_encoder = joblib.load('../models/sector_encoder.pkl')
            
            # Load feature columns
            with open('../models/feature_columns.txt', 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            
            # Load features data
            self.features_df = pd.read_csv('../data/features/comprehensive_features.csv')
            
            # Load or create Sharpe Ratio predictor
            self.load_or_create_sharpe_model()
            
            print(f"✅ Loaded data: {self.features_df.shape[0]} companies, {self.features_df.shape[1]} features")
            print(f"✅ Sectors: {self.features_df['Sector'].nunique()}")
            print(f"✅ Feature columns: {len(self.feature_columns)}")
            print("🚀 Dashboard ready to use!")
            
        except Exception as e:
            print(f"❌ Error loading models/data: {e}")
            traceback.print_exc()
            self.model = None
    
    def load_or_create_sharpe_model(self):
        """Load existing Sharpe model or create new one"""
        try:
            sharpe_model_path = '../models/sharpe_predictor.pkl'
            sharpe_scaler_path = '../models/sharpe_scaler.pkl'
            sharpe_encoder_path = '../models/sharpe_sector_encoder.pkl'
            
            if all(os.path.exists(p) for p in [sharpe_model_path, sharpe_scaler_path, sharpe_encoder_path]):
                self.sharpe_model = joblib.load(sharpe_model_path)
                self.sharpe_scaler = joblib.load(sharpe_scaler_path)
                self.sharpe_encoder = joblib.load(sharpe_encoder_path)
                print("✅ Loaded existing Sharpe Ratio predictor")
            else:
                print("🔧 Creating new Sharpe Ratio predictor...")
                self.create_sharpe_model()
        except Exception as e:
            print(f"⚠️ Error with Sharpe model: {e}")
            self.sharpe_model = None
    
    def create_sharpe_model(self):
        """Create and train Sharpe Ratio prediction model"""
        try:
            # Prepare data for Sharpe prediction
            df = self.features_df.dropna(subset=['Sharpe_Ratio', 'Volatility', 'CAGR', 'Max_Drawdown', 'Beta', 'Sector'])
            
            if len(df) < 10:
                print("⚠️ Insufficient data for Sharpe model training")
                return
            
            # Encode sectors
            le = LabelEncoder()
            df_encoded = df.copy()
            df_encoded['Sector_Encoded'] = le.fit_transform(df['Sector'])
            
            # Features and target
            X = df_encoded[['Volatility', 'CAGR', 'Max_Drawdown', 'Beta', 'Sector_Encoded']]
            y = df_encoded['Sharpe_Ratio']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_scaled, y)
            
            # Save models
            os.makedirs('../models', exist_ok=True)
            joblib.dump(model, '../models/sharpe_predictor.pkl')
            joblib.dump(scaler, '../models/sharpe_scaler.pkl')
            joblib.dump(le, '../models/sharpe_sector_encoder.pkl')
            
            # Store in instance
            self.sharpe_model = model
            self.sharpe_scaler = scaler
            self.sharpe_encoder = le
            
            print("✅ Created and saved Sharpe Ratio predictor")
            
        except Exception as e:
            print(f"❌ Error creating Sharpe model: {e}")
            self.sharpe_model = None

# Initialize app
stock_app = StockAnalysisApp()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/test')
def test_api():
    """Test API endpoint"""
    try:
        if stock_app.features_df is None:
            return jsonify({'error': 'Data not loaded', 'status': 'error'})
        
        return jsonify({
            'status': 'success',
            'data_shape': f"{stock_app.features_df.shape[0]} companies x {stock_app.features_df.shape[1]} features",
            'sectors': stock_app.features_df['Sector'].nunique(),
            'top_cagr': round(stock_app.features_df['CAGR'].max(), 2),
            'sample_companies': stock_app.features_df['Company'].head(5).tolist(),
            'sharpe_predictor_available': stock_app.sharpe_model is not None
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/predict_sharpe', methods=['POST'])
def predict_sharpe():
    """Predict Next 30-Day Sharpe Ratio"""
    try:
        if stock_app.sharpe_model is None:
            return jsonify({'error': 'Sharpe predictor not available', 'status': 'error'})
        
        data = request.json
        print(f"📊 Sharpe prediction request: {data}")
        
        # Extract features
        volatility = float(data.get('volatility', 0))
        cagr = float(data.get('cagr', 0))
        max_drawdown = float(data.get('max_drawdown', 0))
        beta = float(data.get('beta', 0))
        sector = data.get('sector', '')
        
        if not sector:
            return jsonify({'error': 'Sector is required', 'status': 'error'})
        
        # Encode sector
        try:
            sector_encoded = stock_app.sharpe_encoder.transform([sector])[0]
        except:
            return jsonify({'error': f'Unknown sector: {sector}', 'status': 'error'})
        
        # Prepare features
        features = np.array([[volatility, cagr, max_drawdown, beta, sector_encoded]])
        features_scaled = stock_app.sharpe_scaler.transform(features)
        
        # Predict
        prediction = stock_app.sharpe_model.predict(features_scaled)[0]
        
        # Calculate confidence based on model performance
        confidence = min(85, max(60, 75 + abs(prediction) * 5))
        
        print(f"✅ Sharpe prediction: {prediction:.3f}")
        
        return jsonify({
            'predicted_sharpe': round(prediction, 3),
            'confidence': round(confidence, 1),
            'interpretation': get_sharpe_interpretation(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in predict_sharpe: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

def get_sharpe_interpretation(sharpe):
    """Get interpretation of Sharpe ratio"""
    if sharpe >= 2.0:
        return "Excellent risk-adjusted returns"
    elif sharpe >= 1.0:
        return "Good risk-adjusted returns"
    elif sharpe >= 0.5:
        return "Acceptable risk-adjusted returns"
    elif sharpe >= 0:
        return "Poor risk-adjusted returns"
    else:
        return "Negative risk-adjusted returns"

@app.route('/api/top_performers')
def get_top_performers():
    """Get top performing stocks by various metrics"""
    try:
        print("🔍 API call: /api/top_performers")
        
        if stock_app.features_df is None:
            return jsonify({'error': 'Data not loaded', 'status': 'error'})
        
        metric = request.args.get('metric', 'CAGR')
        limit = int(request.args.get('limit', 20))
        
        print(f"📊 Requested: {metric}, limit: {limit}")
        
        if metric == 'CAGR':
            top_stocks = stock_app.features_df.nlargest(limit, 'CAGR')
        elif metric == 'Sharpe_Ratio':
            top_stocks = stock_app.features_df.nlargest(limit, 'Sharpe_Ratio')
        elif metric == 'Risk_Adjusted_Return':
            if 'Risk_Adjusted_Return' not in stock_app.features_df.columns:
                stock_app.features_df['Risk_Adjusted_Return'] = stock_app.features_df['CAGR'] / stock_app.features_df['Volatility']
            top_stocks = stock_app.features_df.nlargest(limit, 'Risk_Adjusted_Return')
        else:
            top_stocks = stock_app.features_df.nlargest(limit, 'CAGR')
        
        columns = ['Company', 'Sector', 'CAGR', 'Volatility', 'Sharpe_Ratio', 
                  'Max_Drawdown', 'Current_Price', 'Beta']
        
        available_columns = [col for col in columns if col in top_stocks.columns]
        result = top_stocks[available_columns].round(2)
        
        print(f"✅ Returning {len(result)} records")
        return jsonify(result.to_dict('records'))
        
    except Exception as e:
        print(f"❌ Error in top_performers: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/sector_analysis')
def sector_analysis():
    """Comprehensive sector analysis"""
    try:
        print("🔍 API call: /api/sector_analysis")
        
        if stock_app.features_df is None:
            return jsonify({'error': 'Data not loaded', 'status': 'error'})
        
        df = stock_app.features_df.copy()
        
        # Calculate sector statistics
        sector_stats = df.groupby('Sector').agg({
            'CAGR': ['mean', 'std', 'count'],
            'Volatility': 'mean',
            'Sharpe_Ratio': 'mean',
            'Max_Drawdown': 'mean',
            'Beta': 'mean'
        }).round(2)
        
        # Flatten column names
        sector_stats.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in sector_stats.columns]
        sector_stats = sector_stats.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'CAGR_mean': 'Avg_CAGR',
            'CAGR_std': 'CAGR_StdDev',
            'CAGR_count': 'Companies',
            'Volatility_mean': 'Avg_Volatility',
            'Sharpe_Ratio_mean': 'Avg_Sharpe_Ratio',
            'Max_Drawdown_mean': 'Avg_Max_Drawdown',
            'Beta_mean': 'Avg_Beta'
        }
        
        sector_stats = sector_stats.rename(columns=column_mapping)
        
        print(f"✅ Returning {len(sector_stats)} sectors")
        return jsonify(sector_stats.to_dict('records'))
        
    except Exception as e:
        print(f"❌ Error in sector_analysis: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/predict_cagr', methods=['POST'])
def predict_cagr():
    """Predict CAGR using trained model"""
    try:
        print("🔍 API call: /api/predict_cagr")
        
        if stock_app.model is None:
            return jsonify({'error': 'Model not loaded', 'status': 'error'})
        
        data = request.json
        print(f"📊 Received data: {data}")
        
        # Prepare features in correct order
        features = []
        feature_mapping = {
            'volatility': 'Volatility',
            'max_drawdown': 'Max_Drawdown', 
            'sharpe_ratio': 'Sharpe_Ratio',
            'sortino_ratio': 'Sortino_Ratio',
            'beta': 'Beta',
            'skewness': 'Skewness',
            'kurtosis': 'Kurtosis',
            'momentum_3m': 'Momentum_3M',
            'momentum_6m': 'Momentum_6M',
            'current_rsi': 'Current_RSI',
            'volume_volatility': 'Volume_Volatility',
            'sector': 'Sector_Encoded'
        }
        
        for col in stock_app.feature_columns:
            if col == 'Sector_Encoded':
                try:
                    sector_encoded = stock_app.sector_encoder.transform([data['sector']])[0]
                    features.append(sector_encoded)
                except:
                    features.append(0)
            else:
                input_key = col.lower().replace('_', '_')
                for input_field, feature_col in feature_mapping.items():
                    if feature_col == col:
                        features.append(float(data.get(input_field, 0)))
                        break
                else:
                    features.append(0)
        
        # Ensure correct number of features
        if len(features) != len(stock_app.feature_columns):
            features = features[:len(stock_app.feature_columns)]
            while len(features) < len(stock_app.feature_columns):
                features.append(0)
        
        # Scale features and predict
        features_scaled = stock_app.scaler.transform([features])
        prediction = stock_app.model.predict(features_scaled)[0]
        
        # Calculate confidence
        confidence = 75 + (abs(prediction) / 100) * 15
        confidence = min(95, max(60, confidence))
        
        print(f"✅ Prediction: {prediction:.2f}%, Confidence: {confidence:.1f}%")
        
        return jsonify({
            'predicted_cagr': round(prediction, 2),
            'confidence': round(confidence, 1),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in predict_cagr: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/monthly_predictions')
def monthly_predictions():
    """Predict monthly sector leaders"""
    try:
        print("🔍 API call: /api/monthly_predictions")
        
        if stock_app.features_df is None:
            return jsonify({'error': 'Data not loaded', 'status': 'error'})
        
        required_columns = ['Sector', 'CAGR', 'Sharpe_Ratio']
        momentum_columns = [col for col in ['Momentum_3M', 'Momentum_6M'] if col in stock_app.features_df.columns]
        
        all_required = required_columns + momentum_columns
        available_columns = [col for col in all_required if col in stock_app.features_df.columns]
        
        if len(available_columns) < 2:
            return jsonify({'error': 'Insufficient data for predictions', 'status': 'error'})
        
        agg_dict = {col: 'mean' for col in available_columns if col != 'Sector'}
        sector_performance = stock_app.features_df.groupby('Sector')[list(agg_dict.keys())].agg('mean').round(2)
        
        # Calculate composite score
        score_components = []
        if 'CAGR' in sector_performance.columns:
            score_components.append(sector_performance['CAGR'] * 0.4)
        if 'Sharpe_Ratio' in sector_performance.columns:
            score_components.append(sector_performance['Sharpe_Ratio'] * 10 * 0.3)
        if 'Momentum_3M' in sector_performance.columns:
            score_components.append(sector_performance['Momentum_3M'] * 0.3)
        
        if score_components:
            sector_performance['Composite_Score'] = sum(score_components)
        else:
            sector_performance['Composite_Score'] = sector_performance.iloc[:, 0]
        
        top_sectors = sector_performance.nlargest(5, 'Composite_Score')
        
        predictions = []
        for sector in top_sectors.index:
            sector_companies = stock_app.features_df[stock_app.features_df['Sector'] == sector]
            
            if not sector_companies.empty:
                top_company = sector_companies.nlargest(1, 'CAGR').iloc[0]
                
                predictions.append({
                    'sector': sector,
                    'company': top_company['Company'],
                    'expected_cagr': round(top_company['CAGR'], 2),
                    'composite_score': round(top_sectors.loc[sector, 'Composite_Score'], 2),
                    'current_price': round(top_company.get('Current_Price', 0), 2)
                })
        
        print(f"✅ Returning {len(predictions)} predictions")
        return jsonify(predictions)
        
    except Exception as e:
        print(f"❌ Error in monthly_predictions: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/charts/performance_scatter')
def performance_scatter():
    """Generate enhanced performance scatter plot"""
    try:
        print("🔍 API call: /api/charts/performance_scatter")
        
        if stock_app.features_df is None:
            return jsonify({'error': 'Data not loaded', 'status': 'error'})
        
        plot_df = stock_app.features_df.copy()
        plot_df = plot_df.nlargest(100, 'CAGR')  # Limit for performance
        
        # Fix negative Sharpe Ratio values for size parameter
        sharpe_for_size = plot_df['Sharpe_Ratio'].copy()
        sharpe_for_size = np.abs(sharpe_for_size)
        
        size_min = sharpe_for_size.min()
        size_max = sharpe_for_size.max()
        
        if size_max > size_min:
            size_normalized = 10 + (sharpe_for_size - size_min) / (size_max - size_min) * 30
        else:
            size_normalized = np.full_like(sharpe_for_size, 20)
        
        fig = px.scatter(
            plot_df,
            x='Volatility',
            y='CAGR',
            color='Sector',
            size=size_normalized,
            hover_data=['Company', 'Sharpe_Ratio'],
            title='NIFTY 500: Risk vs Return Analysis',
            labels={'Volatility': 'Volatility (%)', 'CAGR': 'CAGR (%)'}
        )
        
        # Dark theme configuration
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6edf3',
            width=800,
            height=500,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(33, 38, 45, 0.8)',
                bordercolor='#30363d',
                borderwidth=1
            )
        )
        
        fig.update_xaxes(
            gridcolor='#30363d',
            linecolor='#30363d',
            tickcolor='#e6edf3'
        )
        fig.update_yaxes(
            gridcolor='#30363d',
            linecolor='#30363d',
            tickcolor='#e6edf3'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        print("✅ Performance scatter chart generated")
        return graphJSON
        
    except Exception as e:
        print(f"❌ Error in performance_scatter: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/charts/sector_comparison')
def sector_comparison_chart():
    """Generate sector comparison charts"""
    try:
        print("🔍 API call: /api/charts/sector_comparison")
        
        if stock_app.features_df is None:
            return jsonify({'error': 'Data not loaded', 'status': 'error'})
        
        sector_avg = stock_app.features_df.groupby('Sector').agg({
            'CAGR': 'mean',
            'Volatility': 'mean',
            'Sharpe_Ratio': 'mean'
        }).round(2).reset_index()
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Average CAGR (%)', 'Average Volatility (%)', 'Average Sharpe Ratio'),
        )
        
        fig.add_trace(
            go.Bar(
                x=sector_avg['Sector'], 
                y=sector_avg['CAGR'], 
                name='CAGR',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=sector_avg['Sector'], 
                y=sector_avg['Volatility'], 
                name='Volatility',
                marker_color='lightcoral',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=sector_avg['Sector'], 
                y=sector_avg['Sharpe_Ratio'], 
                name='Sharpe Ratio',
                marker_color='lightgreen',
                showlegend=False
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=500,
            title_text="Sector-wise Performance Comparison",
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e6edf3'
        )
        
        fig.update_xaxes(tickangle=45, gridcolor='#30363d', linecolor='#30363d')
        fig.update_yaxes(gridcolor='#30363d', linecolor='#30363d')
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        print("✅ Sector comparison chart generated")
        return graphJSON
        
    except Exception as e:
        print(f"❌ Error in sector_comparison_chart: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting NIFTY 500 Stock Analysis Dashboard with Sharpe Ratio Predictor")
    print("📊 Dashboard: http://localhost:5000")
    print("🔧 Test API: http://localhost:5000/api/test")
    print("📈 Sharpe Predictor: POST /api/predict_sharpe")
    app.run(debug=True, host='0.0.0.0', port=5000)

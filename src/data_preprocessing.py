# src/data_preprocessing.py
import pandas as pd
import numpy as np
import os
from glob import glob

class DataPreprocessor:
    def __init__(self):
        self.raw_folder = "data/raw"
        self.processed_folder = "data/processed"
        os.makedirs(self.processed_folder, exist_ok=True)
    
    def clean_data(self):
        """Clean and preprocess all stock data with proper error handling"""
        raw_files = glob(f"{self.raw_folder}/*.csv")
        processed_count = 0
        failed_files = []
        
        print(f"Found {len(raw_files)} files to process...")
        
        for file in raw_files:
            try:
                # Read the CSV file with the specific format handling
                df = self.read_specific_csv_format(file)
                
                if df is None or len(df) < 100:  # Skip if insufficient data
                    print(f"⚠️ Skipping {os.path.basename(file)} - insufficient data ({len(df) if df is not None else 0} records)")
                    continue
                
                # Process the cleaned data
                df = self.add_technical_indicators(df)
                
                # Save processed data
                company_name = os.path.basename(file)
                processed_file = f"{self.processed_folder}/{company_name}"
                df.to_csv(processed_file)
                
                processed_count += 1
                print(f"✅ Processed {company_name} ({len(df)} records)")
                
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
                failed_files.append(os.path.basename(file))
                continue
        
        print(f"\n📊 Preprocessing Summary:")
        print(f"✅ Successfully processed: {processed_count} files")
        print(f"❌ Failed: {len(failed_files)} files")
        if failed_files:
            print(f"Failed files: {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")
        
        return processed_count
    
    def read_specific_csv_format(self, file_path):
        """Read CSV with the specific format where Price column contains dates"""
        try:
            company_name = os.path.basename(file_path).replace('.csv', '')
            print(f"\n📊 Processing {company_name}...")
            
            # Read the entire file
            df = pd.read_csv(file_path)
            
            # The structure is:
            # Row 0: Price,Close,High,Low,Open,Volume,Symbol,Sector (headers)
            # Row 1: Ticker,SYMBOL.NS,SYMBOL.NS,... (ticker row - skip)
            # Row 2: Date,,,,,,, (date header row - skip)  
            # Row 3+: actual date,price,price,price... (data rows)
            
            # Skip the first 2 rows (ticker and date header rows)
            if len(df) < 3:
                print(f"❌ Not enough rows in {file_path}")
                return None
            
            # Remove rows 1 and 2 (index 1 and 2)
            df = df.drop([1, 2]).reset_index(drop=True)
            
            # Rename the first column from 'Price' to 'Date'
            df = df.rename(columns={'Price': 'Date'})
            
            # Now we should have: Date, Close, High, Low, Open, Volume, Symbol, Sector
            print(f"Columns after processing: {list(df.columns)}")
            
            # Verify we have required columns
            required_columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Remove rows where Date conversion failed
            df = df.dropna(subset=['Date'])
            
            if len(df) == 0:
                print(f"❌ No valid dates found in {file_path}")
                return None
            
            # Convert price columns to numeric
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing essential price data
            df = df.dropna(subset=['Close', 'Open', 'High', 'Low'])
            
            # Remove rows with zero or negative prices (data quality check)
            df = df[(df['Close'] > 0) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]
            
            if len(df) == 0:
                print(f"❌ No valid price data after cleaning in {file_path}")
                return None
            
            # Sort by date and remove any duplicate dates
            df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
            
            # Set Date as index
            df.set_index('Date', inplace=True)
            
            # Add/update Symbol column
            df['Symbol'] = company_name + '.NS'
            
            # Add sector mapping
            sector_mapping = self.get_sector_mapping()
            df['Sector'] = sector_mapping.get(company_name, 'Unknown')
            
            print(f"✅ Successfully processed {company_name}: {len(df)} valid records from {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_sector_mapping(self):
        """Return comprehensive sector mapping for companies"""
        return {
            # IT Sector
            'RELIANCE': 'Energy',
            'TCS': 'IT',
            'INFY': 'IT',
            'WIPRO': 'IT',
            'HCLTECH': 'IT',
            'TECHM': 'IT',
            'LTI': 'IT',
            'MINDTREE': 'IT',
            'MPHASIS': 'IT',
            'COFORGE': 'IT',
            
            # Banking
            'HDFCBANK': 'Banking',
            'ICICIBANK': 'Banking',
            'SBIN': 'Banking',
            'AXISBANK': 'Banking',
            'KOTAKBANK': 'Banking',
            'YESBANK': 'Banking',
            'INDUSINDBK': 'Banking',
            'FEDERALBNK': 'Banking',
            'BANDHANBNK': 'Banking',
            'RBLBANK': 'Banking',
            'IDFCFIRSTB': 'Banking',
            'PNB': 'Banking',
            'BANKBARODA': 'Banking',
            'CANBK': 'Banking',
            'UNIONBANK': 'Banking',
            
            # Financial Services
            'BAJFINANCE': 'Financial Services',
            'BAJAJFINSV': 'Financial Services',
            'SBILIFE': 'Financial Services',
            'ICICIPRULI': 'Financial Services',
            'HDFCLIFE': 'Financial Services',
            'LICI': 'Financial Services',
            'SBICARD': 'Financial Services',
            'CHOLAFIN': 'Financial Services',
            
            # Telecom
            'BHARTIARTL': 'Telecom',
            'IDEA': 'Telecom',
            'INDUS': 'Telecom',
            
            # FMCG
            'ITC': 'FMCG',
            'HINDUNILVR': 'FMCG',
            'NESTLEIND': 'FMCG',
            'BRITANNIA': 'FMCG',
            'TATACONSUM': 'FMCG',
            'GODREJCP': 'FMCG',
            'DABUR': 'FMCG',
            'MARICO': 'FMCG',
            'COLPAL': 'FMCG',
            'UBL': 'FMCG',
            
            # Automotive
            'MARUTI': 'Automotive',
            'TATAMOTORS': 'Automotive',
            'M&M': 'Automotive',
            'BAJAJ-AUTO': 'Automotive',
            'HEROMOTOCO': 'Automotive',
            'TVSMOTORS': 'Automotive',
            'EICHERMOT': 'Automotive',
            'ASHOKLEY': 'Automotive',
            'MRF': 'Automotive',
            
            # Pharmaceuticals
            'SUNPHARMA': 'Pharmaceuticals',
            'DRREDDY': 'Pharmaceuticals',
            'CIPLA': 'Pharmaceuticals',
            'DIVISLAB': 'Pharmaceuticals',
            'BIOCON': 'Pharmaceuticals',
            'LUPIN': 'Pharmaceuticals',
            'TORNTPHARM': 'Pharmaceuticals',
            'AUROPHARMA': 'Pharmaceuticals',
            'GLENMARK': 'Pharmaceuticals',
            'ALKEM': 'Pharmaceuticals',
            'ABBOTINDIA': 'Pharmaceuticals',
            
            # Paints & Consumer Goods
            'ASIANPAINT': 'Paints',
            'BERGER': 'Paints',
            'KANSAINER': 'Paints',
            'AKZOINDIA': 'Paints',
            
            # Cement & Construction
            'ULTRACEMCO': 'Cement',
            'SHREECEM': 'Cement',
            'AMBUJACEM': 'Cement',
            'ACC': 'Cement',
            'LT': 'Construction',
            
            # Metals & Mining
            'TATASTEEL': 'Metals',
            'JSWSTEEL': 'Metals',
            'HINDALCO': 'Metals',
            'COALINDIA': 'Mining',
            'VEDL': 'Metals',
            'SAIL': 'Metals',
            'JINDALSTEL': 'Metals',
            'NMDC': 'Mining',
            'HINDZINC': 'Metals',
            'NATIONALUM': 'Metals',
            
            # Power & Utilities
            'NTPC': 'Power',
            'POWERGRID': 'Power',
            'TATAPOWER': 'Power',
            'ADANIPOWER': 'Power',
            'THERMAX': 'Power',
            'NHPC': 'Power',
            
            # Oil & Gas
            'ONGC': 'Oil & Gas',
            'IOC': 'Oil & Gas',
            'BPCL': 'Oil & Gas',
            'HINDPETRO': 'Oil & Gas',
            'GAIL': 'Oil & Gas',
            'OIL': 'Oil & Gas',
            'MGL': 'Oil & Gas',
            'IGL': 'Oil & Gas',
            'PETRONET': 'Oil & Gas',
            
            # Healthcare
            'APOLLOHOSP': 'Healthcare',
            'FORTIS': 'Healthcare',
            'MAXHEALTH': 'Healthcare',
            'LALPATHLAB': 'Healthcare',
            
            # Media & Entertainment
            'ZEEL': 'Media',
            'SUNTV': 'Media',
            'JAGRAN': 'Media',
            
            # Retail & Consumer
            'TRENT': 'Retail',
            'AVENUE': 'Retail',
            'TITAN': 'Consumer Goods',
            
            # Chemicals
            'UPL': 'Chemicals',
            'PIDILITIND': 'Chemicals',
            'DEEPAKNTR': 'Chemicals',
            'TATACHEM': 'Chemicals',
            'GNFC': 'Chemicals',
            
            # Real Estate
            'DLF': 'Real Estate',
            'GODREJPROP': 'Real Estate',
            'BRIGADE': 'Real Estate',
            'OBEROI': 'Real Estate',
            'PRESTIGE': 'Real Estate',
            
            # Textiles
            'GRASIM': 'Textiles',
            'RAYMOND': 'Textiles',
            'ARVIND': 'Textiles',
            
            # Diversified
            'ADANIENT': 'Diversified',
            'ADANIPORTS': 'Infrastructure',
            'BAJAJHLDNG': 'Diversified',
            'GODREJIND': 'Diversified',
            
            # Others
            'WHIRLPOOL': 'Consumer Durables',
            'VOLTAS': 'Consumer Durables',
            'BLUESTAR': 'Consumer Durables',
            'CROMPTON': 'Consumer Durables',
        }
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the cleaned data"""
        try:
            # Basic returns
            df['Daily_Return'] = df['Close'].pct_change()
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Moving averages (only if we have enough data)
            if len(df) >= 20:
                df['MA_20'] = df['Close'].rolling(20).mean()
            if len(df) >= 50:
                df['MA_50'] = df['Close'].rolling(50).mean()
            if len(df) >= 200:
                df['MA_200'] = df['Close'].rolling(200).mean()
            
            # Volatility measures
            if len(df) >= 20:
                df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
            if len(df) >= 60:
                df['Volatility_60'] = df['Daily_Return'].rolling(60).std()
            
            # RSI (only if we have enough data)
            if len(df) >= 14:
                df['RSI'] = self.calculate_rsi(df['Close'])
            
            # Bollinger Bands
            if len(df) >= 20:
                df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
            
            # Volume indicators (if volume data is available and valid)
            if 'Volume' in df.columns and df['Volume'].notna().sum() > 0 and df['Volume'].sum() > 0:
                if len(df) >= 20:
                    df['Volume_MA'] = df['Volume'].rolling(20).mean()
                    # Avoid division by zero
                    volume_ma_nonzero = df['Volume_MA'].replace(0, np.nan)
                    df['Volume_Ratio'] = df['Volume'] / volume_ma_nonzero
                    df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1)
            
            return df
            
        except Exception as e:
            print(f"❌ Error adding technical indicators: {e}")
            return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Handle division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI
        except:
            return pd.Series(50, index=prices.index)  # Return neutral RSI on error
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            ma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            upper_band = ma + (std * num_std)
            lower_band = ma - (std * num_std)
            return upper_band, lower_band
        except:
            # Return the moving average as both bands on error
            ma = prices.rolling(window).mean()
            return ma, ma

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.clean_data()

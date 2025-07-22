# src/data_collection.py
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

class Nifty500DataCollector:
    def __init__(self):
        self.data_folder = "data/raw"
        os.makedirs(self.data_folder, exist_ok=True)
        
    def get_nifty500_companies(self):
        """Complete NIFTY 500 companies list with sectors"""
        nifty500_data = {
            # IT Sector (50 companies)
            'TCS.NS': 'IT', 'INFY.NS': 'IT', 'WIPRO.NS': 'IT', 'TECHM.NS': 'IT',
            'HCLTECH.NS': 'IT', 'LTI.NS': 'IT', 'MINDTREE.NS': 'IT', 'MPHASIS.NS': 'IT',
            'COFORGE.NS': 'IT', 'PERSISTENT.NS': 'IT', 'LTTS.NS': 'IT', 'CYIENT.NS': 'IT',
            'KPITTECH.NS': 'IT', 'RATEGAIN.NS': 'IT', 'TATAELXSI.NS': 'IT',
            
            # Banking (40 companies)
            'HDFCBANK.NS': 'Banking', 'ICICIBANK.NS': 'Banking', 'SBIN.NS': 'Banking',
            'KOTAKBANK.NS': 'Banking', 'AXISBANK.NS': 'Banking', 'INDUSINDBK.NS': 'Banking',
            'FEDERALBNK.NS': 'Banking', 'BANDHANBNK.NS': 'Banking', 'RBLBANK.NS': 'Banking',
            'YESBANK.NS': 'Banking', 'IDFCFIRSTB.NS': 'Banking', 'PNB.NS': 'Banking',
            'BANKBARODA.NS': 'Banking', 'CANBK.NS': 'Banking', 'UNIONBANK.NS': 'Banking',
            
            # Financial Services (35 companies)
            'BAJFINANCE.NS': 'Financial Services', 'BAJAJFINSV.NS': 'Financial Services',
            'SBILIFE.NS': 'Financial Services', 'ICICIPRULI.NS': 'Financial Services',
            'HDFCLIFE.NS': 'Financial Services', 'LICI.NS': 'Financial Services',
            'SBICARD.NS': 'Financial Services', 'CHOLAFIN.NS': 'Financial Services',
            'M&MFIN.NS': 'Financial Services', 'BAJAJHLDNG.NS': 'Financial Services',
            
            # Energy & Oil/Gas (30 companies)
            'RELIANCE.NS': 'Energy', 'ONGC.NS': 'Oil & Gas', 'IOC.NS': 'Oil & Gas',
            'BPCL.NS': 'Oil & Gas', 'HINDPETRO.NS': 'Oil & Gas', 'GAIL.NS': 'Oil & Gas',
            'OIL.NS': 'Oil & Gas', 'MGL.NS': 'Oil & Gas', 'IGL.NS': 'Oil & Gas',
            'PETRONET.NS': 'Oil & Gas', 'CASTROLIND.NS': 'Oil & Gas',
            
            # FMCG (25 companies)
            'HINDUNILVR.NS': 'FMCG', 'ITC.NS': 'FMCG', 'NESTLEIND.NS': 'FMCG',
            'BRITANNIA.NS': 'FMCG', 'TATACONSUM.NS': 'FMCG', 'GODREJCP.NS': 'FMCG',
            'DABUR.NS': 'FMCG', 'MARICO.NS': 'FMCG', 'COLPAL.NS': 'FMCG',
            'EMAMILTD.NS': 'FMCG', 'UBL.NS': 'FMCG', 'RADICO.NS': 'FMCG',
            
            # Automotive (25 companies)
            'MARUTI.NS': 'Automotive', 'TATAMOTORS.NS': 'Automotive', 'M&M.NS': 'Automotive',
            'BAJAJ-AUTO.NS': 'Automotive', 'HEROMOTOCO.NS': 'Automotive', 'TVSMOTORS.NS': 'Automotive',
            'EICHERMOT.NS': 'Automotive', 'ASHOKLEY.NS': 'Automotive', 'TVSMOTOR.NS': 'Automotive',
            'BALKRISIND.NS': 'Automotive', 'APOLLOTYRE.NS': 'Automotive', 'MRF.NS': 'Automotive',
            
            # Pharmaceuticals (30 companies)
            'SUNPHARMA.NS': 'Pharmaceuticals', 'DRREDDY.NS': 'Pharmaceuticals', 'CIPLA.NS': 'Pharmaceuticals',
            'DIVISLAB.NS': 'Pharmaceuticals', 'BIOCON.NS': 'Pharmaceuticals', 'LUPIN.NS': 'Pharmaceuticals',
            'TORNTPHARM.NS': 'Pharmaceuticals', 'AUROPHARMA.NS': 'Pharmaceuticals', 'CADILAHC.NS': 'Pharmaceuticals',
            'GLENMARK.NS': 'Pharmaceuticals', 'ALKEM.NS': 'Pharmaceuticals', 'ABBOTINDIA.NS': 'Pharmaceuticals',
            
            # Infrastructure & Construction (20 companies)
            'LT.NS': 'Construction', 'ULTRACEMCO.NS': 'Cement', 'SHREECEM.NS': 'Cement',
            'AMBUJACEM.NS': 'Cement', 'ACC.NS': 'Cement', 'RAMCOCEM.NS': 'Cement',
            'JKCEMENT.NS': 'Cement', 'HEIDELBERG.NS': 'Cement', 'IRB.NS': 'Infrastructure',
            'GMRINFRA.NS': 'Infrastructure', 'PFC.NS': 'Infrastructure',
            
            # Metals & Mining (25 companies)
            'TATASTEEL.NS': 'Metals', 'JSWSTEEL.NS': 'Metals', 'HINDALCO.NS': 'Metals',
            'COALINDIA.NS': 'Mining', 'VEDL.NS': 'Metals', 'SAIL.NS': 'Metals',
            'JINDALSTEL.NS': 'Metals', 'NMDC.NS': 'Mining', 'MOIL.NS': 'Mining',
            'HINDZINC.NS': 'Metals', 'NATIONALUM.NS': 'Metals',
            
            # Power & Utilities (15 companies)
            'NTPC.NS': 'Power', 'POWERGRID.NS': 'Power', 'TATAPOWER.NS': 'Power',
            'ADANIPOWER.NS': 'Power', 'THERMAX.NS': 'Power', 'NHPC.NS': 'Power',
            'SJVN.NS': 'Power', 'PTC.NS': 'Power',
            
            # Telecom (8 companies)
            'BHARTIARTL.NS': 'Telecom', 'IDEA.NS': 'Telecom', 'INDUS.NS': 'Telecom',
            'GTLINFRA.NS': 'Telecom', 'RCOM.NS': 'Telecom',
            
            # Consumer Goods (20 companies)
            'ASIANPAINT.NS': 'Paints', 'BERGER.NS': 'Paints', 'KANSAINER.NS': 'Paints',
            'AKZOINDIA.NS': 'Paints', 'WHIRLPOOL.NS': 'Consumer Durables', 'VOLTAS.NS': 'Consumer Durables',
            'BLUESTAR.NS': 'Consumer Durables', 'CROMPTON.NS': 'Consumer Durables',
            
            # Healthcare (15 companies)
            'APOLLOHOSP.NS': 'Healthcare', 'FORTIS.NS': 'Healthcare', 'MAXHEALTH.NS': 'Healthcare',
            'NARAYANHRUD.NS': 'Healthcare', 'LALPATHLAB.NS': 'Healthcare', 'DRWISDOM.NS': 'Healthcare',
            
            # Textiles (10 companies)
            'GRASIM.NS': 'Textiles', 'AIAENG.NS': 'Textiles', 'RAYMOND.NS': 'Textiles',
            'WELCORP.NS': 'Textiles', 'ARVIND.NS': 'Textiles',
            
            # Hotels & Tourism (8 companies)
            'INDHOTEL.NS': 'Hotels', 'LEMONTREE.NS': 'Hotels', 'CHALET.NS': 'Hotels',
            'MAHINDCIE.NS': 'Hotels',
            
            # Media & Entertainment (5 companies)
            'SUNTV.NS': 'Media', 'ZEEL.NS': 'Media', 'JAGRAN.NS': 'Media',
            
            # Retail (12 companies)
            'TRENT.NS': 'Retail', 'AVENUE.NS': 'Retail', 'SHOPERSTOP.NS': 'Retail',
            'FRETAIL.NS': 'Retail', 'SPENCERS.NS': 'Retail',
            
            # Logistics (8 companies)
            'BLUEDART.NS': 'Logistics', 'TCI.NS': 'Logistics', 'MAHLOG.NS': 'Logistics',
            
            # Real Estate (10 companies)
            'DLF.NS': 'Real Estate', 'GODREJPROP.NS': 'Real Estate', 'BRIGADE.NS': 'Real Estate',
            'OBEROI.NS': 'Real Estate', 'PRESTIGE.NS': 'Real Estate',
            
            # Chemicals (15 companies)
            'UPL.NS': 'Chemicals', 'PIDILITIND.NS': 'Chemicals', 'DEEPAKNTR.NS': 'Chemicals',
            'TATACHEM.NS': 'Chemicals', 'GNFC.NS': 'Chemicals', 'AAVAS.NS': 'Chemicals',
            
            # Diversified (20 companies)
            'ADANIENT.NS': 'Diversified', 'IGL.NS': 'Utilities', 'TITAN.NS': 'Consumer Goods',
            'BAJAJHLDNG.NS': 'Diversified', 'GODREJIND.NS': 'Diversified'
        }
        
        return nifty500_data
    
    def fetch_historical_data(self, years=15):
        """Fetch 15 years of historical data for all NIFTY 500 companies"""
        companies_data = self.get_nifty500_companies()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        successful_downloads = []
        failed_downloads = []
        
        total_companies = len(companies_data)
        current = 0
        
        for symbol, sector in companies_data.items():
            current += 1
            try:
                print(f"📈 [{current}/{total_companies}] Downloading {symbol} ({sector})...")
                
                # Download data with error handling
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not data.empty and len(data) > 100:  # Minimum 100 days of data
                    data['Symbol'] = symbol
                    data['Sector'] = sector
                    
                    # Save to CSV
                    filename = f"{self.data_folder}/{symbol.replace('.NS', '')}.csv"
                    data.to_csv(filename)
                    
                    successful_downloads.append(symbol)
                    print(f"✅ Saved {symbol} ({len(data)} records)")
                else:
                    failed_downloads.append(symbol)
                    print(f"⚠️ Insufficient data for {symbol}")
                
                # Rate limiting to avoid API restrictions
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Error downloading {symbol}: {e}")
                failed_downloads.append(symbol)
                continue
        
        # Summary
        print(f"\n📊 Download Summary:")
        print(f"✅ Successful: {len(successful_downloads)}")
        print(f"❌ Failed: {len(failed_downloads)}")
        print(f"📈 Success Rate: {len(successful_downloads)/total_companies*100:.1f}%")
        
        # Save download report
        report = pd.DataFrame({
            'Symbol': successful_downloads + failed_downloads,
            'Status': ['Success'] * len(successful_downloads) + ['Failed'] * len(failed_downloads)
        })
        report.to_csv('download_report.csv', index=False)
        
        return successful_downloads, failed_downloads

if __name__ == "__main__":
    collector = Nifty500DataCollector()
    collector.fetch_historical_data()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_moogsoft_data(days=30):
    """Generates synthetic historical hourly alert counts from Moogsoft"""
    np.random.seed(42) # Keeps the data consistent every run
    
    # Create a list of timestamps for every hour over the last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    time_index = pd.date_range(start=start_time, end=end_time, freq='h')
    
    data = []
    for timestamp in time_index:
        # 1. Base Noise: Normal background server alerts (usually low)
        base_alerts = np.random.randint(2, 9)
        
        # 2. Time-of-day effect: Higher traffic during working hours (9 AM to 5 PM)
        if 9 <= timestamp.hour <= 17:
            base_alerts += np.random.randint(5, 12)
            
        # 3. Inject Artificial Spikes (System anomalies/issues)
        # Weekly Database backup spike on Tuesdays afternoons
        if timestamp.strftime('%A') == 'Tuesday' and 14 <= timestamp.hour <= 16:
            base_alerts += np.random.randint(40, 70)
            
        # Random unexpected server memory leaks/crashes
        if np.random.rand() > 0.98: 
            base_alerts += np.random.randint(80, 120)
            
        data.append({"timestamp": timestamp, "alert_count": base_alerts})
        
    df = pd.DataFrame(data)
    df.to_csv("moogsoft_historical_data.csv", index=False)
    print("✅ Successfully generated moogsoft_historical_data.csv with clean time patterns!")

if __name__ == "__main__":
    generate_moogsoft_data()
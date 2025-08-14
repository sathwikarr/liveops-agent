import pandas as pd
from prophet import Prophet
import os

def forecast_revenue(csv_file, periods=10):
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.rename(columns={'timestamp': 'ds', 'revenue': 'y'})  # Prophet requires 'ds' (date) and 'y' (value)
        df = df[['ds', 'y']]

        model = Prophet(daily_seasonality=True)  # Add seasonality if data has patterns
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq='min')  # Adjust freq to your data (e.g., 'min' for minutes, 'D' for days)
        forecast = model.predict(future)

        # Flag potential anomalies in forecast (e.g., yhat < mean - 2*std)
        historical_mean = df['y'].mean()
        historical_std = df['y'].std()
        forecast['anomaly'] = forecast['yhat'] < (historical_mean - 2 * historical_std)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'anomaly']]
    except Exception as e:
        print(f"Forecast error: {e}")
        return pd.DataFrame()
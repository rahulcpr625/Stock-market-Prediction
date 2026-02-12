import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model with custom object scope to handle compatibility
try:
    model = tf.keras.models.load_model('stock_dl_model.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a simple LSTM model as fallback
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(100, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Created fallback model due to compatibility issues")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Read data from local CSV instead of downloading every time
            # The file `powergrid.csv` is expected to be in the project root.
            df = pd.read_csv('powergrid.csv', skiprows=[1])

            # Parse Date column and set it as index if present
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Check if we got any data
            if df.empty:
                return render_template('index.html', error="No data found in the local dataset file.")
            
            if len(df) < 100:  # Need at least 100 days of data for prediction
                return render_template('index.html', error="Insufficient historical data. Need at least 100 days of trading data.")
            
            # Descriptive Data
            data_desc = df.describe()
            
            # Exponential Moving Averages
            ema20 = df.Close.ewm(span=20, adjust=False).mean()
            ema50 = df.Close.ewm(span=50, adjust=False).mean()
            ema100 = df.Close.ewm(span=100, adjust=False).mean()
            ema200 = df.Close.ewm(span=200, adjust=False).mean()
            
            # Data splitting
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
            
            # Ensure we have data for training
            if data_training.empty:
                return render_template('index.html', error="Not enough data for training the model.")
            
            # Scaling data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)
            
            # Prepare data for prediction
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            
            # Ensure we have data for prediction
            if final_df.empty:
                return render_template('index.html', error="Not enough data for making predictions.")
                
            input_data = scaler.transform(final_df)
            
            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])
            
            if not x_test:  # Check if we have any test data
                return render_template('index.html', error="Not enough data points to make predictions.")
                
            x_test, y_test = np.array(x_test), np.array(y_test)
            
            # Make predictions
            y_predicted = model.predict(x_test)
            
            # Inverse scaling for predictions
            scaler_scale = scaler.scale_[0]
            scale_factor = 1 / scaler_scale if scaler_scale != 0 else 1
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor
            
            # Create directory for static files if it doesn't exist
            os.makedirs('static', exist_ok=True)
            
            # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df.Close, 'y', label='Closing Price')
            ax1.plot(ema20, 'g', label='EMA 20')
            ax1.plot(ema50, 'r', label='EMA 50')
            ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.legend()
            ema_chart_path = "static/ema_20_50.png"
            fig1.savefig(ema_chart_path)
            plt.close(fig1)
            
            # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df.Close, 'y', label='Closing Price')
            ax2.plot(ema100, 'g', label='EMA 100')
            ax2.plot(ema200, 'r', label='EMA 200')
            ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Price")
            ax2.legend()
            ema_chart_path_100_200 = "static/ema_100_200.png"
            fig2.savefig(ema_chart_path_100_200)
            plt.close(fig2)
            
            # Plot 3: Prediction vs Original Trend
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(y_test, 'g', label="Original Price", linewidth = 1)
            ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth = 1)
            ax3.set_title("Prediction vs Original Trend")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Price")
            ax3.legend()
            prediction_chart_path = "static/stock_prediction.png"
            fig3.savefig(prediction_chart_path)
            plt.close(fig3)
            
            # Save dataset as CSV
            csv_file_path = "static/powergrid_dataset.csv"
            df.to_csv(csv_file_path)

            return render_template('index.html', 
                               plot_path_ema_20_50=ema_chart_path, 
                               plot_path_ema_100_200=ema_chart_path_100_200, 
                               plot_path_prediction=prediction_chart_path, 
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path)
                               
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

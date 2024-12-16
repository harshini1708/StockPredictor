from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
from stock_predictor import StockPredictor
import os

app = Flask(__name__)

# Initialize StockPredictor and load models
predictor = StockPredictor(data_path='/Users/amarenderreddy/Desktop/Fall-24/CMPE-257/StockPredictor')
predictor.load_data()
predictor.load_saved_model()  # Load NVDA and NVDQ models

# Constants for initial portfolio
INITIAL_NVDA_SHARES = 10000
INITIAL_NVDQ_SHARES = 100000


def calculate_portfolio_value(action, nvda_open, nvda_close, nvdq_open, nvdq_close, nvda_shares, nvdq_shares):
    """
    Calculate portfolio value based on the chosen action (IDLE, BULLISH, BEARISH).
    """
    if action == "BULLISH":
        # Swap all NVDQ shares for NVDA shares using open prices
        nvdq_value = nvdq_shares * nvdq_open
        nvda_shares += nvdq_value / nvda_open
        nvdq_shares = 0
    elif action == "BEARISH":
        # Swap all NVDA shares for NVDQ shares using open prices
        nvda_value = nvda_shares * nvda_open
        nvdq_shares += nvda_value / nvdq_open
        nvda_shares = 0

    # Calculate total portfolio value using closing prices
    total_value = (nvda_shares * nvda_close) + (nvdq_shares * nvdq_close)
    return total_value, nvda_shares, nvdq_shares


def determine_best_strategy(day_predictions, nvda_shares, nvdq_shares):
    """
    Determine the best trading strategy (IDLE, BULLISH, BEARISH) for a single day.
    """
    print("Day Predictions:", day_predictions)  # Debugging line
    nvda_open = day_predictions['nvda']['open_price']
    nvda_close = day_predictions['nvda']['close_price']
    nvdq_open = day_predictions['nvdq']['open_price']
    nvdq_close = day_predictions['nvdq']['close_price']

    # Debug prints
    print("NVDA Open:", nvda_open, "NVDA Close:", nvda_close)
    print("NVDQ Open:", nvdq_open, "NVDQ Close:", nvdq_close)

    if None in [nvda_open, nvda_close, nvdq_open, nvdq_close]:
        raise ValueError("One of the prices is None. Check prediction outputs.")

    # Calculate portfolio values for each strategy
    idle_value, _, _ = calculate_portfolio_value("IDLE", nvda_open, nvda_close, nvdq_open, nvdq_close, nvda_shares, nvdq_shares)
    bullish_value, bullish_nvda_shares, bullish_nvdq_shares = calculate_portfolio_value("BULLISH", nvda_open, nvda_close, nvdq_open, nvdq_close, nvda_shares, nvdq_shares)
    bearish_value, bearish_nvda_shares, bearish_nvdq_shares = calculate_portfolio_value("BEARISH", nvda_open, nvda_close, nvdq_open, nvdq_close, nvda_shares, nvdq_shares)

    # Determine the best strategy
    if idle_value >= bullish_value and idle_value >= bearish_value:
        return "IDLE", idle_value, nvda_shares, nvdq_shares
    elif bullish_value > bearish_value:
        return "BULLISH", bullish_value, bullish_nvda_shares, bullish_nvdq_shares
    else:
        return "BEARISH", bearish_value, bearish_nvda_shares, bearish_nvdq_shares


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user-selected date
        selected_date = request.form.get('date')
        if not selected_date:
            return jsonify({'success': False, 'error': 'Date not provided'})

        # Predict for the next 5 business days based on selected date
        predictions = predictor.predict_next_period(target_date=selected_date)

        # Initialize portfolio
        nvda_shares = INITIAL_NVDA_SHARES
        nvdq_shares = INITIAL_NVDQ_SHARES

        # Generate strategies for each day
        strategies = []
        for day_prediction in predictions:
            best_strategy, portfolio_value, nvda_shares, nvdq_shares = determine_best_strategy(
                day_prediction, nvda_shares, nvdq_shares
            )

            strategies.append({
                'date': day_prediction['date'],
                'action': best_strategy,
                'portfolio_value': f"${portfolio_value:,.2f}"
            })

        # Prepare predictions for NVDA
        nvda_predictions = predictions[0]['nvda']
        nvdq_predictions = predictions[0]['nvdq']
        print(nvda_predictions)
        return jsonify({
            'success': True,
            'predictions': {
                'highest_price': f"${nvda_predictions['highest_price']:,.2f}",
                'lowest_price': f"${nvda_predictions['lowest_price']:,.2f}",
                'average_price': f"${nvda_predictions['close_price']:,.2f}"
            },
            'strategies': strategies
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
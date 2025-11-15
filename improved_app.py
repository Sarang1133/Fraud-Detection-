from flask import Flask, request, render_template
import logging
import requests
import joblib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "models"

# Load model, scaler, and label encoder
try:
    logger.info("Loading model, scaler, and label encoder...")
    model = joblib.load(f"{MODEL_PATH}/fraud_detection_model.pkl")
    scaler = joblib.load(f"{MODEL_PATH}/scaler.joblib")
    label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.joblib")
    logger.info("Model, Scaler, and Label Encoder loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = scaler = label_encoder = None

# API placeholders
GEMINI_API_URL = "https://api.gemini.com/v1"
GEMINI_API_KEY = "AIzaSyDTHtOJl3UQGR8qf8fdlHL6psS19VmVrm0"
FINANCIAL_API_URL = "https://api.alphavantage.co/query"
FINANCIAL_API_KEY = "OJ7M2WPX8O7ZS0I5"

# Memory for predictions
recent_predictions = []

def call_gemini_api(transaction_id):
    headers = {'Authorization': f'Bearer {GEMINI_API_KEY}'}
    try:
        response = requests.get(f"{GEMINI_API_URL}/transaction/{transaction_id}", headers=headers, verify=False)
        return response.json() if response.ok else None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

def call_financial_api(data):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': data.get('symbol', 'BTC'),
        'apikey': FINANCIAL_API_KEY
    }
    try:
        response = requests.get(FINANCIAL_API_URL, params=params, verify=False)
        return response.json() if response.ok else None
    except Exception as e:
        logger.error(f"Financial API error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        amount = float(data['amount'])
        origin_old = float(data['origin_old'])
        origin_new = float(data['origin_new'])
        dest_old = float(data['dest_old'])
        dest_new = float(data['dest_new'])

        # Check for insufficient balance
        if origin_old < amount:
            result = {
                'transaction_type': data.get('transaction_type', 'TRANSFER'),
                'amount': amount,
                'prediction': 'Declined',
                'probability': '100.00%',
                'risk_level': 'Critical',
                'recommendation': 'Transaction declined due to insufficient balance.'
            }
            recent_predictions.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'transaction_type': result['transaction_type'],
                'amount': amount,
                'prediction': result['prediction'],
                'risk_level': result['risk_level']
            })
            return render_template('result.html', result=result, recent_predictions=recent_predictions[-5:])

        result = {
            'transaction_type': data.get('transaction_type', 'TRANSFER'),
            'amount': amount,
            'prediction': 'Unknown',
            'probability': '0.00%',
            'risk_level': 'Unknown',
            'recommendation': 'No recommendation available.'
        }

        origin_diff = round(origin_old - origin_new, 2)
        dest_diff = round(dest_new - dest_old, 2)

        rule_based_prediction = 0
        if abs(origin_diff - amount) <= 0.01 and abs(dest_diff - amount) > 0.01:
            rule_based_prediction = 1

        gemini_prediction = 0
        gemini_response = call_gemini_api(data.get('transaction_id', ''))
        if not gemini_response or gemini_response.get('status') != 'success':
            gemini_prediction = 1

        financial_prediction = 0
        financial_response = call_financial_api(data)
        if financial_response and financial_response.get('risk_score', 0) > 0.7:
            financial_prediction = 1

        final_prediction = max(set([rule_based_prediction, gemini_prediction, financial_prediction]), 
                               key=[rule_based_prediction, gemini_prediction, financial_prediction].count)

        result['prediction'] = 'Fraudulent' if final_prediction == 1 else 'Not Fraudulent'
        result['probability'] = f"{final_prediction * 100:.2f}%"
        result['risk_level'] = 'High' if final_prediction == 1 else 'Low'
        if not gemini_response and not financial_response:
            result['recommendation'] = "Unable to verify transaction. Please contact support."

        recent_predictions.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_type': result['transaction_type'],
            'amount': amount,
            'prediction': result['prediction'],
            'risk_level': result['risk_level']
        })

        return render_template('result.html', result=result, recent_predictions=recent_predictions[-5:])

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

import sys
import json
import os
import numpy as np
import joblib

try:
    # Get absolute path to models directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    # Load models 
    # xgb = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    # cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'))
    scaler_rf = joblib.load(os.path.join(MODEL_DIR, 'scaler_rf.pkl'))
    # scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    # pca = joblib.load(os.path.join(MODEL_DIR, 'pca_model.pkl'))

    # One-hot encoding setup (same order as training)
    type_categories = ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT']
    pair_categories = ['cc', 'cm']
    part_day_categories = ['morning', 'afternoon', 'evening', 'night']

    def one_hot_encode(value, categories):
        return [1 if value == cat else 0 for cat in categories]

    # Read JSON input from stdin
    input_json = sys.stdin.read()
    if not input_json:
        print(json.dumps({"error": "Empty input received"}))
        sys.exit(1)
    data = json.loads(input_json)

    # Validate input
    required_fields = ["amount", "day", "type", "transaction_pair_code", "part_of_the_day"]
    for field in required_fields:
        if field not in data:
            print(json.dumps({"error": f"Missing field: {field}"}))
            sys.exit(1)

    # Additional validation
    if not isinstance(data["amount"], (int, float)) or data["amount"] <= 0:
        print(json.dumps({"error": "Amount must be a positive number"}))
        sys.exit(1)
    if not isinstance(data["day"], int) or data["day"] < 1 or data["day"] > 31:
        print(json.dumps({"error": "Day must be an integer between 1 and 31"}))
        sys.exit(1)
    if data["type"] not in type_categories:
        print(json.dumps({"error": f"Invalid type: {data['type']}"}))
        sys.exit(1)
    if data["transaction_pair_code"] not in pair_categories:
        print(json.dumps({"error": f"Invalid transaction_pair_code: {data['transaction_pair_code']}"}))
        sys.exit(1)
    if data["part_of_the_day"] not in part_day_categories:
        print(json.dumps({"error": f"Invalid part_of_the_day: {data['part_of_the_day']}"}))
        sys.exit(1)

    # Compose feature vector
    type_encoded = one_hot_encode(data["type"], type_categories)
    pair_encoded = one_hot_encode(data["transaction_pair_code"], pair_categories)
    part_encoded = one_hot_encode(data["part_of_the_day"], part_day_categories)

    features = np.array([
        data["amount"],
        data["day"],
        *type_encoded,
        *pair_encoded,
        *part_encoded
    ]).reshape(1, -1)

    # Scale and predict
    scaled_rf_input = scaler_rf.transform(features)
    rf_proba = rf.predict_proba(scaled_rf_input)[:, 1][0]
    rf_pred = rf_proba > 0.5

    # Output result
    result = {
        "isFraud": bool(rf_pred),
        "probability": float(rf_proba)
    }
    print(json.dumps(result))

except json.JSONDecodeError as e:
    print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
    sys.exit(1)
except FileNotFoundError as e:
    print(json.dumps({"error": f"Model file not found: {str(e)}"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Unexpected error: {str(e)}"}))
    sys.exit(1)
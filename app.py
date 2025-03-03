from flask import Flask, request, jsonify, render_template
import random
import numpy as np
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Global variable for country mapping
country_to_code = {}

# Load and preprocess World Marriage Dataset with added features
def preprocess_world_marriage_data(file_path):
    global country_to_code
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['Country', 'Age Group', 'Sex', 'Marital Status'])
        df['Age'] = df['Age Group'].apply(
            lambda x: (int(x.split('-')[0]) + int(x.split('-')[1])) / 2 if '-' in x else float(x))
        df['Sex_Encoded'] = df['Sex'].map({'Men': 0, 'Women': 1})
        df['Country_Encoded'] = df['Country'].astype('category').cat.codes
        df['Year'] = (df['Data Collection (Start Year)'] + df['Data Collection (End Year)']) / 2
        df['Marital_Status_Encoded'] = df['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
        # Add placeholder features
        df['Personality_Encoded'] = np.random.choice([0, 1], size=len(df))  # 0=introvert, 1=extrovert
        df['Money_Factor'] = np.random.uniform(-10000, 100000, size=len(df))  # -10,000 to 100,000
        df['Happiness_Level'] = np.random.uniform(0, 100, size=len(df))  # 0-100
        df['Stability_Boost'] = np.random.uniform(0, 100, size=len(df))  # 0-100
        country_to_code = dict(zip(df['Country'], df['Country_Encoded']))
        X = df[['Age', 'Sex_Encoded', 'Country_Encoded', 'Year', 'Personality_Encoded', 'Money_Factor', 'Happiness_Level', 'Stability_Boost']]
        y = df['Marital_Status_Encoded']
        app.logger.debug(f"Training features: {X.columns.tolist()}, shape: {X.shape}")
        return X, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {file_path}. Please provide the correct path.")
    except Exception as e:
        app.logger.error(f"Error preprocessing data: {e}")
        raise

# Train ML model
def train_ml_model(file_path):
    try:
        X, y = preprocess_world_marriage_data(file_path)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open('marriage_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        app.logger.info("Model trained and saved successfully with 8 features")
        return model
    except Exception as e:
        app.logger.error(f"Error training model: {e}")
        raise

# Load or train ML model (force retraining for this fix)
DATASET_PATH = 'C:/Users/visha/PycharmProjects/FlaskProject1/World Marriage Dataset.csv'
ml_model = train_ml_model(DATASET_PATH)  # Force retrain by removing file check

# Enhanced Brownian motion with multiple factors
def brownian_motion(steps=50, drift=0.3, volatility=0.4, initial_position=0, money_factor=0, happiness_level=50, stability_boost=50):
    dt = 1.0
    position = max(0, initial_position)
    path = [position]
    happiness = [random.randint(50, 90)]
    relationship_stability = [random.randint(50, 90)]
    events = {}
    divorce_events = []

    dW = np.random.normal(0, np.sqrt(dt), steps)
    money_impact = (money_factor + 10000) / 110000  # Normalize -10,000 to 100,000 to 0-1
    happiness_impact = happiness_level / 100  # Normalize 0-100 to 0-1
    stability_impact = stability_boost / 100  # Normalize 0-100 to 0-1

    for i in range(steps):
        stability_factor = (relationship_stability[-1] / 100.0) + (money_impact * 0.1) + (stability_impact * 0.1)
        position += drift * stability_factor * dt + volatility * dW[i] * (1 + stability_factor)
        path.append(position)

        happiness.append(max(30, min(100, happiness[-1] + int(5 * dW[i] * stability_factor + happiness_impact * 5))))
        stability_change = int(10 * dW[i] - (1 - stability_factor) * 3 + money_impact * 5 + stability_impact * 5)
        relationship_stability.append(max(20, min(90, relationship_stability[-1] + stability_change)))

        if i in [steps // 4, steps // 2, 3 * steps // 4]:
            events[i] = random.choice(["Career breakthrough!", "Moved to a new city!", "Found a new hobby!", "Met someone special!"])

        divorce_prob = 0.2 - (money_impact * 0.15) - (happiness_impact * 0.05) - (stability_impact * 0.05)
        if position > 5.0 and relationship_stability[-1] < 40 and random.random() < max(0.05, divorce_prob):
            divorce_events.append(i)
            position = 1.5 + (money_impact * 0.5) + (happiness_impact * 0.3)
            path[-1] = position

    return path, happiness, relationship_stability, events, divorce_events

# Stochastic marriage prediction with multiple factors
def predict_marriage_stochastic(name, dob, place, gender, status, personality, money_factor=0, happiness_level=50, stability_boost=50):
    try:
        birth_date = datetime.strptime(dob, "%d-%m-%Y")
        birth_year = birth_date.year
    except ValueError:
        app.logger.error(f"Invalid date format: {dob}")
        return None, None, None, None, False, [], "Invalid date format"

    age_now = datetime.now().year - birth_year
    life_expectancy = 80
    steps = life_expectancy - age_now
    drift = 0.3 if personality == "introvert" else 0.35
    volatility = 0.4 if personality == "introvert" else 0.45
    initial_position = 2.5 if status.lower() in ["in a relationship", "dating"] else 0 if status.lower() == "single" else 4.5
    initial_position += -0.5 if personality == "introvert" else 0.5

    path, happiness, relationship_stability, events, divorce_events = brownian_motion(steps, drift, volatility, initial_position, money_factor, happiness_level, stability_boost)

    marriage_threshold = 2.5
    marriage_age_1 = None
    marriage_age_2 = None
    ages = list(range(age_now, age_now + len(path)))

    for i, pos in enumerate(path):
        if pos > marriage_threshold and marriage_age_1 is None and i not in divorce_events:
            marriage_age_1 = ages[i]
            break

    for i in range(i + 5, len(path)):
        if path[i] > marriage_threshold and marriage_age_2 is None and i not in divorce_events:
            marriage_age_2 = ages[i]
            break

    divorce_ages = [int(age) for age in [ages[i] for i in divorce_events]]
    no_marriage_predicted = marriage_age_1 is None and initial_position < 5.0

    if marriage_age_2:
        message = f"Stochastic Model: You're going to be married at ages {marriage_age_1} and {marriage_age_2}!"
    elif marriage_age_1:
        message = f"Stochastic Model: You're going to be married at age {marriage_age_1}!"
    else:
        message = f"Stochastic Model: No marriage predicted in this simulation."

    if divorce_ages:
        message += f" Divorces predicted at ages: {', '.join(map(str, divorce_ages))}."

    return marriage_age_1, marriage_age_2, list(zip(ages, path, happiness, relationship_stability)), events, no_marriage_predicted, divorce_ages, message

# ML prediction with multiple factors
def predict_marriage_ml(dob, gender, status, personality, place, money_factor=0, happiness_level=50, stability_boost=50):
    try:
        birth_date = datetime.strptime(dob, "%d-%m-%Y")
        birth_year = birth_date.year
    except ValueError:
        app.logger.error(f"Invalid date format: {dob}")
        return None, "Invalid date format", [0.33, 0.33, 0.34], []

    age_now = datetime.now().year - birth_year
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    status_map = {"Single": 0, "Dating": 1, "Married": 2, "In a relationship": 1}
    personality_map = {"introvert": 0, "extrovert": 1}
    country_code = country_to_code.get(place, 0)

    features = [age_now, gender_map.get(gender, 0), country_code, 2019, personality_map[personality], money_factor, happiness_level, stability_boost]
    app.logger.debug(f"Prediction features: {features}")
    probs = ml_model.predict_proba([features])[0]
    prediction = ml_model.predict([features])[0]
    future_probs = []
    for future_age in range(age_now, age_now + 10):
        future_features = [future_age, gender_map.get(gender, 0), country_code, 2019, personality_map[personality], money_factor, happiness_level, stability_boost]
        future_probs.append(ml_model.predict_proba([future_features])[0].tolist())

    if prediction == 1:
        marriage_age = age_now + random.randint(2, 10)
        message = f"ML Model: Marriage predicted at age {marriage_age} (Probability: {probs[1]:.2f})."
    elif prediction == 2:
        marriage_age = age_now - random.randint(2, 10) if age_now > 20 else None
        message = f"ML Model: Marriage and divorce predicted (Probability of divorce: {probs[2]:.2f})."
    else:
        marriage_age = None
        message = f"ML Model: No marriage predicted (Probability of marriage: {probs[1]:.2f})."

    return marriage_age, message, probs.tolist(), future_probs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    app.logger.debug(f"Incoming request data: {data}")

    name = data.get('name')
    dob = data.get('dob')
    place = data.get('place')
    gender = data.get('gender')
    status = data.get('relationship')
    personality = data.get('personality')
    money_factor = float(data.get('money_factor', 0))
    happiness_level = float(data.get('happiness_level', 50))
    stability_boost = float(data.get('stability_boost', 50))

    if not all([name, dob, place, gender, status, personality]):
        app.logger.error("Missing required fields")
        return jsonify({'error': 'Missing input data'}), 400

    if personality not in ["introvert", "extrovert"]:
        app.logger.error(f"Invalid personality value: {personality}")
        return jsonify({'error': 'Invalid personality. Choose "introvert" or "extrovert".'}), 400

    try:
        stoch_age_1, stoch_age_2, life_path_data, events, no_marriage_predicted, divorce_ages, stoch_message = predict_marriage_stochastic(
            name, dob, place, gender, status, personality, money_factor, happiness_level, stability_boost
        )
        ml_age, ml_message, ml_probs, ml_future_probs = predict_marriage_ml(dob, gender, status, personality, place, money_factor, happiness_level, stability_boost)

        response = {
            'stochastic': {
                'marriage_age_1': stoch_age_1,
                'marriage_age_2': stoch_age_2,
                'life_path_data': [list(map(float, item)) for item in life_path_data] if life_path_data else [],
                'happiness_data': [[int(age), float(happiness)] for age, _, happiness, _ in life_path_data],
                'events': {int(k): v for k, v in events.items()},
                'divorce_ages': divorce_ages,
                'no_marriage_predicted': no_marriage_predicted,
                'prediction_message': stoch_message
            },
            'ml': {
                'marriage_age': ml_age,
                'prediction_message': ml_message,
                'probabilities': ml_probs,
                'future_probabilities': ml_future_probs
            }
        }
        app.logger.debug(f"Response prepared: {response}")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error generating response: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    preprocess_world_marriage_data(DATASET_PATH)
    app.run(host='0.0.0.0', port=5000)
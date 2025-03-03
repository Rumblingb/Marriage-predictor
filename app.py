from flask import Flask, request, jsonify, render_template
import random
import numpy as np
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# In-memory list to store names (temporary, resets on restart)
names_list = []

# Global variable for country mapping
country_to_code = {}

# Load and preprocess World Marriage Dataset with added features
def preprocess_world_marriage_data(file_path):
    global country_to_code
    try:
        app.logger.debug(f"Attempting to load dataset from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        df = pd.read_csv(file_path).sample(n=10000, random_state=42)
        app.logger.debug(f"Dataset loaded. Columns: {df.columns.tolist()}, Shape: {df.shape}")

        country_col = 'Country'
        age_group_col = 'AgeGroup'
        sex_col = 'Sex'
        marital_status_col = 'MaritalStatus'
        year_start_col = 'Data Collection (Start Year)'
        year_end_col = 'Data Collection (End Year)'

        required_cols = [country_col, age_group_col, sex_col, marital_status_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in dataset: {missing_cols}")
        app.logger.debug("Required columns verified")

        df = df.dropna(subset=required_cols)
        app.logger.debug(f"After dropna: Shape: {df.shape}")

        def parse_age(age_str):
            try:
                age_str = age_str.strip('[]')
                if '-' in age_str:
                    start, end = age_str.split('-')
                    return (int(start) + int(end)) / 2
                elif '+' in age_str:
                    return int(age_str.strip('+'))
                else:
                    return float(age_str)
            except Exception as e:
                app.logger.error(f"Error parsing age '{age_str}': {e}")
                return 30.0  # Default age if parsing fails

        df['Age'] = df[age_group_col].apply(parse_age)
        app.logger.debug("Age column created")

        df['Sex_Encoded'] = df[sex_col].map({'Man': 0, 'Woman': 1}).fillna(0)  # Default to 0 if unmapped
        app.logger.debug("Sex_Encoded column created")

        df['Country_Encoded'] = df[country_col].astype('category').cat.codes
        app.logger.debug("Country_Encoded column created")

        df['Year'] = (df[year_start_col] + df[year_end_col]) / 2
        app.logger.debug("Year column created")

        df['Marital_Status_Encoded'] = df[marital_status_col].map({
            'Single': 0, 'Never married': 0, 'Not in union': 0,
            'Married': 1, 'Living together': 1,
            'Divorced': 2, 'Separated': 2, 'Divorced or Separated': 2,
            'Widowed': 3
        }).fillna(0)
        app.logger.debug("Marital_Status_Encoded column created")

        df['Personality_Encoded'] = np.random.choice([0, 1], size=len(df))
        df['Money_Factor'] = np.random.uniform(0, 200000, size=len(df))
        df['Height'] = np.random.uniform(36, 96, size=len(df))
        df['Healthiness'] = np.random.uniform(15, 40, size=len(df))
        app.logger.debug("Additional feature columns created")

        country_to_code = dict(zip(df[country_col], df['Country_Encoded']))
        X = df[['Age', 'Sex_Encoded', 'Country_Encoded', 'Year', 'Personality_Encoded', 'Money_Factor', 'Height', 'Healthiness']]
        y = df['Marital_Status_Encoded']
        app.logger.debug(f"Training features: {X.columns.tolist()}, Shape: {X.shape}")
        app.logger.debug(f"Target variable shape: {y.shape}")
        return X, y
    except Exception as e:
        app.logger.error(f"Error in preprocess_world_marriage_data: {e}")
        raise

# Train ML model
def train_ml_model(file_path):
    try:
        X, y = preprocess_world_marriage_data(file_path)
        model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42)
        model.fit(X, y)
        app.logger.info("Model trained successfully with 8 features")
        return model
    except Exception as e:
        app.logger.error(f"Error training model: {e}")
        raise

# Load or train ML model on startup
DATASET_PATH = 'World Marriage Dataset.csv'
ml_model = train_ml_model(DATASET_PATH)

# Enhanced Brownian motion with updated factors
def brownian_motion(steps=50, drift=0.3, volatility=0.4, initial_position=0, money_factor=0, height=50, healthiness=25):
    dt = 1.0
    position = max(0, initial_position)
    path = [position]
    happiness = [random.randint(50, 90)]
    relationship_stability = [random.randint(50, 90)]
    events = {}
    divorce_events = []

    dW = np.random.normal(0, np.sqrt(dt), steps)
    money_impact = money_factor / 200000
    height_impact = (height - 36) / 60
    healthiness_impact = (healthiness - 15) / 25

    for i in range(steps):
        stability_factor = (relationship_stability[-1] / 100.0) + (money_impact * 0.2) + (healthiness_impact * 0.2)
        position += drift * stability_factor * dt + volatility * dW[i] * (1 + stability_factor)
        path.append(position)

        happiness.append(max(30, min(100, happiness[-1] + int(5 * dW[i] * stability_factor + height_impact * 5))))
        stability_change = int(10 * dW[i] - (1 - stability_factor) * 3 + money_impact * 5 + healthiness_impact * 5)
        relationship_stability.append(max(20, min(90, relationship_stability[-1] + stability_change)))

        if i in [steps // 4, steps // 2, 3 * steps // 4]:
            events[i] = random.choice(["Career breakthrough!", "Moved to a new city!", "Found a new hobby!", "Met someone special!"])

        divorce_prob = 0.3 - (money_impact * 0.2) - (height_impact * 0.05) - (healthiness_impact * 0.1)
        if position > 5.0 and relationship_stability[-1] < 40 and random.random() < max(0.05, divorce_prob):
            divorce_events.append(i)
            position = 1.5 + (money_impact * 0.5) + (height_impact * 0.3)
            path[-1] = position

    return path, happiness, relationship_stability, events, divorce_events

# Stochastic marriage prediction with updated factors
def predict_marriage_stochastic(name, dob, place, gender, status, personality, money_factor=0, height=50, healthiness=25):
    try:
        birth_date = datetime.strptime(dob, "%d-%m-%Y")
        birth_year = birth_date.year
    except ValueError:
        app.logger.error(f"Invalid date format: {dob}")
        return None, None, None, None, False, [], "Invalid date format"

    age_now = datetime.now().year - birth_year
    life_expectancy = 80
    steps = life_expectancy - age_now
    drift = 0.35 if personality == "extrovert" else 0.3
    volatility = 0.45 if personality == "extrovert" else 0.4
    initial_position = 2.5 if status.lower() in ["in a relationship", "dating"] else 0 if status.lower() == "single" else 4.5
    initial_position += 0.5 if personality == "extrovert" else -0.5

    path, happiness, relationship_stability, events, divorce_events = brownian_motion(
        steps, drift, volatility, initial_position, money_factor, height, healthiness
    )

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

# ML prediction with updated factors
def predict_marriage_ml(dob, gender, status, personality, place, money_factor=0, height=50, healthiness=25):
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

    features = [age_now, gender_map.get(gender, 0), country_code, 2019, personality_map[personality], money_factor, height, healthiness]
    app.logger.debug(f"Prediction features: {features}")
    probs = ml_model.predict_proba([features])[0]
    app.logger.debug(f"Prediction probabilities: Single={probs[0]:.2f}, Married={probs[1]:.2f}, Divorced={probs[2]:.2f}, Widowed={probs[3]:.2f}")
    prediction = np.argmax(probs)
    future_probs = []
    for future_age in range(age_now, age_now + 10):
        future_features = [future_age, gender_map.get(gender, 0), country_code, 2019, personality_map[personality], money_factor, height, healthiness]
        future_probs.append(ml_model.predict_proba([future_features])[0].tolist())

    threshold = 0.25
    if probs[1] > threshold:
        marriage_age = age_now + random.randint(2, 10)
        message = f"ML Model: Marriage predicted at age {marriage_age} (Probability: {probs[1]:.2f})."
    elif probs[2] > threshold:
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
    height = float(data.get('height', 50))
    healthiness = float(data.get('healthiness', 25))

    if not all([name, dob, place, gender, status, personality]):
        app.logger.error("Missing required fields")
        return jsonify({'error': 'Missing input data'}), 400

    if personality not in ["introvert", "extrovert"]:
        app.logger.error(f"Invalid personality value: {personality}")
        return jsonify({'error': 'Invalid personality. Choose "introvert" or "extrovert".'}), 400

    # Add name to in-memory list
    names_list.append(name)
    if len(names_list) > 100:  # Optional: limit list size
        names_list.pop(0)

    try:
        stoch_age_1, stoch_age_2, life_path_data, events, no_marriage_predicted, divorce_ages, stoch_message = predict_marriage_stochastic(
            name, dob, place, gender, status, personality, money_factor, height, healthiness
        )
        ml_age, ml_message, ml_probs, ml_future_probs = predict_marriage_ml(
            dob, gender, status, personality, place, money_factor, height, healthiness
        )

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
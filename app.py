from flask import Flask, request, jsonify, render_template
import random
from datetime import datetime
import sqlite3



app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Database setup
DATABASE = 'marriage_predictions.db'


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dob TEXT NOT NULL,
                place TEXT NOT NULL,
                marriage_age_1 INTEGER,
                marriage_age_2 INTEGER
            )
        ''')
        conn.commit()
    print("Database initialized successfully!")


# Initialize the database
init_db()


# Brownian motion simulation
def brownian_motion(steps=50):
    position = 0
    path = [position]
    events = {}

    for i in range(steps):
        move = random.choice([-1, 1])  # Random movement
        position += move
        path.append(position)

        # Randomly assign events at different points
        if i in [steps // 4, steps // 2, 3 * steps // 4]:
            events[i] = random.choice([
                "Career milestone achieved!",
                "You traveled to a new place!",
                "Personal growth spurt!",
                "Mysterious good luck this year!"
            ])

    return path, events


# Marriage prediction logic
def predict_marriage(name, dob, place):
    try:
        # Parse the date of birth
        birth_date = datetime.strptime(dob, "%d-%m-%Y")
        birth_year = birth_date.year
    except ValueError:
        return None, None, None, None  # Return None if the date format is invalid

    age_now = datetime.now().year - birth_year  # Calculate current age
    life_expectancy = 80  # Assume a life expectancy of 80 years
    steps = life_expectancy - age_now  # Simulate remaining life
    path, events = brownian_motion(steps)

    # First marriage prediction logic (round to nearest integer)
    marriage_age_1 = int(age_now + (steps * random.uniform(0.4, 0.6)))  # Round to integer

    # Randomly decide if this user gets a second marriage prediction
    gets_second_marriage = random.choice([True, False])  # 50% chance

    # Second marriage prediction (if applicable)
    marriage_age_2 = None
    if gets_second_marriage:
        marriage_age_2 = int(age_now + (steps * random.uniform(0.6, 0.8)))  # Round to integer

    # Generate life path data with corresponding ages
    ages = list(range(age_now, age_now + len(path)))
    life_path_data = list(zip(ages, path))

    return marriage_age_1, marriage_age_2, life_path_data, events


# API endpoint to handle user input
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    name = data.get('name')
    dob = data.get('dob')
    place = data.get('place')

    if not name or not dob or not place:
        return jsonify({'error': 'Missing input data'}), 400

    marriage_age_1, marriage_age_2, life_path_data, events = predict_marriage(name, dob, place)
    if marriage_age_1 is None:
        return jsonify({'error': 'Invalid date format. Use DD-MM-YYYY.'}), 400

    try:
        # Save user data and predictions to the database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO predictions (name, dob, place, marriage_age_1, marriage_age_2)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, dob, place, marriage_age_1, marriage_age_2))
            conn.commit()
        print("Data saved to database successfully!")

        # Return results as JSON
        return jsonify({
            'marriage_age_1': marriage_age_1,
            'marriage_age_2': marriage_age_2,
            'life_path_data': life_path_data,
            'events': events
        })
    except Exception as e:
        print(f"Error saving to database: {e}")
        return jsonify({'error': str(e)}), 500


# Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
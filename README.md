When Are You Going to Get Married!? - README.txt
================================================

Overview
--------
This is a web application that predicts your marriage prospects using stochastic modeling and machine learning. Inspired by the classic question "When are you going to get married lah, so old already?" (thanks, Asian aunties and uncles!), it blends Brownian motion with a Random Forest Classifier to simulate life events and marital status probabilities. Users input personal details (name, DOB, gender, etc.) and tweak sliders for income, height, and healthiness (BMI) to see predictions visualized with interactive charts. Deployed on Render's free tier, it's a fun showcase of applied stochastic calculus and ML.

Project Structure
-----------------
- app.py: Main Flask application with stochastic and ML logic
- templates/index.html: Frontend HTML template
- World Marriage Dataset.csv: Dataset for ML training
- Procfile: Render deployment config (web: gunicorn app:app)
- requirements.txt: Python dependencies
- README.txt: This file

Stochastic Model: Variables and Numbers Explained
------------------------------------------------
The stochastic model simulates a "commitment path" over time using Brownian motion, a continuous-time stochastic process. It predicts marriage and divorce events based on a position path crossing thresholds. Below are the key variables and the specific numbers I've chosen, with their meanings:

### Core Equation
The commitment path X(t) evolves via a discretized stochastic differential equation:
X(t+1) = X(t) + drift * stability_factor * dt + volatility * dW[i] * (1 + stability_factor)

#### 1. Time Step (dt)
- Value: 1.0 (1 year)
- Why: Each step represents one year of life from the user's current age to a life expectancy of 80. This simplifies the simulation while keeping it relatable.

#### 2. Steps
- Value: life_expectancy - age_now (e.g., 80 - 25 = 55 steps for a 25-year-old)
- Why: Steps are calculated dynamically based on the user's date of birth, simulating their remaining lifespan.

#### 3. Drift (μ)
- Values: 
  - 0.3 (introverts)
  - 0.35 (extroverts)
- Why: Drift is the expected upward push in commitment per year. Extroverts get a slightly higher drift (0.35 vs. 0.3) to reflect a tendency toward social engagement, which might nudge them toward relationships faster. These values are small to keep the path gradual and realistic.

#### 4. Volatility (σ)
- Values:
  - 0.4 (introverts)
  - 0.45 (extroverts)
- Why: Volatility controls the randomness of life’s ups and downs. Extroverts have higher volatility (0.45 vs. 0.4) to simulate more variability in their social lives. These numbers ensure noticeable fluctuations without making the path too chaotic.

#### 5. Initial Position (X(0))
- Values:
  - 0 (Single)
  - 2.5 (In a relationship or Dating)
  - 4.5 (Married)
  - Adjusted by ±0.5 (extrovert: +0.5, introvert: -0.5)
- Why: The starting point reflects the user’s current relationship status. Single starts at 0 (no commitment), Dating/In a relationship at 2.5 (near the marriage threshold), and Married at 4.5 (well into committed territory). Personality tweaks it slightly—extroverts start higher, introverts lower—to bias the initial trajectory.

#### 6. Stability Factor
- Formula: stability_factor = (relationship_stability[-1] / 100) + 0.2 * money_impact + 0.2 * healthiness_impact
- Components:
  - relationship_stability[-1]: Last stability value (20–90)
  - money_impact: Income / 200,000 (0 to 1)
  - healthiness_impact: (BMI - 15) / 25 (0 to 1)
- Why: This scales drift and volatility based on life factors. Income (0–200,000 USD/year) and healthiness (BMI 15–40) contribute up to 0.2 each, while stability (normalized 0–0.9) anchors it. Higher values amplify both drift (faster commitment) and volatility (bigger swings).

#### 7. Brownian Increment (dW[i])
- Value: np.random.normal(0, sqrt(dt), steps) (e.g., N(0, 1) since dt = 1)
- Why: This is the random noise from Brownian motion, drawn from a normal distribution with mean 0 and variance dt. It introduces unpredictable life events, scaled by volatility.

#### 8. Marriage Threshold
- Value: 2.5
- Why: If X(t) exceeds 2.5, a marriage is predicted (first or second). This moderate threshold allows early marriages for some paths while requiring sustained growth for others, balancing realism and fun.

#### 9. Divorce Conditions
- Conditions:
  - X(t) > 5.0 (high commitment)
  - relationship_stability < 40 (low stability)
  - Probability: 0.3 - 0.2 * money_impact - 0.05 * height_impact - 0.1 * healthiness_impact (min 0.05)
- Why: Divorce triggers when commitment is high but stability crashes, with a base probability of 0.3 reduced by user inputs (e.g., high income lowers it). Height (3–8 ft, normalized) has a smaller effect (0.05) than money (0.2) or healthiness (0.1), reflecting their relative influence on stability.

#### 10. Relationship Stability
- Update: max(20, min(90, last_value + 10 * dW[i] - 3 * (1 - stability_factor) + 5 * money_impact + 5 * healthiness_impact))
- Why: Stability fluctuates between 20 and 90, driven by random noise (dW[i]), penalized by instability (1 - stability_factor), and boosted by money and healthiness. The ±5 weights ensure tangible user input effects.

#### 11. Happiness
- Update: max(30, min(100, last_value + 5 * dW[i] * stability_factor + 5 * height_impact))
- Why: Happiness ranges 30–100, with random changes scaled by stability and boosted by height (normalized 0–1). It’s a secondary metric for visualization, not directly affecting marriage events.

### Why These Numbers?
- Drift (0.3–0.35) and volatility (0.4–0.45) are small to keep paths smooth yet variable, avoiding wild jumps.
- Thresholds (2.5, 5.0) and weights (0.2, 0.05, 0.1) are tuned for balance—marriage isn’t too easy or rare, and user inputs matter without dominating.
- Ranges (e.g., income 0–200,000, BMI 15–40) reflect realistic bounds normalized for the model.

Machine Learning Model
---------------------
- Algorithm: Random Forest Classifier
- Dataset: World Marriage Dataset (10,000-row sample)
- Features: Age, Sex_Encoded, Country_Encoded, Year, Personality_Encoded, Money_Factor, Height, Healthiness
- Target: Marital_Status_Encoded (0=Single, 1=Married, 2=Divorced, 3=Widowed)
- Parameters: n_estimators=50, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42
- Threshold: 0.25 for marriage/divorce predictions
- Why: Trained on startup to predict probabilities, complementing the stochastic model with data-driven insights.

Setup Instructions
------------------
1. Clone the repo: git clone https://github.com/your-username/your-repo.git
2. Install dependencies: pip install -r requirements.txt
3. Ensure World Marriage Dataset.csv is in the root directory
4. Run locally: python app.py
   - Visit http://localhost:5000

Deployment on Render
--------------------
1. Push to GitHub with all files
2. On Render (render.com):
   - New > Web Service
   - Connect your repo
   - Settings:
     - Build: pip install -r requirements.txt
     - Start: gunicorn app:app
     - Plan: Free
3. Access at https://your-app-name.onrender.com

Notes
-----
- Names are collected in-memory (resets on restart).
- Free tier limitations: App sleeps after inactivity

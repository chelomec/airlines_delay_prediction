Machine learning project to predict whether a flight will be delayed by more than 15 minutes, using historical operational and meteorological data from U.S. airlines.

📌 Problem Statement
Flight punctuality is a key performance indicator for airlines — directly impacting operational efficiency, customer satisfaction, and revenue. This project builds a binary classification model to anticipate delays before they occur, enabling proactive operational decisions.
Target variable: DEP_DEL15 — 1 if departure delay > 15 minutes, 0 otherwise.

📦 Dataset

Source: Kaggle — 2019 Airline Delays and Cancellations
Size: 4.5M+ records · 30 variables
Coverage: All U.S. commercial airline flights in 2019
Variables include: flight schedule, airline, airport, aircraft age, concurrent flights, weather conditions (precipitation, snow, wind, temperature), and historical delay rates


🔍 Key Findings from EDA

No single variable shows a strong direct correlation with delays — prediction requires combining multiple features
Time of day is the strongest predictor: evening and night flights are significantly more likely to be delayed
Segment number matters: delay probability increases sharply from segment 9 onwards
Precipitation and snow increase delay risk, with snow having a seasonal effect (stronger in spring/fall)
Summer has the highest delay rate despite no snow — driven by air traffic volume
Frontier Airlines shows the lowest punctuality; Hawaiian Airlines the highest
Atlanta is the busiest departure airport; highest traffic concentration is on the East Coast


⚙️ Feature Engineering
New features created from raw variables:
FeatureDescriptionMOMENT_OF_THE_DAYTime block categorized as dawn / morning / midday / afternoon / nightSEASONMonth mapped to meteorological seasonPRCP_CAT / SNOW_CAT / SNWD_CATBinary flags for precipitation and snow presencePREVIOUS_AIRPORT_ENCODEDLabel-encoded airport of originCARRIER_NAME_ENCODEDLabel-encoded airline identifier
Final feature set used for modeling:
SEGMENT_NUMBER, PREVIOUS_AIRPORT_ENCODED, DEPARTING_AIRPORT_ENCODED, MOMENT_OF_THE_DAY_ENCODED, CARRIER_NAME_ENCODED, PRCP_CAT, SNOW_CAT

🤖 Models Evaluated
ModelAccuracyClass 1 RecallClass 1 F1Logistic Regression (baseline)81.2%0.4%0.008Logistic Regression (tuned)78.4%14.4%0.200KNN (baseline)75.9%15.2%0.192KNN (tuned)74.9%17.6%0.208Random Forest (baseline)77.1%14.5%0.193Random Forest (tuned)63.3%58.5%0.375XGBoost (threshold=0.7)63.9%56.8%0.372

All models face a common challenge: class imbalance (~81% on-time vs ~19% delayed).


✅ Best Model: XGBoost with Threshold Optimization
Best hyperparameters:

n_estimators: 200
learning_rate: 0.1
max_depth: 5
subsample: 0.8
colsample_bytree: 0.8
scale_pos_weight: 10

Optimal threshold: 0.7
At threshold 0.7, the model achieves:

Class 1 (delayed) recall: 56.8%
Class 1 F1-score: 0.372
Overall accuracy: 63.9%

Threshold tuning was essential: lower thresholds maximize recall for delayed flights but produce too many false positives. Threshold 0.7 provides the best balance between sensitivity and precision.

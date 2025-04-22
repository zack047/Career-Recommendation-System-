from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler, label encoder, model, and class names=====================
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# Recommendations ===========================================================
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, 
                              extracurricular_activities_encoded, weekly_self_study_hours, 
                              math_score, history_score, physics_score, chemistry_score, 
                              biology_score, english_score, geography_score, total_score,
                              average_score]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Normalize probabilities to ensure they sum to 100%
    normalized_probabilities = (probabilities[0] / np.sum(probabilities[0])) * 100

    # Create list of tuples with (career_name, normalized_probability)
    all_classes_names_probs = [
        (class_names[idx], float(normalized_probabilities[idx]))
        for idx in range(len(class_names))
        if idx < len(normalized_probabilities)
    ]

    # Sort by probability in descending order and handle equal probabilities by name
    all_classes_names_probs.sort(key=lambda x: (-x[1], x[0]))
    
    # Verify total percentage is 100%
    total_percentage = sum(prob for _, prob in all_classes_names_probs)
    if abs(total_percentage - 100.0) > 0.1:  # Allow small floating point differences
        # Adjust the last probability to make total exactly 100%
        difference = 100.0 - total_percentage
        all_classes_names_probs[-1] = (
            all_classes_names_probs[-1][0], 
            all_classes_names_probs[-1][1] + difference
        )
    
    # Split into top 3 and other paths
    top_three = all_classes_names_probs[:3]
    other_paths = all_classes_names_probs[3:]
    
    # Sort other_paths again to ensure descending order
    other_paths = sorted(other_paths, key=lambda x: (-x[1], x[0]))

    return top_three, other_paths



@app.route("/")
def home():
    return render_template('home.html')  # Corrected path

@app.route("/recommend")
def recommend():
    return render_template('recommend.html')  # Corrected path

@app.route("/pred", methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job']
        absence_days = int(request.form['absence_days'])
        extracurricular_activities = request.form['extracurricular_activities']
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        top_three, other_paths = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                          weekly_self_study_hours, math_score, history_score, physics_score,
                                          chemistry_score, biology_score, english_score, geography_score,
                                          total_score, average_score)

        return render_template('result.html', top_three=top_three, other_paths=other_paths)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)

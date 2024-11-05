from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('credit_model.pkl')

# Define the mapping between form fields and model's expected columns
form_to_model_columns = {
    'age': 'Attribute1',
    'job': 'Attribute2',
    'credit_amount': 'Attribute3',
    'duration': 'Attribute4',
    'housing': 'Attribute5',
    'saving_accounts': 'Attribute6',
    'checking_account': 'Attribute7',
    'purpose': 'Attribute8',
}

# Define mappings for categorical fields
job_mapping = {
    'manager': 0,
    'technician': 1,
    'clerk': 2,
    # Add more mappings as needed
}

housing_mapping = {
    'own': 0,
    'rent': 1,
    'free': 2,
}

savings_mapping = {
    'little': 0,
    'moderate': 1,
    'rich': 2,
}

checking_mapping = {
    'little': 0,
    'moderate': 1,
    'rich': 2,
}

purpose_mapping = {
    'car': 0,
    'furniture': 1,
    'radio/tv': 2,
    # Add more mappings as needed
}

# Define all expected columns by the model
expected_columns = [f'Attribute{i}' for i in range(1, 21)]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        form_data = request.form.to_dict()
        print("Form Data:", form_data)
        
        # Map form data to the expected column names and values
        mapped_data = {
            'Attribute1': int(form_data['age']),
            'Attribute2': job_mapping[form_data['job']],
            'Attribute3': int(form_data['credit_amount']),
            'Attribute4': int(form_data['duration']),
            'Attribute5': housing_mapping[form_data['housing']],
            'Attribute6': savings_mapping[form_data['saving_accounts']],
            'Attribute7': checking_mapping[form_data['checking_account']],
            'Attribute8': purpose_mapping[form_data['purpose']],
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([mapped_data])
        print("DataFrame:", df)
        
        # Check if all expected columns are present, add missing columns with default values
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        print("DataFrame with expected columns:", df)
        
        # Reorder columns to match the expected order
        df = df[expected_columns]
        print("DataFrame after reordering columns:", df)
        
        # Make a prediction
        prediction = model.predict(df)[0]
        print("Prediction:", prediction)
        
        # Interpret the prediction
        result = 'Eligible' if prediction == 1 else 'Not Eligible'
        
        # Render the result
        return render_template('index.html', result=result)
    
    except Exception as e:
        # Handle errors
        result = 'Error: {}'.format(str(e))
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

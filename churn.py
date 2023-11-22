import uvicorn
from fastapi import FastAPI, Query
import pickle
import pandas as pd
import warnings
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi import Form

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Create a FastAPI app instance
app = FastAPI()

# Load the pickled model using a relative file path
with open('model_churn.pkl', "rb") as model_file:
    model = pickle.load(model_file)

# Define the HTML content
html_content = """
<!DOCTYPE html>
<html>
<style>
    body {
        background-image: url(https://media.istockphoto.com/id/1328094196/photo/cellular-communications-tower-for-mobile-phone-and-video-data-transmission.jpg?s=612x612&w=0&k=20&c=OdZUBu8Az9bhDkmsaYnBoOPWMA6yU0anPQImhsS4Ulg=);
        background-size: cover;
        text-align: center; /* Center-align text within the body */
    }
    
    form {
        text-align: left;
    }
    
    select {
    font-size: 16px; 
    width: 102.4%; 
    box-sizing: border-box; 
    padding: 8px; 
    }
    h1 {
        font-size: 28px; /* Change the font size for the heading */
        color: #333; /* Change the text color for the heading */
        font-family: "Verdana", sans-serif;
    }

    label {
        font-size: 18px; /* Change the font size for labels */
        color: #555; /* Change the text color for labels */
    }

    input {
        font-size: 16px; /* Change the font size for input fields */
        padding: 5px; /* Add padding to input fields */
        margin: 5px 0; /* Add margin to input fields */
        width: 100%; /* Make input fields 100% width of their container */
        font-family: "Verdana", sans-serif;
    }

    input[type="submit"] {
        background-color: #007BFF; /* Change the background color for the submit button */
        color: #fff; /* Change the text color for the submit button */
        font-size: 18px; /* Change the font size for the submit button */
        padding: 10px 20px; /* Add padding to the submit button */
        cursor: pointer;
        font-family: "Verdana", sans-serif;
    }

    input[type="submit"]:hover {
        background-color: #0056b3; /* Change the background color on hover */
    }

    
    .header {
        background-color: rgba(255, 255, 255, 0.5);
        padding: 2px;
        border: 10px solid rgba(255, 255, 255, 0.5);
    }

    .form-container {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 20px;
        margin: 20px auto; /* Add margin to create space */
        width: 50%;
        border-radius: 10px;
    }
    .header h1 
    {
        color: black; /* Set text color to white */
        font-size: 30px;
        font-family: "Verdana", sans-serif;
    }

    h2 {
        font-size: 20px; /* Change the font size for the result heading */
        color: #333; /* Change the text color for the result heading */
        font-family: "Verdana", sans-serif;
    }

    p {
        font-size: 18px; /* Change the font size for the result text */
        color: #333; /* Change the text color for the result text */
        font-family: "Verdana", sans-serif;
    }
</style>
<head>
    <title>Telecom Customer Churn Prediction</title>
</head>
<body>
    <div class="header">
        <img src="https://www.freepnglogos.com/uploads/logo-chatgpt-png/black-chatgpt-logo-circle-symbol-black-png-0.png" alt="Company Logo" style="width: 270px; height: 130px; margin-top: 25px;">
        <h1>Telecom Company Customer Churn</h1>
    </div>
    <div class="form-container">
    <h1>Customer Churn Predict</h1>
    <form action="/predict/" method="post" id="churnForm">

        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required><br>
       
        <label for="gender">Gender:</label>
        <select id="gender" name="gender" required>
            <option value="Female">Female</option>
            <option value="Male">Male</option>
        </select><br>

        <label for="SeniorCitizen">Senior Citizen:</label>
        <input type="number" id="SeniorCitizen" name="SeniorCitizen" required><br>

        <label for="Partner">Partner:</label>
        <select id="Partner" name="Partner" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select><br>

        <label for="Dependents">Dependents:</label>
        <select id="Dependents" name="Dependents" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select><br>

        <label for="tenure">Tenure:</label>
        <input type="number" id="tenure" name="tenure" required><br>

        <label for="PhoneService">Phone Service:</label>
        <select id="PhoneService" name="PhoneService" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select><br>

        
        <label for="MultipleLines">MultipleLines:</label>
        <select id="MultipleLines" name="MultipleLines" required>
            <option value="Yes">Yes</option>
            <option value="No">No phone service</option>
            <option value="No">No</option>
        </select><br>

        <label for="InternetService">Internet Service:</label>
        <select id="InternetService" name="InternetService" required>
            <option value="DSL">DSL</option>
            <option value="Fiberoptic">Fiber optic</option>
            <option value="No">No</option>
        </select><br>

        <label for="OnlineSecurity">Online Security:</label>
        <select id="OnlineSecurity" name="OnlineSecurity" required>
            <option value="Yes">Yes</option>
            <option value="Nointernetservice">No internet service</option>
            <option value="No">No</option>
        </select><br>

        <label for="OnlineBackup">Online Backup:</label>
        <select id="OnlineBackup" name="OnlineBackup" required>
            <option value="Yes">Yes</option>
            <option value="Nointernetservice">No internet service</option>
            <option value="No">No</option>
        </select><br>

        <label for="DeviceProtection">Device Protection:</label>
        <select id="DeviceProtection" name="DeviceProtection" required>
            <option value="Yes">Yes</option>
            <option value="Nointernetservice">No internet service</option>
            <option value="No">No</option>
        </select><br>

        <label for="TechSupport">Tech Support:</label>
        <select id="TechSupport" name="TechSupport" required>
            <option value="Yes">Yes</option>
            <option value="Nointernetservice">No internet service</option>
            <option value="No">No</option>
        </select><br>

        <label for="StreamingTV">Streaming TV:</label>
        <select id="StreamingTV" name="StreamingTV" required>
            <option value="Yes">Yes</option>
            <option value="Nointernetservice">No internet service</option>
            <option value="No">No</option>
        </select><br>

        <label for="StreamingMovies">Streaming Movies:</label>
        <select id="StreamingMovies" name="StreamingMovies" required>
            <option value="Yes">Yes</option>
            <option value="Nointernetservice">No internet service</option>
            <option value="No">No</option>
        </select><br>

        
        <label for="Contract">Contract:</label>
        <select id="Contract" name="Contract" required>
            <option value="Month-to-month">Month-to-month</option>
            <option value="Oneyear">One year</option>
            <option value="Twoyear">Twoyear</option>
        </select><br>
        
        <label for="PaperlessBilling">Paperless Billing:</label>
        <select id="PaperlessBilling" name="PaperlessBilling" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select><br>

        <label for="PaymentMethod">Payment Method:</label>
        <select id="PaymentMethod" name="PaymentMethod" required>
            <option value="Banktransfer(automatic)">Bank transfer(automatic)</option>
            <option value="CreditCard(automatic)">Credit Card(automatic)</option>
            <option value="Electroniccheck">Electronic check</option>
            <option value="Mailedchecked">Mailed checked</option>
        </select><br>

        <label for="MonthlyCharges">Monthly Charges:</label>
        <input type="number" id="MonthlyCharges" name="MonthlyCharges" required><br>

        <label for="TotalCharges">Total Charges:</label>
        <input type="number" id="TotalCharges" name="TotalCharges" required><br>
 

        <input type="submit" value="Predict Satisfaction">
    </form>
    <h2>Prediction Result:</h2>
    <p id="prediction_result"></p>
    </div>

    <script>
        const form = document.getElementById('churnForm');
        const predictionResult = document.getElementById('prediction_result');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const formData = new FormData(form);

            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            // Handle the prediction result directly in JavaScript
            const customerName = formData.get('name');
            const predictionText = data['prediction'] === 1
                ? `${customerName} is likely to churn. Consider taking preventive actions.`
                : `${customerName} is not likely to churn. No immediate action required.`;

            predictionResult.textContent = `Prediction:   ${predictionText}`;
        });
    </script>
</body>
</html>
"""

# Define the endpoint to serve the HTML content
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=html_content)

# Define the endpoint to make predictions
@app.post("/predict/")
async def predict(
    name: str = Form(...),
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
):
    # The rest of your code remains unchanged

    data = pd.DataFrame({
        'name': [name], 
        'gender': [gender],  # replace with the actual value
        'SeniorCitizen': [SeniorCitizen],  # replace with the actual value
        'Partner': [Partner],  # replace with the actual value
        'Dependents': [Dependents],  # replace with the actual value
        'tenure': [tenure],  # replace with the actual value
        'PhoneService': [PhoneService],  # replace with the actual value
        'MultipleLines': [MultipleLines],  # replace with the actual value
        'InternetService': [InternetService],  # replace with the actual value
        'OnlineSecurity': [OnlineSecurity],  # replace with the actual value
        'OnlineBackup': [OnlineBackup],  # replace with the actual value
        'DeviceProtection': [DeviceProtection],  # replace with the actual value
        'TechSupport': [TechSupport],  # replace with the actual value
        'StreamingTV': [StreamingTV],  # replace with the actual value
        'StreamingMovies': [StreamingMovies],  # replace with the actual value
        'Contract': [Contract],  # replace with the actual value
        'PaperlessBilling': [PaperlessBilling],  # replace with the actual value
        'PaymentMethod': [PaymentMethod],  # replace with the actual value
        'MonthlyCharges': [MonthlyCharges],  # replace with the actual value
        'TotalCharges': [TotalCharges],  # replace with the actual value
        })
    
    # Mapping Contract to numerical values
    contract_mapping = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    data['Contract'] = data['Contract'].map(contract_mapping)

    data['Sqrt_tenure'] = data['tenure'] ** 0.5
    data['Sqrt_MonthlyCharges'] = data['MonthlyCharges'] ** 0.5
    data['Sqrt_TotalCharges'] = data['TotalCharges'] ** 0.5

    # data = pd.get_dummies(data, columns=['InternetService'])
    
    data['Sum_Contract_InternetService_Fiber optic'] = data['Contract']
    data.loc[data['InternetService'] == 'Fiber optic', 'Sum_Contract_InternetService_Fiber optic'] += 1

    data['Sum_PaymentMethod_Electronic check_Contract'] = data['Contract']
    data.loc[data['PaymentMethod'] == 'Electronic check', 'Sum_PaymentMethod_Electronic check_Contract'] += 1

    data['Subt_tenure_MonthlyCharges'] = data['tenure'] - data['MonthlyCharges']
    
    data['Subt_Contract_InternetService_No'] = data['Contract']
    data.loc[data['InternetService'] == 'No', 'Subt_Contract_InternetService_No'] += 1

    data['Div_Contract_MonthlyCharges'] = data['Contract'] / data['MonthlyCharges']

    # Selecting features for prediction
    features = [
        'tenure', 'OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'TotalCharges',
        'Sqrt_tenure', 'Sqrt_MonthlyCharges', 'Sqrt_TotalCharges',
        'Sum_Contract_InternetService_Fiber optic',
        'Sum_PaymentMethod_Electronic check_Contract',
        'Subt_tenure_MonthlyCharges', 'Subt_Contract_InternetService_No',
        'Div_Contract_MonthlyCharges'
    ]
    data['OnlineSecurity'] = data['OnlineSecurity'].replace({'Yes':1,'No':0})
    data['TechSupport'] = data['TechSupport'].replace({'Yes':1,'No':0})
    data['PaperlessBilling'] = data['PaperlessBilling'].replace({'Yes':1,'No':0})
    data['OnlineSecurity'] = data['OnlineSecurity'].replace({'Yes':1,'No':0})

    # Make predictions using the pre-trained model
    predictions = model.predict(data[features])

    # Convert the NumPy array to a Python scalar
    prediction_result = predictions.item()

    return {"prediction": prediction_result}



# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5002,
        log_level="debug",
    )

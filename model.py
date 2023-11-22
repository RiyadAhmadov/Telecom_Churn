# Pandas for data manipulation and analysis
import pandas as pd  
# NumPy for numerical operations
import numpy as np  
# XGBoost, a popular machine learning library for gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
# Using LDA for dimension reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
# Scikit-learn for machine learning metrics
from sklearn.metrics import accuracy_score ,classification_report
# Scikit-learn for model selection and training
from sklearn.model_selection import RandomizedSearchCV, train_test_split  
# Warnings module to manage and handle warning messages during code execution
import warnings as wg  
wg.filterwarnings('ignore')
# Import Pickle Libraries for converting model to pickle file
import pickle


df = pd.read_csv(r'C:\Users\HP\OneDrive\İş masası\python\churn\churn_data_telecom.csv')
print(df.head())

del df['Unnamed: 0']


# Let's define input and target features
y = df['Churn']
X = df.drop(columns = ['Churn'])

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 42, test_size = 0.2)

# Create an XGBoost classifier
model = GradientBoostingClassifier(n_estimators = 170, learning_rate  = 0.05, random_state = 42, max_depth = 3)


# Train the model on the training data
model.fit(X_train, y_train)
# Make predictions on the training data
y_pred_tr = model.predict(X_train)
# Calculate accuracy score for training data
accuracy_score_train = accuracy_score(y_train, y_pred_tr)
# Print the accuracy score for training data
print("Accuracy on training data:", accuracy_score_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Calculate accuracy score for test data
accuracy_score_test = accuracy_score(y_test, y_pred)
# Print the accuracy score for test data
print("Accuracy on test data:", accuracy_score_test)

new_value = np.array([1, 1, 1, 1, 1.0, 1.0, 1.0, 1, 1, 1, 1, 1, 3.0])
new_value_reshaped = new_value.reshape(1, -1)

y_pred = model.predict(new_value_reshaped)
print(y_pred)


#        'Unnamed: 0', 'tenure', 'OnlineSecurity', 'TechSupport',
#        'PaperlessBilling', 'TotalCharges', 'Sqrt_tenure',
#        'Sqrt_MonthlyCharges', 'Sqrt_TotalCharges',
#        'Sum_Contract_InternetService_Fiber optic',
#        'Sum_PaymentMethod_Electronic check_Contract',
#        'Subt_tenure_MonthlyCharges', 'Subt_Contract_InternetService_No',
#        'Div_Contract_MonthlyCharges'
#        -------------------------------------------------------------------
#        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
#        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
#        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
#        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'


# # Save the model to a pickle file
# with open('model_churn.pkl', 'wb') as file:
#     pickle.dump(model, file)

# # Load the model from the pickle file
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# DATA PREPROCESSING
# Load the data
data = pd.read_csv('merged_eskom_data_1.csv')

# Convert date to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Day_of_week'] = data['Date'].dt.dayofweek
data['Day_of_year'] = data['Date'].dt.dayofyear

# Convert time to hours since midnight
data['Time'] = data['Time'].astype(str).str.zfill(4)
data['Hour'] = data['Time'].str[:2].astype(int)
data['Minute'] = data['Time'].str[2:].astype(int)
data['Time_of_day'] = data['Hour'] + data['Minute']/60

# Prepare features and target
features = ['Year', 'Month', 'Day', 'Day_of_week', 'Day_of_year', 
            'Hour', 'Minute', 'Time_of_day', 'Load Block']
target = 'Load Shedding Stage'

X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL TRAINING
# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")



# MAKING PREDICTION
def predict_load_shedding(date, time, load_block):
    """
    Predict load shedding stage given date, time, and load block
    
    Parameters:
    date (str): Date in YYYYMMDD format
    time (str): Time in HHMM format
    load_block (int): Load block number
    
    Returns:
    int: Predicted load shedding stage
    """
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Date': [date],
        'Time': [time],
        'Load Block': [load_block]
    })
    
    # Convert date to datetime and extract features
    input_data['Date'] = pd.to_datetime(input_data['Date'], format='%Y%m%d')
    input_data['Year'] = input_data['Date'].dt.year
    input_data['Month'] = input_data['Date'].dt.month
    input_data['Day'] = input_data['Date'].dt.day
    input_data['Day_of_week'] = input_data['Date'].dt.dayofweek
    input_data['Day_of_year'] = input_data['Date'].dt.dayofyear
    
    # Convert time to hours since midnight
    input_data['Time'] = input_data['Time'].astype(str).str.zfill(4)
    input_data['Hour'] = input_data['Time'].str[:2].astype(int)
    input_data['Minute'] = input_data['Time'].str[2:].astype(int)
    input_data['Time_of_day'] = input_data['Hour'] + input_data['Minute']/60
    
    # Prepare features
    features = ['Year', 'Month', 'Day', 'Day_of_week', 'Day_of_year', 
                'Hour', 'Minute', 'Time_of_day', 'Load Block']
    X_input = input_data[features]
    
    # Scale the features
    X_input_scaled = scaler.transform(X_input)
    
    # Make prediction
    # prediction = model.predict(X_input_scaled)
    
    # return prediction[0]
    return X_input_scaled

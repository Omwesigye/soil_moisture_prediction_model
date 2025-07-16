import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import sys
import pickle
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

data_path =  r"D:\SOIL_MOISTURE\cleaned_soil_moisture_dataset.csv"
model_save_path = r"D:\SOIL_MOISTURE\ml_models\soil_moisture_model.keras"
scaler_save_paths = {
    'feature': r"D:\SOIL_MOISTURE\feature_scaler.pkl",
    'moisture': r"D:\SOIL_MOISTURE\moisture_scaler.pkl"
}


def retrain_model(data_path, model_save_path, scaler_save_paths):
    # Load data
    df = pd.read_csv(data_path)
    
    # Features and targets
    features = ['temperature_celsius', 'humidity_percent', 'battery_voltage', 'hour',
                'day', 'month', 'weekday', 'location_encoded', 'status_encoded']
    X = df[features]
    y_moisture = df['soil_moisture_percent']
    y_action = df['irrigation_action_encoded']
    
    # One-hot encode target for classification
    y_action_onehot = to_categorical(y_action)
    
    # Split dataset
    X_train, X_test, y_moisture_train, y_moisture_test, y_action_train, y_action_test = train_test_split(
        X, y_moisture, y_action_onehot, test_size=0.2, random_state=42, stratify=y_action_onehot)
    
    # Scale input features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale moisture target (regression output)
    moisture_scaler = StandardScaler()
    y_moisture_train_scaled = moisture_scaler.fit_transform(y_moisture_train.values.reshape(-1, 1))
    y_moisture_test_scaled = moisture_scaler.transform(y_moisture_test.values.reshape(-1, 1))
    
    # Build the Keras model
    input_layer = Input(shape=(X_train_scaled.shape[1],))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(256, activation='relu')(x)
    moisture_output = Dense(1, name='moisture_level')(x)
    action_hidden = Dense(64, activation='relu')(x)
    irrigation_action_output = Dense(y_action_train.shape[1], activation='softmax', name='irrigation_action')(action_hidden)
    
    model = Model(inputs=input_layer, outputs={
        'moisture_level': moisture_output,
        'irrigation_action': irrigation_action_output
    })
    
    model.compile(
        optimizer='adam',
        loss={
            'moisture_level': 'mse',
            'irrigation_action': 'categorical_crossentropy'
        },
        metrics={
            'moisture_level': 'mean_absolute_error',
            'irrigation_action': 'accuracy'
        }
    )
    
    # Train model
        # Train model
    model.fit(
        X_train_scaled,
        {
            'moisture_level': y_moisture_train_scaled,
            'irrigation_action': y_action_train
        },
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Save model in .keras format
    model.save(model_save_path) 
    
    # Save scalers (INDENTED INSIDE FUNCTION)
    with open(scaler_save_paths['feature'], 'wb') as f:
        pickle.dump(feature_scaler, f)

    with open(scaler_save_paths['moisture'], 'wb') as f:
        pickle.dump(moisture_scaler, f)
    print("✅ Model re_training completed.")

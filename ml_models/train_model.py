import sys
import os
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler # type: ignore
import traceback

# Get MEDIA_ROOT from Django settings
import django # type: ignore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Soil_Moisture.settings')
django.setup()
from django.conf import settings # type: ignore
ml_models_dir = os.path.join(settings.BASE_DIR, 'ml_models')  # Always use BASE_DIR/ml_models


def retrain_and_save_model(data_path, final_model_path):
    # Save all artifacts in the same directory as the model
    ml_models_dir = os.path.dirname(final_model_path)
    feature_scaler_path = os.path.join(ml_models_dir, "feature_scaler.pkl")
    moisture_scaler_path = os.path.join(ml_models_dir, "moisture_scaler.pkl")
    action_encoder_path = os.path.join(ml_models_dir, "action_encoder.pkl")
    if not os.path.exists(ml_models_dir):
        os.makedirs(ml_models_dir)

    # === Load Data ===
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    print("CSV columns:", df.columns.tolist())

    # === Define Features and Target ===
    X = df[["temperature_celsius", "humidity_percent", "battery_voltage", "hour", "day", "month", "weekday", "location_encoded"]]
    y_moisture = df["soil_moisture_percent"]

    # === Inspect Data ===
    print(df.head())
    print("irrigation_action dtype:", df['irrigation_action'].dtype)
    print("irrigation_action has NaNs:", df['irrigation_action'].isna().any())
    df = df.dropna(subset=['irrigation_action'])
    print("After dropping NaNs, value counts:", df['irrigation_action'].value_counts())
    print("irrigation_action values:", df['irrigation_action'].unique())
    print("irrigation_action value counts:", df['irrigation_action'].value_counts())

    # === Encode irrigation action ===
    # Always fit the encoder on all possible labels
    possible_actions = ["Irrigate", "Reduce Irrigation", "None"]
    action_encoder = LabelEncoder()
    action_encoder.fit(possible_actions)
    # Now transform the actual data
    y_action_encoded = action_encoder.transform(df['irrigation_action'])
    y_action_onehot = to_categorical(y_action_encoded)
    print("Unique irrigation_action values:", np.unique(y_action_encoded))
    print("y_action_train shape:", y_action_onehot.shape)
    if y_action_onehot.shape[1] < 2:
        raise ValueError("Not enough classes in 'irrigation_action' for classification. At least 2 are required.")

    # Save the encoder for use in prediction
    with open(action_encoder_path, 'wb') as f:
        pickle.dump(action_encoder, f)
    print("Irrigation action classes:", action_encoder.classes_)
    print(f"Action encoder saved to: {action_encoder_path}")

    # === Fit and Save Feature Scaler ===
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(X)
    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    print(f"Feature scaler saved to: {feature_scaler_path}")

    # === Fit and Save Moisture Scaler ===
    y_moisture = y_moisture.to_numpy().reshape(-1, 1)
    moisture_scaler = MinMaxScaler()
    moisture_scaler.fit(y_moisture)
    with open(moisture_scaler_path, 'wb') as f:
        pickle.dump(moisture_scaler, f)
    print(f"Moisture scaler saved to: {moisture_scaler_path}")

    # === Scale Inputs ===
    X_scaled = feature_scaler.transform(X)
    y_moisture_scaled = moisture_scaler.transform(y_moisture)

    # === Train/Test Split ===
    X_train, X_test, y_moisture_train, y_moisture_test, y_action_train, y_action_test = train_test_split(
        X_scaled, y_moisture_scaled, y_action_onehot, test_size=0.2, random_state=42, stratify=y_action_encoded
    )

    # === Build Model ===
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(256, activation='relu')(x)

    moisture_output = Dense(1, activation  = 'sigmoid', name='moisture_level')(x)
    action_hidden = Dense(64, activation='relu')(x)
    action_output = Dense(y_action_train.shape[1], activation='sigmoid', name='irrigation_action')(action_hidden)

    model = Model(inputs=input_layer, outputs={
        'moisture_level': moisture_output,
        'irrigation_action': action_output
    })

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'moisture_level': 'mse',
            'irrigation_action': 'categorical_crossentropy'
        },
        metrics={
            'moisture_level': 'mae',
            'irrigation_action': 'accuracy'
        }
    )

    # === Train Model with EarlyStopping ===
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        {
            'moisture_level': y_moisture_train,
            'irrigation_action': y_action_train
        },
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[early_stop]
    )

    # === Save Model ===
    print(f"Saving model to: {final_model_path}")
    try:
        print(f"Attempting to save model to: {final_model_path}")
        model.save(final_model_path)
        if os.path.exists(final_model_path):
            print(f"✅ Model file exists after save: {final_model_path}")
        else:
            print(f"❌ Model file does NOT exist after save: {final_model_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        traceback.print_exc()
        raise
    print(f"All artifacts saved in: {ml_models_dir}")
    print(f"Model: {final_model_path}")
    print(f"Feature scaler: {feature_scaler_path}")
    print(f"Moisture scaler: {moisture_scaler_path}")
    print(f"Action encoder: {action_encoder_path}")

if __name__ == "__main__":
    import sys
    # Usage: python train_model.py <data_path> <model_path>
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <data_path> <model_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    final_model_path = sys.argv[2]
    retrain_and_save_model(data_path, final_model_path)
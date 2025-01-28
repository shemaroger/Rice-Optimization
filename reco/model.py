import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = "swamp_yield_prediction_2000.csv"  # Adjust path as needed
df = pd.read_csv(file_path)

# Define target and features
target = "Risk_Level"
features = df.drop(columns=["Risk_Level", "Recommendation"])

# Encode categorical columns
categorical_cols = ["Drainage", "Crop_Variety"]
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    features[col] = label_encoders[col].fit_transform(features[col])

# Encode the target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[target])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Preprocessing and model pipeline
numeric_features = features.select_dtypes(include=["float64", "int64"]).columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ],
    remainder="passthrough"  # Keep other columns as-is
)

# Model training pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate accuracy
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model_file_path = "yield_prediction_model.joblib"  # Adjust path as needed
joblib.dump(model_pipeline, model_file_path)

print(f"Model saved to {model_file_path}")

# train_lightgbm_model.py
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from your_image_processing_module import extract_features

# Load your labeled dataset
# Assuming you have a CSV file with columns 'Label', 'Feature1', 'Feature2', ..., 'FeatureN'
# Adjust the file path accordingly
dataset_path = r'C:\Users\Yashaswini\OneDrive\Desktop\Project\Indian Sign Language Web Application\features.csv'
df = pd.read_csv(dataset_path)

# Extract features and labels
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a LightGBM classifier
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)  # You can adjust parameters as needed

# Train the LightGBM model
lgbm_model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file (optional)
# Adjust the file path accordingly
model_save_path = r'C:\Users\Yashaswini\OneDrive\Desktop\Project\Indian Sign Language Web Application\lightgbm_model.pkl'
joblib.dump(lgbm_model, model_save_path)
print(f'Model saved to {model_save_path}')































# # train_random_forest_model.py
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from your_image_processing_module import extract_features

# # Load your labeled dataset
# # Assuming you have a CSV file with columns 'Label', 'Feature1', 'Feature2', ..., 'FeatureN'
# # Adjust the file path accordingly
# dataset_path = r'C:\Users\Yashaswini\OneDrive\Desktop\Project\Indian Sign Language Web Application\features.csv'
# df = pd.read_csv(dataset_path)

# # Extract features and labels
# X = df.drop('Label', axis=1)  # Features
# y = df['Label']  # Labels

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize a Random Forest classifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust parameters as needed

# # Train the Random Forest model
# rf_model.fit(X_train, y_train)

# # Evaluate the model on the testing set
# y_pred = rf_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Model Accuracy: {accuracy * 100:.2f}%')

# # Save the trained model to a file (optional)
# # Adjust the file path accordingly
# model_save_path = r'C:\Users\Yashaswini\OneDrive\Desktop\Project\Indian Sign Language Web Application\random_forest_model.pkl'
# joblib.dump(rf_model, model_save_path)
# print(f'Model saved to {model_save_path}')

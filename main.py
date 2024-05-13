import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pprint import pprint

from config import DataConfig, ModelConfig, Settings

# Read the training data
df_train = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.training_data_filename}")

if Settings.dev:
    df_train = df_train.sample(frac=Settings.sample_size)


kf = StratifiedKFold(n_splits=Settings.stratified_k_fold_n_splits, shuffle=True, random_state=Settings.random_state)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='macro')
}

# Separate features and target
X = df_train.drop(columns=[ModelConfig.target_column])
y = df_train[ModelConfig.target_column].str.strip()

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ModelConfig.test_size, random_state=Settings.random_state)

# Encode the target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)  # Transform but do not fit to avoid leakage

# Setting up preprocessing for numerical columns
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
numerical_pipeline = Pipeline([
    ('impute', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler())
])

# Combining preprocessing steps into a single transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('drop', 'drop', ModelConfig.columns_to_drop)  # Explicitly dropping columns
    ],
    remainder='passthrough'  # Include other columns as is
)

# Setting up the classification model
model = DecisionTreeClassifier(random_state=Settings.random_state)

# Creating the full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Fit the model
pipeline.fit(X_train, y_train_encoded)

# Evaluate the model (optional, if you want to see how it performs on unseen data)
pprint(cross_validate(pipeline, X, y, cv=kf, scoring=scoring))
#print(f"Validation Accuracy: {pipeline.score(X_test, y_test_encoded)}")

# Read the unknown (test) data
df_unknown = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.submission_data_filename}")

# Make predictions
y_unknown_pred_encoded = pipeline.predict(df_unknown)
y_unknown_pred = label_encoder.inverse_transform(y_unknown_pred_encoded)

# Format predictions for submission
y_unknown_pred_formatted = [f"{label}      " for label in y_unknown_pred]  # Adding spaces to match the expected format

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': df_unknown['ID'],  # Assuming there is an 'ID' column in the test dataset
    'SpType-ELS': y_unknown_pred_formatted
})

# Save the submission DataFrame to a .csv file
submission_df.to_csv(f"{DataConfig.data_path}/submission_file.csv", index=False)

print("Submission file created successfully!")

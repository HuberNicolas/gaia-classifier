from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

# Creating the full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', None)
])

# Comprehensive parameter grid
param_grid = [
    {'classifier': [DecisionTreeClassifier(random_state=Settings.random_state)],
     'classifier__max_depth': [None, 10, 20, 30],
     'classifier__min_samples_split': [2, 10, 20],
     'classifier__min_samples_leaf': [1, 5, 10]},
    {'classifier': [LogisticRegression(random_state=Settings.random_state)],
     'classifier__C': [0.1, 1.0, 10, 100],
     'classifier__solver': ['liblinear', 'lbfgs']},
    {'classifier': [RandomForestClassifier(random_state=Settings.random_state)],
     'classifier__n_estimators': [100, 200, 300],
     'classifier__max_features': ['auto', 'sqrt', 'log2']},
    {'classifier': [KNeighborsClassifier()],
     'classifier__n_neighbors': [3, 5, 7],
     'classifier__weights': ['uniform', 'distance']},
    {'classifier': [SVC(random_state=Settings.random_state)],
     'classifier__C': [1, 10, 100],
     'classifier__kernel': ['linear', 'rbf']},
    {'classifier': [MLPClassifier(random_state=Settings.random_state)],
     'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
     'classifier__activation': ['tanh', 'relu'],
     'classifier__learning_rate_init': [0.001, 0.01]},
    {'classifier': [XGBClassifier(random_state=Settings.random_state, use_label_encoder=False)],
     'classifier__n_estimators': [100, 200, 300],
     'classifier__max_depth': [3, 6, 9],
     'classifier__learning_rate': [0.01, 0.1, 0.3],
     'classifier__subsample': [0.7, 0.9, 1],
     'classifier__colsample_bytree': [0.7, 0.9, 1]}
]

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', verbose=3, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train_encoded)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Test accuracy
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test_encoded)))

# Use the best estimator for predictions
best_pipeline = grid_search.best_estimator_
results = cross_validate(best_pipeline, X, y, cv=kf, scoring=scoring)
# Evaluate the model (optional, if you want to see how it performs on unseen data)
pprint(results)

# Read the unknown (test) data
df_unknown = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.submission_data_filename}")

# Make predictions
y_unknown_pred_encoded = best_pipeline.predict(df_unknown)
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

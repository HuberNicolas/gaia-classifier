from itertools import cycle
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import DataConfig, EvaluationConfig, ModelConfig, Settings, FeatureSelectionConfig


class FeatureSelector:
    def __init__(self, selection_type, num_features):
        self.selection_type = selection_type
        self.num_features = num_features
        self.model = DecisionTreeClassifier(random_state=0)
        self.selected_indices = None
        self.original_feature_names = None

    def fit(self, X, y):
        # Store the original feature names
        if isinstance(X, pd.DataFrame):
            self.original_feature_names = X.columns.to_numpy()
        else:
            self.original_feature_names = np.arange(X.shape[1])

        self.model.fit(X, y)
        importances = self.model.feature_importances_

        if self.selection_type == 'BEST':
            self.selected_indices = np.argsort(importances)[-self.num_features:]
        elif self.selection_type == 'LEAST':
            self.selected_indices = np.argsort(importances)[:self.num_features]
        else:
            raise ValueError("selection_type should be 'BEST' or 'LEAST'")

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices]
        else:
            return X[:, self.selected_indices]

    def get_selected_features(self):
        return self.original_feature_names[self.selected_indices]

    def get_feature_names(self):
        return self.original_feature_names
# Read the training data
df_train = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.training_data_filename}")

# Speed up during development
if Settings.dev:
    df_train = df_train.sample(frac=Settings.sample_size)

kf = StratifiedKFold(n_splits=Settings.stratified_k_fold_n_splits, shuffle=True, random_state=Settings.random_state)

# Separate features and target
X = df_train.drop(columns=[ModelConfig.target_column])
y = df_train[ModelConfig.target_column].str.strip()

# Define Scoring Metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='macro'),
    'roc_auc': 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'  # Handle binary and multi-class
}

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ModelConfig.test_size,
                                                    random_state=Settings.random_state)

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
    ('feature_selection', FeatureSelector(FeatureSelectionConfig.selection_type, FeatureSelectionConfig.num_features)),
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
     'classifier__colsample_bytree': [0.7, 0.9, 1]},
]

# Add Bagging and Boosting methods
param_grid += [
    {'classifier': [BaggingClassifier(estimator=DecisionTreeClassifier(random_state=Settings.random_state), random_state=Settings.random_state)],
     'classifier__n_estimators': [10, 50, 100],
     'classifier__max_samples': [0.5, 1.0],
     'classifier__max_features': [0.5, 1.0]},
    {'classifier': [AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=Settings.random_state), random_state=Settings.random_state)],
     'classifier__n_estimators': [50, 100, 200],
     'classifier__learning_rate': [0.01, 0.1, 1]},
    {'classifier': [GradientBoostingClassifier(random_state=Settings.random_state)],
     'classifier__n_estimators': [100, 200, 300],
     'classifier__learning_rate': [0.01, 0.1, 0.2],
     'classifier__max_depth': [3, 5, 7]}
]

param_grid = [
    {'classifier': [DecisionTreeClassifier(random_state=Settings.random_state)],
     'classifier__max_depth': [None, 10, 20, 30],
     'classifier__min_samples_split': [2, 10, 20],
     'classifier__min_samples_leaf': [1, 5, 10]},
]

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', verbose=3, n_jobs=-1) # verbose=3 to have detailed output

# Fit GridSearchCV
grid_search.fit(X_train, y_train_encoded)

# Print selected features
def print_columns_at_indices(df, indices):
    try:
        columns_to_print = [df.columns[i] for i in indices]
        print(", ".join(columns_to_print))
    except IndexError:
        print("One or more indices are out of range.")

selected_features = grid_search.best_estimator_.named_steps['feature_selection'].get_selected_features()
print_columns_at_indices(X, selected_features)
print("Selected features:", selected_features)

# Use the best estimator for predictions
best_pipeline = grid_search.best_estimator_
results = cross_validate(best_pipeline, X, y, cv=kf, scoring=scoring)
# Evaluate the model (how it performs on unseen data)
pprint(results)

# Use best model and parameters to calculate confusion matrix
y_pred = best_pipeline.predict(X_test) # best_model = grid_search.best_estimator_
cm = confusion_matrix(y_test_encoded, y_pred)


# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, square=True,
            annot_kws={"size": 16})  # 'fmt' specifies numeric formatting to integers

# Labels, title, and ticks
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
class_labels = label_encoder.classes_
plt.xticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)) + 0.5, class_labels, rotation=0)
plt.savefig(EvaluationConfig.heatmap_plot_path)
print(f"Heat map plot saved as '{EvaluationConfig.heatmap_plot_path}'.")

# Save pipeline
dump(best_pipeline, EvaluationConfig.pipeline_path)
print(f"Pipeline hase been saved to {EvaluationConfig.pipeline_path}")

# Dictionary to hold the best configuration for each classifier type
with open(EvaluationConfig.logfile_path, 'w') as logfile:
    # Write a header for the log file
    logfile.write("Model and Parameters - Scores\n")
    logfile.write("-" * 50 + "\n")

    #  Write confusion matrix
    logfile.write(f"Confusion Matrix:\n{cm}\n")
    logfile.write("-" * 50 + "\n")

    # Iterate over each set of parameters and corresponding results
    for i in range(len(grid_search.cv_results_['params'])):
        # Extract the parameters and scores
        params = grid_search.cv_results_['params'][i]
        mean_test_score = grid_search.cv_results_['mean_test_score'][i]
        std_test_score = grid_search.cv_results_['std_test_score'][i]

        # Create a log entry for this parameter set
        model_details = f"Model: {params['classifier'].__class__.__name__}, Params: {params}, "
        scores_details = f"Mean Score: {mean_test_score:.3f}, Std Dev: {std_test_score:.3f}\n"

        # Write the combined details to the log file
        logfile.write(model_details + scores_details)
print(f"GridSearchCV results have been saved to {EvaluationConfig.logfile_path}")

# Analyze the results to find the best configuration per model type
best_configs = {}
for i in range(len(grid_search.cv_results_['params'])):
    model_type = grid_search.cv_results_['params'][i]['classifier'].__class__.__name__
    model_score = grid_search.cv_results_['mean_test_score'][i]

    if model_type not in best_configs or model_score > best_configs[model_type]['score']:
        best_configs[model_type] = {
            'params': grid_search.cv_results_['params'][i],
            'score': model_score
        }


# Write the best configurations to the file
with open(EvaluationConfig.best_configs_file, 'w') as f:
    for model, config in best_configs.items():
        f.write(f"Best configuration for {model}:\n")
        f.write(f"Parameters: {config['params']}\n")
        f.write(f"Score: {config['score']:.3f}\n")
        f.write("-" * 50 + "\n")

print(f"Best configurations have been saved to {EvaluationConfig.best_configs_file}")

# Best parameters and score
print("Best parameters overall:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Test accuracy
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test_encoded)))


# Prepare to plot ROC curves
y_prob = best_pipeline.predict_proba(X_test)
if len(np.unique(y)) > 2:  # Multi-class case
    # We don't need this case.
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    n_classes = y_test_binarized.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot ROC curve
        plt.plot(fpr[i], tpr[i], color=next(colors), lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
else:  # Binary case
    fpr, tpr, _ = roc_curve(y_test_encoded, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = {0:0.2f})'.format(roc_auc))

# Plotting the diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(EvaluationConfig.ROC_plot_path)
print(f"ROC curves plot saved as '{EvaluationConfig.ROC_plot_path}'.")



# Read the unknown (test) data
df_unknown = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.submission_data_filename}")

# Make predictions
y_unknown_pred_encoded = best_pipeline.predict(df_unknown)
y_unknown_pred = label_encoder.inverse_transform(y_unknown_pred_encoded)

# Format predictions for submission
y_unknown_pred_formatted = [f"{label}      " for label in y_unknown_pred]  # Adding spaces to match the expected format

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': df_unknown['ID'],
    'SpType-ELS': y_unknown_pred_formatted
})

# Save the submission DataFrame to a .csv file
submission_df.to_csv(f"{DataConfig.data_path}/submission_file.csv", index=False)

print("Submission file created successfully!")

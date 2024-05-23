import os
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import DataConfig, EvaluationConfig, ModelConfig, Settings

param_grid = [
    {'classifier': [DecisionTreeClassifier(random_state=Settings.random_state)],
     'classifier__max_depth': [None, 10, 20, 30],
     'classifier__min_samples_split': [2, 10, 20],
     'classifier__min_samples_leaf': [1, 5, 10]},
    {'classifier': [LogisticRegression(random_state=Settings.random_state)],
     'classifier__C': [0.1, 1.0, 10],
     'classifier__solver': ['liblinear', 'lbfgs']},
    {'classifier': [RandomForestClassifier(random_state=Settings.random_state)],
     'classifier__n_estimators': [200, 300],
     'classifier__max_features': ['sqrt', 'log2']},
    {'classifier': [KNeighborsClassifier()],
     'classifier__n_neighbors': [1, 3, 5],
     'classifier__weights': ['uniform', 'distance']},
    {'classifier': [SVC(max_iter=1000, random_state=Settings.random_state)],
     'classifier__C': [1, 2],
     'classifier__kernel': ['rbf', 'sigmoid']},
    {'classifier': [MLPClassifier(random_state=Settings.random_state)],
     'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
     'classifier__activation': ['tanh', 'relu'],
     'classifier__learning_rate_init': [0.01, 0.1]},
    {'classifier': [XGBClassifier(random_state=Settings.random_state, use_label_encoder=False)],
     'classifier__n_estimators': [100, 300],
     'classifier__max_depth': [1, 2],
     'classifier__learning_rate': [0.1, 0.3],
     'classifier__subsample': [0.7, 1],
     'classifier__colsample_bytree': [0.7, 1]},
]

# Add Bagging and Boosting methods
param_grid += [
    {'classifier': [BaggingClassifier(estimator=DecisionTreeClassifier(random_state=Settings.random_state),
                                      random_state=Settings.random_state)],
     'classifier__n_estimators': [10, 50, 100],
     'classifier__max_samples': [0.5, 1.0],
     'classifier__max_features': [0.5, 1.0]},
    {'classifier': [
        AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=Settings.random_state),
                           random_state=Settings.random_state)],
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.1, 1]},
    {'classifier': [GradientBoostingClassifier(random_state=Settings.random_state)],
     'classifier__n_estimators': [10, 100],
     'classifier__learning_rate': [0.1, 0.3],
     'classifier__max_depth': [3, 5]}
]

# Define Scoring Metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='macro'),
    'roc_auc': 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'  # Handle binary and multi-class
}


# Ensure the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Read the training data

selected_features = None
for config in ModelConfig.configs:
    df_train = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.training_data_filename}")
    selected_features = config['features']
    config_nr = config['config_nr']

    selected_features.append(ModelConfig.target_column)
    print(config['features'])

    df_train = df_train[selected_features]
    print(df_train.columns)

    # Speed up during development
    if Settings.dev:
        df_train = df_train.sample(frac=Settings.sample_size)

    kf = StratifiedKFold(n_splits=Settings.stratified_k_fold_n_splits, shuffle=True, random_state=Settings.random_state)

    # Separate features and target
    X = df_train.drop(columns=[ModelConfig.target_column])
    y = df_train[ModelConfig.target_column].str.strip()

    # Encode the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=ModelConfig.test_size,
                                                        random_state=Settings.random_state)

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
            # ('drop', 'drop', ModelConfig.columns_to_drop)  # Explicitly dropping columns
        ],
        remainder='passthrough'
    )

    # Creating the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', None)
    ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', verbose=3, n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Use the best estimator for predictions
    best_pipeline = grid_search.best_estimator_
    results = cross_validate(best_pipeline, X, y_encoded, cv=kf, scoring=scoring)
    # Evaluate the model (how it performs on unseen data)
    pprint(results)

    # Use best model and parameters to calculate confusion matrix
    y_pred = best_pipeline.predict(X_test)  # best_model = grid_search.best_estimator_
    cm = confusion_matrix(y_test, y_pred)

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
    heatmap_path = f"{config_nr}/{EvaluationConfig.heatmap_plot_path}"
    ensure_dir(heatmap_path)
    plt.savefig(heatmap_path)
    print(f"Heat map plot saved as '{heatmap_path}'.")

    # Save pipeline
    pipeline_path = f"{config_nr}/{EvaluationConfig.pipeline_path}"
    ensure_dir(pipeline_path)
    dump(best_pipeline, pipeline_path)
    print(f"Pipeline has been saved to {pipeline_path}")

    # Dictionary to hold the best configuration for each classifier type
    logfile_path = f"{config_nr}/{EvaluationConfig.logfile_path}"
    ensure_dir(logfile_path)
    with open(logfile_path, 'w') as logfile:
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

    best_configs_file_path = f"{config_nr}/{EvaluationConfig.best_configs_file}"
    ensure_dir(best_configs_file_path)
    # Write the best configurations to the file
    with open(best_configs_file_path, 'w') as f:
        for model, config in best_configs.items():
            f.write(f"Best configuration for {model}:\n")
            f.write(f"Parameters: {config['params']}\n")
            f.write(f"Score: {config['score']:.3f}\n")
            f.write("-" * 50 + "\n")

    print(f"Best configurations have been saved to {best_configs_file_path}")

    # Best parameters and score
    print("Best parameters overall:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Test accuracy
    print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

    # Prepare to plot ROC curves
    y_prob = best_pipeline.predict_proba(X_test)
    if len(np.unique(y_encoded)) > 2:  # Multi-class case
        # We don't need this case.
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_encoded))
        n_classes = y_test_binarized.shape[1]
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i])
            # Plot ROC curve
            plt.plot(fpr[i], tpr[i], color=next(colors), lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    else:  # Binary case
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
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

    roc_plot_path = f"{config_nr}/{EvaluationConfig.ROC_plot_path}"
    ensure_dir(roc_plot_path)
    plt.savefig(roc_plot_path)
    print(f"ROC curves plot saved as '{roc_plot_path}'.")

    # Read the unknown (test) data
    df_unknown = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.submission_data_filename}")

    # Make predictions
    y_unknown_pred_encoded = best_pipeline.predict(df_unknown)
    y_unknown_pred = label_encoder.inverse_transform(y_unknown_pred_encoded)

    # Format predictions for submission
    y_unknown_pred_formatted = [f"{label}      " for label in
                                y_unknown_pred]  # Adding spaces to match the expected format

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'ID': df_unknown['ID'],
        'SpType-ELS': y_unknown_pred_formatted
    })

    submission_file_path = f"{config_nr}/{DataConfig.data_path}/submission_file.csv"
    ensure_dir(submission_file_path)
    # Save the submission DataFrame to a .csv file
    submission_df.to_csv(submission_file_path, index=False)

    print("Submission file created successfully!")

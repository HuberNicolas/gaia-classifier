import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt

from config import DataConfig, EvaluationConfig, ModelConfig, Settings


def load_data():
    # Read the training data
    df = pd.read_csv(f"{DataConfig.data_path}/{DataConfig.training_data_filename}")
    return df


def preprocess_data(df):
    # Remove specified columns
    columns_to_drop = ['ID', 'Unnamed: 0', 'Source']
    X = df.drop(columns=columns_to_drop + ['SpType-ELS'])
    y = df['SpType-ELS'].str.strip()

    # Define numerical features
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Create a preprocessing pipeline
    numerical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features)
        ],
        remainder='passthrough'  # Include other columns as is
    )

    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, numerical_features


def fit_decision_tree(X_preprocessed, y):
    # Fit a Decision Tree to get feature importances
    initial_tree = DecisionTreeClassifier(random_state=42)
    initial_tree.fit(X_preprocessed, y)
    return initial_tree


def get_feature_importance_df(tree, feature_names):
    # Get feature importances
    importances = tree.feature_importances_

    # Create a DataFrame to hold feature names and their importance
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    return feature_importance_df


def calculate_actual_importance(feature_importance_df, selected_features):
    actual_importance = feature_importance_df[feature_importance_df['feature'].isin(selected_features)][
        'importance'].sum()
    return actual_importance


def generate_configurations(feature_importance_df, feature_lists):
    configurations = []

    for idx, feature_list in enumerate(feature_lists):
        actual_importance = calculate_actual_importance(feature_importance_df, feature_list)

        configuration = {
            'config_nr': idx,
            'importance': actual_importance,
            'features': feature_list
        }
        configurations.append(configuration)

    return configurations

def generate_configurations__with_thresholds(feature_importance_df, thresholds):
    cumulative_importance = feature_importance_df['importance'].cumsum()
    configurations = []

    for idx, threshold in enumerate(thresholds):
        selected_features = feature_importance_df[cumulative_importance <= threshold]['feature'].tolist()
        actual_importance = cumulative_importance.iloc[len(selected_features) - 1] if selected_features else 0
        # In case no feature meets the exact threshold, include the next highest importance feature
        if not selected_features:
            selected_features = feature_importance_df.iloc[:1]['feature'].tolist()
            actual_importance = cumulative_importance.iloc[0]

        configuration = {
            'config_nr': idx,
            'importance': actual_importance,
            'features': selected_features
        }
        configurations.append(configuration)

    return configurations


def main():
    # Load and preprocess data
    df = load_data()
    X_preprocessed, y, numerical_features = preprocess_data(df)

    # Fit decision tree
    initial_tree = fit_decision_tree(X_preprocessed, y)

    # Get feature importances DataFrame
    feature_names = numerical_features + [col for col in
                                          df.drop(columns=['ID', 'Unnamed: 0', 'Source', 'SpType-ELS']).columns if
                                          col not in numerical_features]
    feature_importance_df = get_feature_importance_df(initial_tree, feature_names)

    # Define the lists of features for configurations
    feature_lists = [
        feature_names,
        ['Teff', 'GRVSmag', 'DE_ICRS', 'RA_ICRS', 'Dist'],
        ['Teff', 'GRVSmag', 'DE_ICRS'],
        ['Teff'],
        ['GRVSmag']
    ]

    # Generate configurations
    configurations = generate_configurations(feature_importance_df, feature_lists)

    for config in configurations:
        print(config)

    # Visualization of feature importances with concrete numbers on the right
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    # Annotate bars with importance values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    plt.show()

    # Visualization of cumulative importances
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=range(1, len(feature_importance_df) + 1), y='cumulative_importance', data=feature_importance_df,
                 marker='o')
    plt.title('Cumulative Feature Importances')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.show()


if __name__ == "__main__":
    main()

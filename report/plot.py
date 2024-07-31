import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = './data/full_data/'
# Function to load and preprocess model data
def load_and_preprocess_model_data(filepath, model_name):
    df = pd.read_csv(filepath, index_col=0)
    df['model'] = model_name
    return df

# Load data for each model
model_1 = load_and_preprocess_model_data(f'{PATH}model_1_scores.csv', 'Model 1')
model_2 = load_and_preprocess_model_data(f'{PATH}model_2_scores.csv', 'Model 2')
model_3 = load_and_preprocess_model_data(f'{PATH}model_3_scores.csv', 'Model 3')
model_4 = load_and_preprocess_model_data(f'{PATH}model_4_scores.csv', 'Model 4')

# Combine data into a single DataFrame
combined_df = pd.concat([model_1, model_2, model_3, model_4])

# Filter out rows used for calculating averages and standard deviations
metrics_df = combined_df.loc[combined_df.index != 'avg']

# List of metrics to compare
metrics = ['fit_time', 'test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_roc_auc']

# Plotting the average scores
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y=metric, data=metrics_df)
    plt.title(f'Average {metric.replace("_", " ").title()} Comparison')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel('Model')

    # Use logarithmic scale for fit_time
    if metric == 'fit_time':
        plt.yscale('log')

    plt.show()

# Plotting box plots for score distributions
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y=metric, data=metrics_df)
    plt.title(f'{metric.replace("_", " ").title()} Distribution Comparison')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel('Model')

    # Use logarithmic scale for fit_time
    if metric == 'fit_time':
        plt.yscale('log')

    plt.show()

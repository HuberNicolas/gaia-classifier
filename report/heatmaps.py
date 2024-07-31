import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Confusion matrices for 50% data
confusion_matrices_50 = [
    np.array([[7948, 0], [467, 6439]]),
    np.array([[7788, 314], [1, 6751]]),
    np.array([[8084, 38], [31, 6701]]),
    np.array([[8040, 27], [22, 6765]])
]

# Confusion matrices for 100% data
confusion_matrices_100 = [
    np.array([[15997, 0], [849, 12862]]),
    np.array([[15392, 605], [0, 13711]]),
    np.array([[15925, 72], [47, 13664]]),
    np.array([[15950, 47], [35, 13676]])
]

def calculate_scores(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = accuracy_score([0]*tn + [0]*fp + [1]*fn + [1]*tp, [0]*(tn+fn) + [1]*(fp+tp))
    precision = precision_score([0]*tn + [0]*fp + [1]*fn + [1]*tp, [0]*(tn+fn) + [1]*(fp+tp))
    recall = recall_score([0]*tn + [0]*fp + [1]*fn + [1]*tp, [0]*(tn+fn) + [1]*(fp+tp))
    f1 = f1_score([0]*tn + [0]*fp + [1]*fn + [1]*tp, [0]*(tn+fn) + [1]*(fp+tp))
    specificity = tn / (tn + fp)
    return accuracy, precision, recall, f1, specificity

def calculate_percentage_matrix(conf_matrix):
    return conf_matrix / conf_matrix.sum() * 100

def create_combined_matrix(conf_matrix):
    percentage_matrix = calculate_percentage_matrix(conf_matrix)
    combined_matrix = np.empty(conf_matrix.shape, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            combined_matrix[i, j] = f'{conf_matrix[i, j]} ({percentage_matrix[i, j]:.2f}%)'
    return combined_matrix

# Function to create a plot with four subplots
def create_heatmap_plot(confusion_matrices, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        combined_matrix = create_combined_matrix(confusion_matrices[i])
        sns.heatmap(confusion_matrices[i], annot=combined_matrix, fmt='', cmap="YlGnBu", ax=ax, cbar=False, annot_kws={"size": 10})
        ax.set_title(f'Model {i+1} (Total samples: {confusion_matrices[i].sum()})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

# Calculate and print scores
def print_scores(confusion_matrices, data_percentage):
    print(f"\nScores for {data_percentage}% data:")
    for i, cm in enumerate(confusion_matrices):
        accuracy, precision, recall, f1, specificity = calculate_scores(cm)
        print(f"Model {i+1}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Specificity: {specificity:.4f}")

# Create and display plots
create_heatmap_plot(confusion_matrices_50, 'Confusion Matrices - 50% Data')
create_heatmap_plot(confusion_matrices_100, 'Confusion Matrices - 100% Data')

# Print scores
print_scores(confusion_matrices_50, 50)
print_scores(confusion_matrices_100, 100)

# Show plots
plt.show()

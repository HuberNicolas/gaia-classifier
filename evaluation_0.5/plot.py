import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the JSON file
json_file_path = 'combined_model_performance_data.json'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Assume each set contains data for 9 models
sets = [df[i:i + 9] for i in range(0, len(df), 9)]

# Create the plot
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 20))

for i, ax in enumerate(axes.flatten()):
    sns.barplot(x='model', y='score', data=sets[i], palette='viridis', ax=ax)
    ax.set_title(f'Data (Sub)set {i + 1}')
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy Scores')
    ax.set_xticklabels(sets[i]['model'], rotation=45, ha='right')

    # Adding the actual value labels on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    # Adjust the y-axis limits to add some padding
    ax.set_ylim(0, max(sets[i]['score']) * 1.1)

plt.tight_layout()
plt.show()

import pandas as pd

data = {
    "Subset": ["50% Data - Subset 1", "50% Data - Subset 2", "50% Data - Subset 3", "50% Data - Subset 4",
               "100% Data - Subset 1", "100% Data - Subset 2", "100% Data - Subset 3", "100% Data - Subset 4"],
    "DTC": [0.970, 0.978, 0.994, 0.995, 0.970, 0.978, 0.995, 0.995],
    "LR": [0.538, 0.976, 0.991, 0.993, 0.539, 0.976, 0.991, 0.993],
    "RFC": [0.944, 0.969, 0.996, 0.996, 0.944, 0.973, 0.996, 0.997],
    "KNC": [0.969, 0.976, 0.993, 0.989, 0.969, 0.977, 0.994, 0.990],
    "SVC": [0.476, 0.968, 0.995, 0.995, 0.741, 0.936, 0.994, 0.995],
    "MLPC": [0.969, 0.978, 0.995, 0.996, 0.969, 0.978, 0.995, 0.997],
    "XGBC": [0.967, 0.977, 0.996, 0.997, 0.967, 0.977, 0.996, 0.997],
    "BC": [0.966, 0.974, 0.995, 0.996, 0.966, 0.976, 0.996, 0.997],
    "ABC": [0.970, 0.978, 0.995, 0.996, 0.970, 0.978, 0.995, 0.996],
    "GBC": [0.970, 0.978, 0.996, 0.997, 0.970, 0.978, 0.995, 0.997]
}

df = pd.DataFrame(data)
import seaborn as sns
df.set_index("Subset", inplace=True)

# Calculate averages per classifier
classifier_averages = df.mean()

# Calculate averages per subset
subset_averages = df.mean(axis=1)

# Display the averages
classifier_averages = classifier_averages.to_frame(name='Average Score').reset_index()
subset_averages = subset_averages.to_frame(name='Average Score').reset_index()

print(classifier_averages)
print(subset_averages)

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 8))
heatmap = sns.heatmap(df, annot=True, cmap="coolwarm", vmin=0.5, vmax=1, linewidths=.5, fmt=".4f")
heatmap.set_title('Model Scores Heatmap')

plt.show()
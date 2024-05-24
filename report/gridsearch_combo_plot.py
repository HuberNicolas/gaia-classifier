import matplotlib.pyplot as plt
import seaborn as sns

# Define the parameter grid
param_grid = [
    {'classifier': ['DecisionTreeClassifier'],
     'classifier__max_depth': [None, 10, 20, 30],
     'classifier__min_samples_split': [2, 10, 20],
     'classifier__min_samples_leaf': [1, 5, 10]},
    {'classifier': ['LogisticRegression'],
     'classifier__C': [0.1, 1.0, 10],
     'classifier__solver': ['liblinear', 'lbfgs']},
    {'classifier': ['RandomForestClassifier'],
     'classifier__n_estimators': [200, 300],
     'classifier__max_features': ['sqrt', 'log2']},
    {'classifier': ['KNeighborsClassifier'],
     'classifier__n_neighbors': [1, 3, 5],
     'classifier__weights': ['uniform', 'distance']},
    {'classifier': ['SVC'],
     'classifier__C': [1, 2],
     'classifier__kernel': ['rbf', 'sigmoid']},
    {'classifier': ['MLPClassifier'],
     'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
     'classifier__activation': ['tanh', 'relu'],
     'classifier__learning_rate_init': [0.01, 0.1]},
    {'classifier': ['XGBClassifier'],
     'classifier__n_estimators': [100, 300],
     'classifier__max_depth': [1, 2],
     'classifier__learning_rate': [0.1, 0.3],
     'classifier__subsample': [0.7, 1],
     'classifier__colsample_bytree': [0.7, 1]},
    {'classifier': ['BaggingClassifier'],
     'classifier__n_estimators': [10, 50, 100],
     'classifier__max_samples': [0.5, 1.0],
     'classifier__max_features': [0.5, 1.0]},
    {'classifier': ['AdaBoostClassifier'],
     'classifier__n_estimators': [50, 100],
     'classifier__learning_rate': [0.1, 1]},
    {'classifier': ['GradientBoostingClassifier'],
     'classifier__n_estimators': [10, 100],
     'classifier__learning_rate': [0.1, 0.3],
     'classifier__max_depth': [3, 5]}
]

# Calculate the number of combinations for each classifier
def calculate_combinations(grid):
    combinations = 1
    for key in grid:
        if key != 'classifier':
            combinations *= len(grid[key])
    return combinations

# Get the number of combinations for each classifier
classifiers = []
combinations = []

for grid in param_grid:
    classifiers.append(grid['classifier'][0])
    combinations.append(calculate_combinations(grid))

# Create the bar chart with Seaborn in a vertical layout
plt.figure(figsize=(10, 11))
barplot = sns.barplot(y=combinations, x=classifiers, palette='viridis')
barplot.set_ylabel('Number of Combinations')
barplot.set_xlabel('Classifier')
plt.xticks(rotation=30)
barplot.set_title('Number of Grid Search Combinations per Classifier')

# Add the number of combinations at the end of the bars
for index, value in enumerate(combinations):
    barplot.text(index, value, f' {value}', ha='center', va='bottom')

# Calculate the total number of combinations
total_combinations = sum(combinations)

# Create the donut chart data
donut_data = [value / total_combinations for value in combinations]
# Use the same color palette as the bar chart
colors = sns.color_palette('viridis', len(classifiers))

# Create the donut chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(donut_data, labels=classifiers, autopct='%1.1f%%', startangle=140, colors=colors, pctdistance=0.85)

# Adjust text properties
for text in texts:
    text.set_fontsize(10)
for autotext in autotexts:
    autotext.set_fontsize(10)

# Create the donut hole
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
ax.axis('equal')
plt.title('Proportion of Combinations per Classifier')
plt.tight_layout()

plt.show()

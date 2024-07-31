from joblib import load

# Load the model
model = load('../evaluation_1.0/e_4_25f/evaluation/gaia_pipeline.joblib')

print(model)

# 50%
# DecisionTreeClassifier('max_leaf_nodes': None,  'min_samples_leaf': 10, 'min_samples_split': 2, )])
# AdaBoostClassifier(estimator=DecisionTreeClassifier(n_estimators = 50, ), learning_rate=0.1))
# XGBClassifier(colsample_bytree 0.7, learning_rate=0.1, max_depth=2, n_estimators=300, subsample 0.7
# XGBClassifier(colsample_bytree 1, learning_rate=0.3, max_depth=2, n_estimators=300,  subsample 1


# 100%

# DecisionTreeClassifier('max_leaf_nodes': None,  'min_samples_leaf': 10, 'min_samples_split': 2, ))])
# AdaBoostClassifier(estimator=DecisionTreeClassifier(n_estimators = 50, , learning_rate=0.1,)
# XGBClassifier(colsample_bytree 1, learning_rate=0.1, max_depth=2, n_estimators=300, subsample 0.7
# XGBClassifier(colsample_bytree 0.7, learning_rate=0.3, max_depth=2, n_estimators=300,  subsample 1


# Access the classifier step in the pipeline
classifier = model.named_steps['classifier']

# Get the name of the classifier
classifier_name = classifier.__class__.__name__
print(classifier_name)
# Get the parameters of the classifier
classifier_params = classifier.get_params()


for k,v in classifier_params.items():
    if k in ['n_estimators', 'max_depth',  'learning_rate', 'subsample', 'colsample_bytree']:
        print(k, v)


print(classifier_params)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def XGBoost_model(X_train, y_train, sample_weights):
    model = XGBClassifier(
        # sets the parameters for the main model, using most successful parameters that have been found from bayesian, and grid testing
        n_estimators=100,  # amount of trees
        max_depth=30,  # depth of each tree
        learning_rate=0.23930193875497338,  # tree contribution
        subsample=0.8,  # the fraction of training data used per tree, this is used to reduce overfitting
        colsample_bytree=0.2,  # fraction of features used per tree, also for overfitting
        device='cuda',
        tree_method='hist',
        verbose=3,
        enable_categorical=True
    )

    print("training")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    return model, le

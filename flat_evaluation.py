import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

def flat_evaluate(X_test, y_test, model, encoder):
    y_pred_encoded = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_encoded.astype(int))

    XGBoost_classif_rep = classification_report(y_test, y_pred, digits=4)
    print("\n classicication report for standard XGBoost model: ")
    print(XGBoost_classif_rep)



    return y_pred

def flat_xgb_confusion(y_test, y_pred, dataset, label_col):
    labels = sorted(dataset[label_col].unique())

    cm = confusion_matrix(y_test, y_pred, labels = labels)

    sns.heatmap(cm, annot=True, fmt='d',cmap= 'Blues', xticklabels=labels, yticklabels=labels)#show's heatmap of what the model classified.
    plt.show()


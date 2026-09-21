from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def binary_conf_mat(binary_true, binary_predictions):   #-creates a confusion matrix for binary classification layer

    confusion_mat = confusion_matrix(binary_true, binary_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign' ,'attack'])
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title('Binary Confusion Matrix')
    plt.show()

def family_evaluation(family_model, family_le, X_test, y_family_test, y_family_string_test): #calculates the accuracy of classifying into families
    #this is based on accurate data being passed into the family classifier, rather than possible outliers passing through the binary classifier.
    y_family_string_test = np.array(y_family_string_test)

    attack_mask = (y_family_string_test != "Benign")#removes any beign data from the test section as the hierarchy doesn't check for these at the family level
    X_test_attack = X_test[attack_mask]
    y_true_attack = y_family_string_test[attack_mask]


    y_family_pred_encoded = family_model.predict(X_test_attack)

    y_family_pred = family_le.inverse_transform(y_family_pred_encoded.astype(int))

    print("classification report for family classification")
    print(classification_report(y_true_attack, y_family_pred,digits=4))

    return y_family_pred, attack_mask

def family_confusion_mat(y_family_string_test, y_family_pred, attack_mask, dataset):
    y_family_string_test = np.array(y_family_string_test)
    y_true_attack = y_family_string_test[attack_mask]

    labels = sorted([i for i in dataset["label_family"].unique() if i != "benign"])

    confusion_mat = confusion_matrix(y_true_attack, y_family_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Family Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def specific_predictions(family_model, family_le, specific_models, specific_encoders, X_test, y_family_string_test, attack_mask, y_specific_test):

    #generates specific attack type predictions by routing each attack record to the corresponding model from the layer 2 prediction.

    y_family_string_test = np.array(y_family_string_test)
    n = len(y_family_string_test)

    y_specific_pred_full = np.full(n, "NO_PRED", dtype=object)
    y_specific_true_full = np.full(n, "NO_PRED", dtype=object)

    X_attack = X_test[attack_mask]
    y_family_true_attack = y_family_string_test[attack_mask]

    #get family layer predictions to route each record to the correct model
    y_family_pred_encoded = family_model.predict(X_attack)
    y_family_pred_strings = family_le.inverse_transform(y_family_pred_encoded.astype(int))


    attack_indices = np.where(attack_mask)[0]


    #run specific model for each family
    for family_key, specific_model in specific_models.items():
        encoder = specific_encoders[family_key]

        family_pred_mask = (y_family_pred_strings == family_key)
        if not np.any(family_pred_mask):
            continue

        X_family = X_attack[family_pred_mask]
        y_specific_pred_encoded = specific_model.predict(X_family)
        y_specific_pred = encoder.inverse_transform(y_specific_pred_encoded.astype(int))

        family_indices = attack_indices[family_pred_mask]
        y_specific_pred_full[family_indices] =y_specific_pred



    y_specific_true_full[attack_mask] = y_specific_test[attack_mask]

    return y_specific_pred_full, y_specific_true_full

def pipeline_evaluation(binary_true, binary_pred, y_family_true, y_family_pred, attack_mask, family_le, y_specific_true, y_specific_pred_full):
    #evaluates the full pipeline, displaying results from start to finish, as if a record passes throug, taking into account any error propagation
    binary_true = np.array(binary_true)
    y_family_true = np.array(y_family_true)
    n = len(binary_true)

    y_family_pred_full = np.full(n, "NO_PRED", dtype=object)
    y_family_pred_full[attack_mask] = y_family_pred

    # Compute per-record pipeline correctness
    results = []
    for i in range(n):
        true_binary = binary_true[i]
        pred_binary = binary_pred[i]
        true_family = y_family_true[i]

        l1_correct = (pred_binary == true_binary)

        if true_binary == "Benign":
            # only checks benign records
            pipeline_correct = l1_correct
        else:
            # Attack records: L1 AND L2 must both be correct
            l2_correct = (y_family_pred_full[i] == true_family)

            if y_specific_true is not None and y_specific_pred_full is not None:
                l3_correct = (y_specific_pred_full[i] == y_specific_true[i])
                pipeline_correct = l1_correct and l2_correct and l3_correct
            else:
                pipeline_correct = l1_correct and l2_correct

        results.append({
            "l1_correct": l1_correct,
            "pipeline_correct": pipeline_correct,
            "true_binary": true_binary
        })

    df = pd.DataFrame(results)

    # --- Summary ---
    total = len(df)
    pipeline_acc = df["pipeline_correct"].sum() / total

    benign_mask = df["true_binary"] == "Benign"
    attack_mask_df = ~benign_mask

    benign_acc = df[benign_mask]["pipeline_correct"].mean()
    attack_acc = df[attack_mask_df]["pipeline_correct"].mean()


    print("\n l1 binary classificaiton report")
    print(classification_report(binary_true, binary_pred, digits=4))#displays classification report of the binary classifier

    print("L2 Family Classification Report")
    y_family_true_attack = y_family_true[attack_mask]
    family_labels = sorted([l for l in np.unique(y_family_true_attack) if l != "Benign"])
    print(classification_report(y_family_true_attack, y_family_pred, labels=family_labels, digits=4))
    #the classificaiton report for the 2nd layer of the model


    # L3 classification report  -   finds the classification report of the final layer
    if y_specific_true is not None and y_specific_pred_full is not None:
        print("L3 Specific Classification Report")
        # Only evaluate records that got routed to a specific model
        routed_mask = (y_specific_pred_full != "NO_PRED")
        y_specific_true_routed = y_specific_true[routed_mask]
        y_specific_pred_routed = y_specific_pred_full[routed_mask]
        print(classification_report(y_specific_true_routed, y_specific_pred_routed, digits=4))

        print("L3 Report (removing unsupported classes, <100)")
        # Get labels with enough support
        from collections import Counter
        support_counts = Counter(y_specific_true_routed)
        sufficient_labels = [l for l, count in support_counts.items() if count > 100]
        sufficient_labels = sorted(sufficient_labels)

        print(classification_report(
            y_specific_true_routed,
            y_specific_pred_routed,
            labels=sufficient_labels,
            digits=4
        ))
    # Full pipeline classification report
    #creates a classification report for the complete hierarachy, from start to finish, including any errors from earlier models
    if y_specific_true is not None and y_specific_pred_full is not None:
        print("Full Pipeline End-to-End Classification Report")
        # Build full true and pred label arrays at the most specific level
        y_pipeline_true = np.where(binary_true == "Benign", "Benign", y_specific_true)
        y_pipeline_pred = np.where(binary_pred == "Benign", "Benign", y_specific_pred_full)

        # Only include records that were routed (exclude NO_PRED)
        valid_mask = (y_pipeline_pred != "NO_PRED")
        print(classification_report(
            y_pipeline_true[valid_mask],
            y_pipeline_pred[valid_mask],
            digits=4
        ))
    return df
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from preprocessing import load_data, preprocessing, split_data, datascaling, sample_weights
from hierarchical_model import hierarchical_models
from hierarchical_evaluation import binary_conf_mat, family_evaluation, family_confusion_mat, pipeline_evaluation, specific_predictions
from flat_XGBoost_model import XGBoost_model
from flat_evaluation import flat_evaluate, flat_xgb_confusion

#---- set these values to true or false depending if execution of either is wanted.
hierarchical = True

flat_xgboost = True

#assign this to the location of the dataset

path = "C:\\uni\\diss work\\Blakeley_25241281_CW2_Artefact\\cicddos2019/*.csv"
dataset = load_data(path)

print(dataset.shape)
label_type = dataset[' Label'].value_counts()
print(label_type)

dataset, y_binary, y_family, y_family_string, y_specific, X, binary_classes, family_classes, label_col=preprocessing(dataset)

X_train, X_test, y_binary_train, y_binary_test, y_specific_train, y_specific_test, y_family_train, y_family_test, y_family_string_train, y_family_string_test = split_data(X, y_binary, y_specific, y_family, y_family_string)
print("post preprocessing train shape:\n")
print(X_train.shape)
print("post preprocessing test shape:\n")
print(X_test.shape)


def hierarchical_processes(X_train, X_test, hierarchical_models):

    X_train, X_test = datascaling(X_train, X_test)



    sample_weights_binary = sample_weights(y_binary_train)

    start = time.time()
    binary_model, family_model, family_le, specific_models, specific_encoders = hierarchical_models(X_train, X_test, y_binary_train, y_family_train,
                                                                 y_specific_train, y_family_string_train,
                                                                 sample_weights_binary)
    end = time.time()
    Total_time = end-start
    print("the time taken for the hierarchical approach is ", Total_time)
    # hierarchical evaluation


    #binary evaluation
    train_binary_pred = binary_model.predict(X_train)
    test_binary_pred = binary_model.predict(X_test)
    print("binary model - train f1: ", f1_score(y_binary_train, train_binary_pred))
    print("binary model - test f1: ", f1_score(y_binary_test, test_binary_pred))
    binary_conf_mat(y_binary_test, test_binary_pred)

    #family evaluation
    y_family_pred, attack_mask = family_evaluation(family_model, family_le, X_test, y_family_test, y_family_string_test)
    family_confusion_mat(y_family_string_test, y_family_pred, attack_mask, dataset)


    binary_pred_string = np.where(test_binary_pred == 0, "Attack", "Benign")

    y_binary_string_test = np.where(np.array(y_family_string_test) == "Benign", "Benign", "Attack")


    y_specific_pred_full, y_specific_true_full = specific_predictions(family_model, family_le, specific_models, specific_encoders, X_test, y_family_string_test, attack_mask, y_specific_test)

    pipeline_df = pipeline_evaluation(
        binary_true=y_binary_string_test,
        binary_pred=binary_pred_string,
        y_family_true=y_family_string_test,
        y_family_pred=y_family_pred,
        attack_mask=attack_mask,
        family_le=family_le,
        y_specific_true = y_specific_true_full,
        y_specific_pred_full = y_specific_pred_full
    )



#----standard XGBoost model
def flat_xgb(X, Y_specific):
    X_train, X_test, y_binary_train, y_binary_test, y_specific_train, y_specific_test, y_family_train, y_family_test, y_family_string_train, y_family_string_test = split_data(X, y_binary, y_specific, y_family, y_family_string)




    X_train, X_test = datascaling(X_train, X_test)

    sample_weights_XGBoost = sample_weights(y_specific_train)


    start = time.time()
    flat_XGBoost, flat_encoder = XGBoost_model(X_train, y_specific_train, sample_weights_XGBoost)
    end = time.time()
    Total_time = end - start
    print("the time taken for the flat approach is ", Total_time)

    xgb_y_pred = flat_evaluate(X_test, y_specific_test, flat_XGBoost, flat_encoder)

    flat_xgb_confusion(y_specific_test, xgb_y_pred, dataset, label_col)


while hierarchical == True:
    hierarchical_processes(X_train, X_test, hierarchical_models)
    hierarchical = False





while flat_xgboost == True:
    flat_xgb(X, y_specific)
    flat_xgboost = False


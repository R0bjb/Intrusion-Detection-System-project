import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier



def hierarchical_models (X_train, X_test, y_binary_train, y_family_train, y_specific_train, y_family_string_train, binary_weight):

    #--- binary model
    binary_model = XGBClassifier(device='cuda', objective="binary:logistic")

    print("training the first model\n")
    binary_model.fit(X_train, y_binary_train, sample_weight=binary_weight)

    # ---- family model

    #removes any benign data to train solely on attack families
    attack_mask_train = (y_family_string_train != "benign")

    #create the train values with only attack data
    X_train_attack = X_train[attack_mask_train]
    y_family_train_attack = y_family_string_train[attack_mask_train]

    family_le = LabelEncoder()#store the encoder and encode the labels so the XGBoost algorithm can execute it, and the testing data can be encoded the same way
    y_family_train_encoded = family_le.fit_transform(y_family_train_attack)

    n_families = len(np.unique(y_family_train_encoded))

    family_model = XGBClassifier(device='cuda', objective="multi:softmax", num_class=n_families)

    sample_weight_family = compute_sample_weight(class_weight='balanced', y=y_family_train_encoded)

    print("training the second model \n")
    #train the family layer model, storing the results.
    family_model.fit(X_train_attack, y_family_train_encoded, sample_weight=sample_weight_family)

    y_family_pred_encoded = family_model.predict(X_test)

    specific_models = {}
    specific_encoders = {}
    #-----      layer 3


    # finds the next unique family, trains a model specifically on that family,
    # storing in the dictionary with the family name as the key

    y_family_string_train = np.array(y_family_string_train)
    for family_name in np.unique(y_family_string_train):
        if family_name == "Benign":
            continue

        family_mask = (y_family_string_train == family_name)

        # slice the dataset to include only relevant attacks from attack families.
        X_family = X_train[family_mask]
        y_family_slice = y_specific_train[family_mask]
        # encodes the specific slice locally
        le = LabelEncoder()
        y_family_encoded = le.fit_transform(y_family_slice)

        n_classes = len(np.unique(y_specific_train[family_mask]))

        specific_model = XGBClassifier(device='cuda', objective='multi:softmax', num_class=n_classes)
        sample_weight_specific = compute_sample_weight(class_weight='balanced', y=y_family_encoded)
        specific_model.fit(X_family, y_family_encoded, sample_weight = sample_weight_specific)

        specific_models[family_name] = specific_model
        specific_encoders[family_name] = le  # saves the encoder to decode predictions

        print(f"trained specific model for: {family_name} with classes: {le.classes_} ")



    return binary_model, family_model, family_le, specific_models, specific_encoders
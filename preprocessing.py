import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings('ignore')

family_map = {
    # TCP based reflection attacks
    "DrDoS_MSSQL": "TCP_Based",
    "MSSQL": "TCP_Based",

    # TCP/UDP reflection attacks
    "DrDoS_DNS": "TCP/UDP_Based",


    "DrDoS_LDAP": "TCP/UDP_Based",
    "LDAP": "TCP/UDP_Based",
    "DrDoS_NetBIOS": "TCP/UDP_Based",
    "NetBIOS": "TCP/UDP_Based",
    "Portmap": "TCP/UDP_Based",
    "DrDoS_SNMP": "TCP/UDP_Based",


    # UDP reflection attacks
    "DrDoS_NTP": "UDP_Based_reflection",
    "TFTP": "UDP_Based_reflection",

    "UDP": "UDP_Based_reflection",
    "DrDoS_UDP": "UDP_Based_reflection",

    # TCP Exploitation attacks
    "Syn": "TCP_Based_exploitation",
    # UDP Exploitation attacks

    "UDP-lag": "UDP_Based_exploitation",
    "UDPLag": "UDP_Based_exploitation",
    "WebDDoS": "UDP_Based_exploitation",

    # benign
    "BENIGN": "Benign"
}

updated_family_map ={   #   used within testing but wasn't beneficial

    "DrDoS_MSSQL": "TCP_Based_DrDoS",
    "MSSQL": "TCP_Based",

    "DrDoS_DNS": "DNS_Based", # large enough to be alone

    #seperates the LDAP varients from other attacks
    "DrDoS_LDAP": "DrDoS_LDAP_Based",
    "LDAP": "LDAP_Based",

    #netBIOS varients combined
    "DrDoS_NetBIOS": "NetBIOS_Based_DrDoS",
    "NetBIOS": "NetBIOS_Based",

    # SNMP and Portmap were less concerning so can be kept together
    "DrDoS_SNMP": "SNMP_Portmap",
    "Portmap": "SNMP_Portmap",


    "DrDoS_NTP": "UDP_Based_reflection",
    "TFTP": "UDP_Based_reflection",
    "UDP": "UDP_Based_reflection",

    "DrDoS_UDP": "UDP_DrDoS",  # This struggled amongst the other UDP_based_reflection, so seperating may benefit

    # TCP Exploitation attacks
    "Syn": "TCP_Based_exploitation",
    # UDP Exploitation attacks

    "UDP-lag": "UDP_Based_exploitation",
    "UDPLag": "UDP_Based_exploitation",
    "WebDDoS": "UDP_Based_exploitation",

    # benign
    "BENIGN": "Benign"

}

label_col = "Label"


def preprocessing(dataset):     #-- remove empty spaces, unusable features, empty records, factorize variables into integers, split X and y (feature columns and label columns)
    dataset.columns = dataset.columns.str.strip()
    dataset = dataset.drop(
        columns=['Unnamed: 0', "Flow ID", 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol',
                 'Timestamp', 'Inbound', 'SimillarHTTP'])


    #remove records with empty values
    dataset = dataset.dropna()
    print(dataset[label_col].unique())
    print("\nhere is the shape after removing n/a values\n")
    print(dataset.shape)

    X = dataset.drop(columns=[label_col])
    y_specific = dataset[label_col]

    dataset["label_binary"] = np.where(dataset[label_col] == "BENIGN", "Benign", "Attack")

    y_binary, binary_classes = pd.factorize(dataset["label_binary"])

    #add a column with the family label to train the family layer
    dataset["label_family"] = dataset[label_col].map(family_map)
    dataset = dataset.dropna(subset=['label_family'])

    y_family_string = dataset["label_family"]  # stores the raw strings before factorizing
    y_family, family_classes = pd.factorize(dataset["label_family"])

    print(len(y_binary))
    print(len(y_family))
    print(len(y_specific))
    print(len(X))

    return dataset, y_binary, y_family, y_family_string, y_specific, X, binary_classes,family_classes, label_col

def load_data(path):    #combines the files in the CICDDoS2019 dataset folder
    all_files = glob.glob(path)
    combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    return combined

def split_data(X, y_binary, y_specific, y_family, y_family_string): # seperates the test/train data, to be used with each level of the hierarchy


    X_train, X_test, y_binary_train, y_binary_test, y_specific_train, y_specific_test, y_family_train, y_family_test, y_family_string_train, y_family_string_test = train_test_split(
        X, y_binary, y_specific, y_family, y_family_string, test_size=0.2, random_state=42, stratify=y_binary)


    return (X_train, X_test, y_binary_train, y_binary_test, y_specific_train, y_specific_test,
            y_family_train, y_family_test, y_family_string_train, y_family_string_test)

def datascaling(X_train, X_test):
    #   Sclae data entries so they can be used, instead of being computationally infinite as long as numpy was considered.

    # replacing infinite values with nan temporarily
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)


    # fill nan values with the median for each column required, using the median of the train split to avoid data leakage
    column_median = X_train.median()  # median of training data
    X_train = X_train.fillna(column_median)
    X_test = X_test.fillna(column_median)

    # scale the values within the columns, using the same scaling for test set to avoid data leakage
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def sample_weights(train):
    sample_weight = compute_sample_weight(class_weight='balanced', y=train)

    return sample_weight




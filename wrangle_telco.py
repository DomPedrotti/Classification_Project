import pandas as pd
import numpy as np
import seaborn as sns
from env import get_db_url
import split_scale
import explore

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def clean_telco_data():
    #pull data
    query = '''
    select * 
    from customers as cust
    join `internet_service_types` as net
    on cust.`internet_service_type_id` = net.internet_service_type_id
    join `contract_types` as cont
    on cust.`contract_type_id` = cont.`contract_type_id`
    join payment_types as pmt
    using(`payment_type_id`);
    '''
    churn_df = pd.read_sql(query, get_db_url('telco_churn'))
    
    #for duplicate columns
    churn_df = churn_df.loc[:,~churn_df.columns.duplicated()]

    #for duplicat rows
    churn_df = churn_df.drop_duplicates()
    
    #drop redundant collumns
    churn_df = (churn_df.drop('contract_type_id', axis = 1)
                        .drop('internet_service_type_id', axis = 1)
                        .drop('payment_type_id', axis = 1))
    
    #change 'no internets' and no phones to just no
    churn_df.replace('No internet service', 'No', inplace=True)
    churn_df.replace('No phone service', 'No', inplace=True)
    
    # change to float
    churn_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    churn_df = churn_df.dropna(axis=0)
    churn_df.total_charges = churn_df.total_charges.astype(float)

    #get features and target
    target = 'churn'
    features = churn_df.columns.tolist()
    features.remove(target)
    features.remove('customer_id')

    #change churn column to boolean
    churn_df['churn'] = LabelEncoder().fit_transform(churn_df['churn']).astype(bool)
    churn_df.senior_citizen = churn_df.senior_citizen.astype(bool)
    
    #create new e-check collumn
    churn_df['e_check'] = churn_df.payment_type == 'Electronic check'

    #remove total_charges and senior citizens
    features.remove('total_charges')
    
    #remove collumns with little effect on tenure
    features.remove('gender')
    features.remove('phone_service')
    features.remove('payment_type')
    features.remove('contract_type')
    features.remove('internet_service_type')
    features.remove('multiple_lines')

    #encode yes no collumns
    for i in features:
        if churn_df[i].unique().tolist() == ['No', 'Yes'] or churn_df[i].unique().tolist() == ['Yes', 'No']:
            churn_df[i] = churn_df[i] == 'Yes'

    #one hot encode collumns
    churn_df = (churn_df.join(pd.get_dummies(churn_df.contract_type), on= churn_df.index)
                        .join(pd.get_dummies(churn_df.internet_service_type), on = churn_df.index))
    
    #add to features
    new_features = pd.get_dummies(churn_df.contract_type).columns.tolist()
    new_features += pd.get_dummies(churn_df.internet_service_type).columns.tolist()
    features += new_features
    
    #split data
    train, test = split_scale.split_my_data(churn_df, stratify=churn_df.churn)
    return train, test, features, target
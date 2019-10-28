import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from env import get_db_url

### Create split_scale.py that will contain the functions that follow. Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train df scaled, test df scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe. Be sure to set a random state where applicable for reproducibility!

def split_my_data(df, train_pct = .80, stratify = None):
    '''
    split_my_data(df, train_pct = .80):

    takes in dataframe as input, and splits it on percent train_pct
    
    train_pct controlls where the split cut off ends, default to an 80%-20% train and test split

    returns training and test, respectively:
    return train, test
    '''
    train, test = train_test_split(df, train_size = train_pct, random_state = 123, stratify = stratify )
    return train, test

def standard_scaler(train, test):
    '''
    def standard_scaler(train, test):
    
    reveives train and test dataframes and returns their standard scalar transformations along with their scalar object for reference later
    '''
    scaler_object = StandardScaler(copy=True, 
                                   with_mean=True, 
                                   with_std=True).fit(train) 
    scaled_train = apply_object(train, scaler_object)
    scaled_test  = apply_object(test,  scaler_object)
    return scaler_object, scaled_train, scaled_test

def scale_inverse(df, scaler_object):
    '''
    scale_inverse(df, scaler_object):

    receives dataframe associated scalar object and returns un transformed dataframe
    '''
    inverse_transformation = pd.DataFrame(scaler_object.inverse_transform(df), columns=df.columns.values).set_index([df.index.values])

    return inverse_transformation

def uniform_scaler(train, test, quantiles = 100):
    '''
    uniform_scaler(train, test, quantiles = 100):

    receives train and test data frames as arguments
    optional argument for number of quantile applied to transformation

    creates uniform scalar object

    returns scalar object, and dataframe transformations
    '''
    scaler_object = QuantileTransformer(n_quantiles=quantiles, 
                                 output_distribution='uniform', 
                                 random_state=123, 
                                 copy=True).fit(train)
    scaled_train = apply_object(train, scaler_object)
    scaled_test  = apply_object(test,  scaler_object)
    return scaler_object, scaled_train, scaled_test

def gaussian_scaler(train, test ,method = 'yeo-johnson'):
    '''
    gaussian_scaler(train, test ,method = 'yeo-johnson'):
    
    receives train and test data frames as arguments
    optional argument for Gaussian transformation method (yeo-johnson or box-cox)

    creates gaussian scalar object

    returns scalar object, and dataframe transformations
    '''
    scaler_object = PowerTransformer(method = method, 
                                     standardize = False, 
                                     copy = True).fit(train)
    scaled_train = apply_object(train, scaler_object)
    scaled_test  = apply_object(test,  scaler_object)
    return scaler_object, scaled_train, scaled_test
    
def min_max_scaler(train, test):
    '''
    min_max_scaler(train, test):
    
    receives train and test data frames as arguments

    creates minimum-maximum scalar object

    returns scalar object, and dataframe transformations
    '''
    scaler_object = MinMaxScaler(copy=True, 
                                 feature_range=(0,1)).fit(train)
    scaled_train = apply_object(train, scaler_object)
    scaled_test  = apply_object(test,  scaler_object)
    return scaler_object, scaled_train, scaled_test

def iqr_robust_scaler(train, test, quantiles = (25.0, 75.0)):
    '''
    iqr_robust_scaler(train, test, quantiles = (25.0, 75.0)):
    
    receives train and test data frames as arguments
    optional argument for quantile range

    creates interquartile range scalar object

    returns scalar object, and dataframe transformations
    '''
    scaler_object = RobustScaler(quantile_range = quantiles, 
                                 copy = True, with_centering=True, 
                                 with_scaling = True).fit(train)
    scaled_train = apply_object(train, scaler_object)
    scaled_test  = apply_object(test,  scaler_object)
    return scaler_object, scaled_train, scaled_test

def apply_object(x, scaler_object):
    ''' 
    apply_object(x, scaler_object):

    applys object to dataframe for scalar transformations
    '''
    return pd.DataFrame(scaler_object.transform(x), 
                        columns=x.columns.values).set_index([x.index.values])


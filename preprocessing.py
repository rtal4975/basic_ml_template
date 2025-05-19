'''
Tuck all preprocessing and experimentation here
'''
### globals
import os, sys
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def normalize(data, output_dir):
    ''''
    Going to use sklearn StandardScaler to normalize
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        "The standard score of a sample x is calculated as:
            z = (x - u) / s
        where u is the mean of the training samples or zero if
        with_mean=False, and s is the standard deviation of the
        training samples or one if with_std=False."
    
    Going to allow hand-picking which columns we normalize so we
    could avoid doing this on categorical data
    '''
    scaler = StandardScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data, copy=True)

    # Save the scaler to a file
    if data.shape[1] == 1:
        # then assume target data
        output_file = os.path.join(output_dir,'scaler_y.pkl')
    else:
        output_file = os.path.join(output_dir,'scaler_X.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(scaler, file)
    
    return normalized_data


def feature_reduction_PCA(data, to_dims, output_dir):
    '''
    Using PCA
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

        "Linear dimensionality reduction using Singular
        Value Decomposition of the data to project it to
        a lower dimensional space. The input data is
        centered but not scaled for each feature before
        applying the SVD."
    '''
    pca = PCA(n_components=to_dims)
    pca.fit(data)
    reduced_data = pca.transform(data)
    
    # Save the PCA mapping to a file
    output_file = os.path.join(output_dir,'pca_map_dim.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(pca, file)
    
    return reduced_data


def split_data(X, y, output_dir, train_frac=0.75, seed=9):
    np.random.seed(seed)
    num_rows = len(X)
    assert len(X) == len(y) # idk jic sue me

    train_count = math.floor(num_rows*train_frac)
    shuffled_indices = np.random.choice(np.arange(num_rows), num_rows, replace=False)
    train_indices = shuffled_indices[:train_count]
    test_indices = shuffled_indices[train_count:]

    output_file = os.path.join(output_dir,'train_indices.npy')
    np.save(output_file, train_indices)
    
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def main(df, reduce_to_dims, output_dir, target_colname, categ_colnames=[]):
    # would be convenient for later if target_colname is a list
    if isinstance(target_colname, str):
        target_colname = [target_colname]

    numeric_colnames = [col for col in df.columns if col not in categ_colnames+target_colname]

    # rearrange for convenience so that categ columns are last
    X = df[numeric_colnames+categ_colnames].values
    y = df[target_colname].values # <- since target_colname is list, shape should be (nrows,1) so don't need to reshape

    X = normalize(X, output_dir)
    y = normalize(y, output_dir)
    X_numeric = feature_reduction_PCA(X[:,:len(numeric_colnames)], reduce_to_dims, output_dir)
    X = np.hstack([X_numeric, X[:,-1*len(categ_colnames):]])
    X_train, y_train, X_test, y_test = split_data(X, y, output_dir)

    return X_train, y_train, X_test, y_test


'''
Use this to run the model on a new set of data to 'evaluate' for the target.
Run like ...
    $ conda activate
    $ python evaluate.py TODO
'''
### globals
import os, sys
import glob
import pickle
import numpy as np
import pandas as pd

### locals


def load_model(model_dir):
    model = ...
    # TODO
    return model


def load_preprocessing(output_dir, pca_tag):
    scaler_X_file = glob.glob(os.path.join(output_dir, 'scaler_X.pkl'))[0]
    assert os.path.exists(scaler_X_file)
    with open(scaler_X_file, 'rb') as f:
        scaler_X = pickle.load(f)

    scaler_y_file = glob.glob(os.path.join(output_dir, 'scaler_y.pkl'))[0]
    assert os.path.exists(scaler_y_file)
    with open(scaler_y_file, 'rb') as f:
        scaler_y = pickle.load(f)

    pca_file = glob.glob(os.path.join(output_dir, 'pca_map_%s.pkl' % pca_tag))[0]
    assert os.path.exists(pca_file)
    with open(pca_file, 'rb') as f:
        pca = pickle.load(f)
    
    return scaler_X, scaler_y, pca


def predict(data_filepath, output_dir, pca_tag, target_colname=None, debug=False):
    '''
    mimic a lot of other code

    assuming column order is maintained in this data_file vs the one used in training
    '''
    assert os.path.exists(data_filepath)
    df = pd.read_excel(data_filepath)

    ### mimic preprocess.main()
    scaler_X, scaler_y, pca = load_preprocessing(output_dir, pca_tag)
    categ_colnames = ['', '']
    numeric_colnames = [col for col in df.columns if col not in categ_colnames+target_colname]
    X = df[numeric_colnames+categ_colnames].values
    X = scaler_X.transform(X, copy=True) # normalize
    X_numeric = pca.transform(X[:,:len(numeric_colnames)]) # pca
    X = np.hstack([X_numeric, X[:,-1*len(categ_colnames):]])

    ### modelling
    model = load_model(output_dir)

    if target_colname is None:
        # then we want to predict!
        y_pred = model.predict(X, batch_size=None)
        return y_pred
    
    else:
        # then let's evaluate
        y = df[target_colname].values
        y = scaler_y.transform(y, copy=True)
        loss = model.evaluate(X, y, batch_size=None)
        return loss

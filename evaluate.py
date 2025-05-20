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
import tensorflow as tf

### locals


def load_model(output_dir):
    # look for model subdir in output_dir
    poss_model_file = glob.glob(os.path.join(output_dir, "model", "**", "saved_model.pb"))
    if len(poss_model_file) > 1:
        raise Exception("Somehow found %i model .pb files in %s\nExpected only 1." % (len(poss_model_file), output_dir))
    elif len(poss_model_file) == 0:
        raise Exception("Coudln't find model .pb file in %s\nAssuming model dir not saved there after training." % output_dir)
    else:
        model_dir = os.path.dirname(poss_model_file[0])

    model = tf.keras.models.load_model(model_dir)
    # pretty sure we have to recompile, so just gonna copy/paste compile ftn from modelling.build()
    # but if we change the method there, then we gotta remember to change here too!
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae',])
    
    return model


def load_preprocessing(output_dir):
    scaler_X_file = glob.glob(os.path.join(output_dir, 'scaler_X.pkl'))[0]
    assert os.path.exists(scaler_X_file)
    with open(scaler_X_file, 'rb') as f:
        scaler_X = pickle.load(f)

    scaler_y_file = glob.glob(os.path.join(output_dir, 'scaler_y.pkl'))[0]
    if os.path.exists(scaler_y_file):
        with open(scaler_y_file, 'rb') as f:
            scaler_y = pickle.load(f)
    else:
        # then we didn't want to scale our target variable
        scaler_y = None

    pca_file = glob.glob(os.path.join(output_dir, 'pca_map_dim.pkl'))[0]
    assert os.path.exists(pca_file)
    with open(pca_file, 'rb') as f:
        pca = pickle.load(f)
    
    return scaler_X, scaler_y, pca


def main(data_filepath, output_dir, target_colname=None, debug=False):
    '''
    mimic a lot of other code

    assuming column order is maintained in this data_file vs the one used in training
    '''
    df = pd.read_excel(data_filepath)

    ### mimic preprocess.main()
    scaler_X, scaler_y, pca = load_preprocessing(output_dir)
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
        if scaler_y is not None:
            y = scaler_y.transform(y, copy=True)
        loss = model.evaluate(X, y, batch_size=None)
        return loss



if __name__=="__main__":
    '''
    conda activate py38_tf
    python evaluate.py -f [FILEPATH] -o [OUTPUTDIR]
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath",
                        type = str,
                        required = True,
                        help = "Filepath to data.")
    parser.add_argument("-o", "--output_dir",
                        type = str,
                        required = True,
                        help = "Filepath to model folder and training metadata.")
    parser.add_argument("-d", "--debug",
                        action = 'store_true',
                        default = False,
                        help = "Flag to go in debug-mode.")
    args = parser.parse_args()
    assert os.path.exists(args.filepath)
    assert os.path.exists(args.output_dir)
    assert os.path.isdir(args.output_dir)

    target_colname = None
    model_output = main(args.filepath, args.output_dir, target_colname, args.debug)

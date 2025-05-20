'''
Put it all together here. Run like
    $ conda activate ____
    $ python main.py

This will load data, preprocess data, build model, and train the model, then save the model.
Need to also save any normalization or PCA mapping
'''
### globals
import os, sys
import time, datetime
import numpy as np
import pandas as pd
import pdb

### locals
import preprocessing as my_pp
import modelling as my_nn


def get_data(filepath=None, use_fake=True, debug=False):
    try:
        if use_fake:
            import fake_data
            df = fake_data.get_data()
        
        elif filepath is None:
            err_msg = "Hey Dingus, you forgot to fillout the filepath."
            raise Exception(err_msg)
        
        elif not os.path.exists(filepath):
            err_msg = "Hey Dingus, you didn't give a valid filepath:\n%s" % filepath
            raise Exception(err_msg)
            
        else:
            ####################
            # TODO - FILL ME IN
            df = pd.read_excel(...)
            ####################
    
    except Exception as err:
        if debug:
            print(err)
            pdb.set_trace()
        else:
            raise Exception(err)
    
    return df


def main(output_parent_dir, data_filepath, use_fake, debug):
    df = get_data(data_filepath, use_fake, debug)
    if use_fake:
        target_colname = 'Target'
        categ_colnames = ['Feature 17', 'Feature 18']
        reduce_to_dims_range = [8,5]
    else:
        ####################
        # TODO - FILL ME IN
        target_colname = ''
        categ_colnames = ['', '']
        reduce_to_dims_range = [] # After experimenting a range, pick single value and store as list like [##,]
        ####################
    
    modelling_history = []
    for reduce_to_dims in reduce_to_dims_range:
        model_name = "VI_Prediction"
        output_child_dir = os.path.join(output_parent_dir, "dim%02i" % reduce_to_dims)
        os.makedirs(output_child_dir)
        print("\nModel %s training with dims reduced to %s with PCA:" % (model_name, reduce_to_dims))

        # Preprocess
        X_train, y_train, X_test, y_test = my_pp.main(df, reduce_to_dims, output_child_dir, target_colname, categ_colnames)
        
        # Constants
        epochs = 10
        num_batches = 4
        batch_size = len(X_train)//num_batches
        input_dim = X_train.shape[1]

        # Modelling
        model = my_nn.build(input_dim)
        model, history = my_nn.train(model, X_train, y_train, epochs, batch_size)
        my_nn.save_model(model, output_child_dir, model_name)
        hist_df = my_nn.save_metrics(history, output_child_dir, model_name)
    
        # Performance Tracking
        print("...\nPerformance Results")
        print(hist_df)
        test_loss = model.evaluate(X_test, y_test, batch_size=None, verbose=0) # <- "Returns the loss value & metrics values for the model"
        print("Mean Squared Error & Mean Absolute Error on Test Data:\n\t%s" % test_loss)
        modelling_history.append([history, test_loss])

    return modelling_history


if __name__=="__main__":
    '''
    conda activate py38_tf
    python main.py -f -d
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath",
                        type = str,
                        required = False,
                        default = None,
                        help = "Filepath to data. Don't use for 'fake' data.")
    parser.add_argument("-d", "--debug",
                        action = 'store_true',
                        default = False,
                        help = "Flag to go in debug-mode.")
    parser.add_argument("-z", "--use_fake",
                        action = 'store_true',
                        default = False,
                        help = "Flag to use fake data.")
    args = parser.parse_args()

    output_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join('output', output_tag)
    os.makedirs(output_directory)

    modelling_history = main(output_directory, args.filepath, args.use_fake, args.debug)

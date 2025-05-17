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


def main(output_dir, data_filepath, use_fake, debug):
    df = get_data(data_filepath, use_fake, debug)
    if use_fake:
        target_colname = 'Target'
        categ_colnames = ['Feature 17', 'Feature 18']
    else:
        ####################
        # TODO - FILL ME IN
        target_colname = ''
        categ_colnames = ['', '']
        ####################
    X_train, y_train, X_test, y_test = my_pp.main(df, output_dir, target_colname, categ_colnames)
    
    epochs = 10
    num_batches = 4
    batch_size = len(X_train)//num_batches
    model = my_nn.build()
    model, history = my_nn.train(model, X_train, y_train, epochs, batch_size)
    test_loss = model.evaluate(X_test, y_test, batch_size=None)
    my_nn.save(model, output_dir, model_name="TODO")

    return


if __name__=="__main__":
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
                        help = "Flag to remove drop size of the data.")
    parser.add_argument("-z", "--fake",
                        action = 'store_true',
                        default = False,
                        help = "Flag to use fake data.")
    args = parser.parse_args()

    output_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join('output', output_tag)
    os.makedirs(output_directory)

    main(output_directory, args.filepath, args.use_fake, args.debug)
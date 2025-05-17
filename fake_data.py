'''
Generate fake data to help build out pipeline without private info
'''
import os, sys
import numpy as np
import pandas as pd


def get_data(size=100):
    np.random.seed(9) # set seed for reproducability

    filepath = os.path.join("datasets","Honeywell_FakeDataSpecs.xlsx")
    assert os.path.exists(filepath)

    df = pd.read_excel(filepath)
    df.drop(columns=['Unnamed: 0'], inplace=True)

    fake_data_dict = {}
    for col in df.columns:
        _min = df[col].min()
        _max = df[col].max()
        if (_min==0) and (_max==1):
            fake_data_dict[col] = np.random.choice([0,1], size, replace=True)
        else:
            fake_data_dict[col] = np.random.uniform(_min, _max, size)


    final_df = pd.DataFrame(fake_data_dict)
    return final_df


if __name__=="__main__":
    df = get_data()
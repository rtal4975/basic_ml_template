# Honeywell Consult
Quick consult on how to build out a pipeline to train mixed data with a neural network.

## Setup

Here is how to build out the conda environment with Python 3.__

```
$ conda env create --file environment.yml
$ conda activate py38_tf
$ pip install tensorflow
```

## Run

Here is how to run the code

```
$ conda activate py38_tf
$ python main.py -f [FILEPATH]
```

## Design Choice

### Preprocessing

TODO talk about feature reduction with PCA, why, how, whatever

Talk about normalization, why how etc

copy pasting 


    Going to use sklearn StandardScaler to normalize
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        "The standard score of a sample x is calculated as:
            z = (x - u) / s
        where u is the mean of the training samples or zero if
        with_mean=False, and s is the standard deviation of the
        training samples or one if with_std=False."

and

    Using PCA
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

        "Linear dimensionality reduction using Singular
        Value Decomposition of the data to project it to
        a lower dimensional space. The input data is
        centered but not scaled for each feature before
        applying the SVD."


### NN Architecture

How many layers why how etc


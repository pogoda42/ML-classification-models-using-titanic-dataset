# pylint: disable=invalid-name
"""
Data preprocessing module for the Titanic dataset.
Contains pipelines and functions to transform raw data into model-ready features.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler, OrdinalEncoder

# -----------------------------------------------------------------------------
# Define preprocessing pipeline
# -----------------------------------------------------------------------------


def column_sum(X):
    """Calculate sum of two columns."""
    return X[:, [0]] + X[:, [1]]


def sum_name(function_transformer, feature_names_in):  # pylint: disable=unused-argument
    """Return feature name for summed columns."""
    return ["sum"]  # feature names out


log_pipeline = make_pipeline(
    KNNImputer(),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    MinMaxScaler(feature_range=(0, 1)))

one_hot_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder())

ordinal_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(categories=[['S', 'C', 'Q']]))

kmeans_pipeline = make_pipeline(
    KNNImputer(),
    MinMaxScaler(feature_range=(0, 1)))

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    MinMaxScaler(feature_range=(0, 1)))

sum_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    FunctionTransformer(column_sum, feature_names_out=sum_name),
    MinMaxScaler(feature_range=(0, 1)))

preprocessing = ColumnTransformer([
    ("Relatives", sum_pipeline, ['Parch', 'SibSp']),
    ("Log", log_pipeline, ["Fare"]),
    ("One_hot", one_hot_pipeline, ["Sex"]),
    ("Ordinal", ordinal_pipeline, ["Embarked"]),
    ("Numeric", default_num_pipeline, ['SibSp', 'Parch']),
    ("KNN", kmeans_pipeline, ['Age']),
    ("Pass", "passthrough", ['Pclass', 'Survived'])
],
    remainder="drop"
)


def prepare_data(input_folder: str, output_folder: str) -> None:
    """
    Prepare Titanic dataset by applying preprocessing transformations and splitting into train/dev/test sets.

    Args:
        input_folder: Path to folder containing raw data files
        output_folder: Path to folder where processed data will be saved
    """
    # Load data
    input_data = pd.read_csv(f'{input_folder}/strat_train.csv')
    test_data = pd.read_csv(f'{input_folder}/strat_test.csv')

    train_data, dev_data = train_test_split(
        input_data,
        test_size=0.2,
        stratify=input_data['Pclass'],
        random_state=18
    )

    titanic_test = test_data.copy()

    X_train = preprocessing.fit_transform(train_data)
    X_dev = preprocessing.fit_transform(dev_data)
    X_test = preprocessing.fit_transform(titanic_test)

    titanic_train = pd.DataFrame(
        X_train, columns=preprocessing.get_feature_names_out(), index=train_data.index)
    titanic_dev = pd.DataFrame(
        X_dev, columns=preprocessing.get_feature_names_out(), index=dev_data.index)
    titanic_test = pd.DataFrame(
        X_test, columns=preprocessing.get_feature_names_out(), index=test_data.index)

    # Save the processed data
    titanic_train.to_csv(f'{output_folder}/titanic_train.csv')
    titanic_dev.to_csv(f'{output_folder}/titanic_dev.csv')
    titanic_test.to_csv(f'{output_folder}/titanic_test.csv')


if __name__ == "__main__":
    prepare_data(
        input_folder='data/interim',
        output_folder='data/processed'
    )

from huggingface_hub import hf_hub_url, cached_download
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pytest
import pandas as pd
from typing import Union
import numpy as np
REPO_ID = "julien-c/wine-quality"
FILENAME = "sklearn_model.joblib"


@pytest.fixture
def test_dataset() -> Union[np.ndarray, np.ndarray]:
    data_file = cached_download(hf_hub_url(REPO_ID, "winequality-red.csv"))
    winedf = pd.read_csv(data_file, sep=";")
    X_test = winedf.drop(["quality"], axis=1)
    y_test = winedf["quality"]

    return X_test, y_test

@pytest.fixture
def model() -> sklearn.ensemble._forest.RandomForestClassifier:
    REPO_ID = "julien-c/wine-quality"
    FILENAME = "sklearn_model.joblib"
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    return model


def test_model_inference_types(model, test_dataset):
    assert isinstance(model.predict(test_dataset[0]), np.ndarray)
    #assert isinstance(test_dataset[0], np.ndarray)
    #assert isinstance(test_dataset[1], np.ndarray)

def test_data_types(test_dataset):
    assert isinstance(test_dataset[0], pd.DataFrame)
    #assert isinstance(test_dataset[1], np.ndarray)
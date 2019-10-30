import pandas as pd
import config
from sklearn.externals import joblib

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data

def load_pipeline(*, file_name): 
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

import pathlib


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'
INTERMEDIATE_TESTS = PACKAGE_ROOT / 'tests'

IDV_MASTER = 'idv_master.csv'
IDV_VALUATION = 'valuation.csv'
CARWALE = 'carwale.csv'
POPULARITY = 'popularity.csv'
MARGIN_DIVISION = 'use_final_grid.csv'
TARGET = 'dep_percentage'
DATA_FOR_TRAINING = 'data_for_training.csv'
CLEANED_POPULARITY = 'after_popularity.csv'
LABEL_ENCO_DIC = 'label_en_dic.pkl'
TRAINED_MODEL = 'car_dep_model.pkl'

FEATURES_CARWALE = ['make','model','city','owners','kms_run', 'age', 'Popularity Index',
'ex_showroom_price','fuel_type','transmission','color']

FEATURES_IDV_MASTER = ["id","created","modified","make","model","variant","idv_id","mmv_name"]

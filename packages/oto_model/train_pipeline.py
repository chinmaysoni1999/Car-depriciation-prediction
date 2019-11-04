import pipeline
import pandas as pd
from data_management import load_dataset
from sklearn.model_selection import train_test_split
import config
import lightgbm as lgb
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.externals import joblib

def save_pipeline() -> None:
    """Persist the pipeline."""

    pass


def run_training() -> None:
    """Train the model."""
    idv = load_dataset(file_name=config.IDV_MASTER)
    carwale = load_dataset(file_name=config.CARWALE)
    popularity = load_dataset(file_name=config.POPULARITY)
    valuation = load_dataset(file_name=config.IDV_VALUATION)
    use_final_grid = load_dataset(file_name=config.MARGIN_DIVISION)

    carwale = pipeline.carwale_pp.transform(carwale)
    idv = pipeline.idv_pp.transform(idv)
    carwale_idv_m = pipeline.carwale_idv_merger.transform(carwale,idv)
    data = pipeline.color_city_transform.transform(carwale_idv_m)
    final = pipeline.data_for_pop.transform(data)
    pop = pipeline.popularity_cleaner.transform(popularity)
    final_ = pipeline.data_pop_merger.transform(final,pop)
    final_ = pipeline.ex_showroom_price.transform(final_,valuation,idv)
    final_ = pipeline.outlier_dep.transform(final_,use_final_grid)
    data = pipeline.training_prep.transform(final_)
    save_path = config.TRAINED_MODEL_DIR / 'label_en_dic.pkl'
    joblib.dump(pipeline.training_prep.dic, save_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES_CARWALE],
        data[config.TARGET],
        test_size=0.1,
        random_state=0)

    params = {
    "objective" : "regression",
    "metric" : "mae",
    #"num_leaves" : 800,
    "num_leaves" : 500,
   "learning_rate" : 0.005,
    "bagging_fraction" : 0.6,
    "feature_fraction" : 0.6,
   "bagging_frequency" : 6,
   # "bagging_frequency" : 1,
    "bagging_seed" : 42,
    "verbosity" : -1,
    "seed": 42
    }

    lgb_train_data = lgb.Dataset(X_train, label=y_train)

    model = lgb.train(params, lgb_train_data,
                 num_boost_round= 10000,
                  verbose_eval=500)

    y_pred_lgbm = model.predict(X_test,num_iteration=model.best_iteration)
    score = mean_squared_error(y_test, y_pred_lgbm)
    print(score)

    save_file_name = 'car_dep_model.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    joblib.dump(model, save_path)

    print('saved pipeline')


if __name__ == '__main__':
    run_training()

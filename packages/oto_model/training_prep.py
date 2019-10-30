from pyod.models.iforest import IForest
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

"""Removes multivariate outliers from the dataset"""

class TrainingPrep(BaseEstimator, TransformerMixin):
    """Cleans the data for matching with the popularity dataset"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
            self.dic = None
        else:
            self.variables = variables
            self.dic = None
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, df2: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        le = LabelEncoder()
        df2['mm'] = df2['make'] +' '+ df2['model']
        g_mm_count = df2.groupby(['mm']).count().reset_index()
        mm_more_than_100 = g_mm_count[g_mm_count['make']>100]['mm']
        df2 = df2[df2['mm'].isin(mm_more_than_100)]
        dfn3=df2.copy()
        g1 = dfn3.groupby('mm')
        clf1=IForest(contamination = 0.01)
        flag=[1]

        if 1 in flag:

            dff1 = pd.DataFrame(columns = ['idv_id', 'kms_run', 'owners', 'age', 'Popularity Index','quoted_price', 'outlier','dep_percentage'])

            for idv_id,idv_id_df in g1:
                idv_id_df1 = idv_id_df[['kms_run', 'owners', 'age','quoted_price','dep_percentage']]
                clf1.fit(idv_id_df1)
                y_pred = clf1.predict(idv_id_df1)
                idv_id_df['outlier'] = y_pred.tolist()
                dff1 = pd.concat([dff1,idv_id_df])
            outlier_idv_if_dff1 = set(dff1[dff1['outlier']==1].index)

        df2=df2.drop(outlier_idv_if_dff1)
        df=df2.copy()
        X=df[['make','model','city','variant','owners','kms_run', 'age', 'Popularity Index','ex_showroom_price','fuel_type','transmission','color']]
        categorical_feature_mask = X.dtypes==object
        categorical_cols = X.columns[categorical_feature_mask].tolist()
        self.dic = {}
        for i in categorical_cols:
            X[i] = le.fit_transform(X[i])
            self.dic[i]=dict(zip(le.classes_, le.transform(le.classes_)))
        y=df[['dep_percentage']]
        aa=pd.concat([X,y],axis=1)

        return aa

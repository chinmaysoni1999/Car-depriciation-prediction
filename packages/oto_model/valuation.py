import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

""" Here we are going to calculate the ex-showroom price for the car data collected.
The new idv_valuation data containing prices are depreciated according to given standards to their
original price which merged with idv_master data we can get the ex-showroom price for data"""

class ExshowroomPriceCal(BaseEstimator, TransformerMixin):
    """Cleans the Carwale data for matching with the idv_id."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, df_final:pd.DataFrame, df_valuation1: pd.DataFrame, df_idv1: pd.DataFrame) -> pd.DataFrame:
        df_valuation1['mmv@'] = df_valuation1['make']+df_valuation1['model']+df_valuation1['variant']
        df_idv1['mmv@'] = df_idv1['make']+df_idv1['model']+df_idv1['variant']
        df_valuation1['mmv@'] = df_valuation1['mmv@'].map(lambda x:x.lower())
        df_idv1['mmv@'] = df_idv1['mmv@'].map(lambda x:x.lower())
        df_idv1['created'] = df_idv1['created'].map(lambda x:x.lower())
        df_idv_final = pd.merge(df_idv1,df_valuation1,on=['mmv@'],how='inner')
        df_idv_final['modified_y'] = df_idv_final['modified_y'].map(lambda x:datetime.strptime(x[0:19],"%Y-%m-%d %H:%M:%S"))
        df_idv_final['month_age'] = df_idv_final['modified_y'].map(lambda x: x.month+1 if x.month==1 else x.month)-df_idv_final['month']
        df_idv_final['year_age'] = df_idv_final['modified_y'].map(lambda x: x.year)-df_idv_final['year']
        df_idv_final['year_age'] = df_idv_final['year_age'].map(lambda x:x*12)
        df_idv_final['final_age'] = df_idv_final['year_age'] + df_idv_final['month_age']
        df_idv_final['idv'] = df_idv_final['idv'].astype(int)
        df_below_60 = df_idv_final[df_idv_final.final_age<=60]
        df_below_60.loc[df_below_60.final_age<=6,'ex_showroom_price'] = df_below_60['idv']/0.95
        df_below_60.loc[(df_below_60.final_age>6)&(df_below_60.final_age<=12),'ex_showroom_price'] = df_below_60['idv']/0.85
        df_below_60.loc[(df_below_60.final_age>12)&(df_below_60.final_age<=24),'ex_showroom_price'] = df_below_60['idv']/0.80
        df_below_60.loc[(df_below_60.final_age>24)&(df_below_60.final_age<=36),'ex_showroom_price'] = df_below_60['idv']/0.70
        df_below_60.loc[(df_below_60.final_age>36)&(df_below_60.final_age<=48),'ex_showroom_price'] = df_below_60['idv']/0.60
        df_below_60.loc[(df_below_60.final_age>48)&(df_below_60.final_age<=60),'ex_showroom_price'] = df_below_60['idv']/0.50
        df_below_60.ex_showroom_price = df_below_60.ex_showroom_price.astype(int)
        df_below_60_f = df_below_60[['idv_id','ex_showroom_price','state']]
        df_below_60_f = df_below_60_f.drop_duplicates(['idv_id','state'])
        df_below_60_f.columns = ['idv_id','ex_showroom_price','city1']
        df_above_60 = df_idv_final[df_idv_final.final_age>60]
        l=list(set(df_above_60.idv_id)-set(df_below_60.idv_id))
        t = df_idv_final[df_idv_final.idv_id.isin(l)][['idv_id','final_age','idv','state']]
        t.loc[(t.final_age>60)&(t.final_age<=72),'ex_showroom_price'] = t['idv']/0.45
        t.loc[(t.final_age>72)&(t.final_age<=84),'ex_showroom_price'] = t['idv']/0.40
        t.loc[(t.final_age>84)&(t.final_age<=96),'ex_showroom_price'] = t['idv']/0.365
        t.loc[(t.final_age>96)&(t.final_age<=108),'ex_showroom_price'] = t['idv']/0.328
        t.loc[(t.final_age>108)&(t.final_age<=120),'ex_showroom_price'] = t['idv']/0.30
        t['ex_showroom_price'] = t['ex_showroom_price'].astype(int)
        df_above_60_f=t.drop_duplicates(['idv_id','state'])
        df_above_60_f=df_above_60_f[['idv_id','ex_showroom_price','state']]
        df_above_60_f.columns = ['idv_id','ex_showroom_price','city1']
        idv_price=pd.concat([df_below_60_f,df_above_60_f],ignore_index = True)
        idv_price['city'] = idv_price['city1'].map(lambda x: x.lower())
        idv_price['idv_id']=idv_price['idv_id'].astype('int')
        idv_price.city=idv_price.city.map(lambda x: x.replace('gurgaon','gurugram'))
        df_final=df_final[~df_final.city.str.contains('chennai')]
        df_final=df_final[~df_final.city.str.contains('pune')]
        df_final.idv_id=df_final.idv_id.astype(int)
        df_model_data=pd.merge(df_final,idv_price,on=['idv_id','city'],how='inner')

        return df_model_data

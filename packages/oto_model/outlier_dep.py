import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.stats import zscore

class DepOutier(BaseEstimator, TransformerMixin):
    """Cleans the data for matching with the popularity dataset"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, df: pd.DataFrame,df1: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        df = df[df['quoted_price'].notnull()]
        df = df[df['kms_run'].notnull()]
        df['quoted_price'] = df['quoted_price'].astype(int)
        df['fuel_type'] = df['fuel_type'].astype(str)
        df['quoted_price'] = df['quoted_price'].replace(',','')
        df['kms_run'] = df['kms_run'].astype(str).str.replace(',','')
        df['kms_run'] = df['kms_run'].astype(int)
        df['quoted_price'] = df['quoted_price'].astype(int)
        df['transmission'] = df['transmission'].astype(str)

        def fuel(s):
            l=s.split(' ')
            return(str(l[0]))

        df['fuel_type']=df['fuel_type'].apply(fuel)

        df = df[df['kms_run']>0]

        dfn = df._get_numeric_data()

        df.loc[(df['variant'].str.contains(' mt'))&(df['transmission']=='automatic'),'transmission'] = 'manual'
        df.loc[(df['variant'].str.contains(' at'))&(df['transmission']=='manual'),'transmission'] = 'automatic'
        df.loc[(df['variant'].str.contains(' amt'))&(~df['transmission'].str.contains('auto')),'transmission']='automated manual transmission'

        df['transmission']=df['transmission'].map(lambda x:x.replace('automated manual transmission','automatic'))

        l = [1,2,3,5,4,6]

        df = df[df['owners'].isin(l)]


        df.drop(df.loc[(df['age']==0.0)&(df['owners']>2)].index,inplace=True)

        df.drop(df.loc[(df['age']==1.0)&(df['owners']>3)].index,inplace=True)



        df['check']=stats.zscore(df['ex_showroom_price'])
        df['check1']=stats.zscore(df['quoted_price'])
        df['check2']=stats.zscore(df['kms_run'])

        df=df[(df.check<1.5)&(df.check>-1.5)]
        df=df[(df.check1<1.5)&(df.check1>-1.5)]
        df=df[(df.check2<1.5)&(df.check2>-1.5)]


        df=df[df['ex_showroom_price']>df['quoted_price']]

        df=df.drop_duplicates()
        df1.drop('Unnamed: 0',axis=1,inplace=True)
        df1.columns = ['max_price', 'min_price', 'negotiation', 'margin', 'Popularity Index','percent']

        def c_allotment(n):
            if 199999>n>0:
                return 199999
            elif 299999>n>200000:
                return 299999
            elif 399999>n>300000:
                return 399999
            elif 499999>n>400000:
                return 499999
            elif 599999>n>500000:
                return 599999
            elif 699999>n>600000:
                return 699999
            elif 799999>n>700000:
                return 799999
            elif 899999>n>800000:
                return 899999
            elif 999999>n>900000:
                return 999999
            elif 1199999>n>1000000:
                return 1199999
            elif 1399999>n>1200000:
                return 1399999
            elif 1599999>n>1400000:
                return 1599999
            elif 1899999>n>1600000:
                return 1899999
            elif 2399999>n>1900000:
                return 2399999
            elif 3000000>n>2400000:
                return 3000000
            else:
                return 3000000

        df['max_price'] = df['quoted_price'].apply(c_allotment)

        df['Popularity Index'] = df['Popularity Index'].astype(int)

        dfm = pd.merge(df,df1,how='inner',on=['Popularity Index','max_price'])

        dfm = dfm[['id', 'make', 'model', 'variant', 'city', 'fuel_type', 'transmission',
               'owners', 'quoted_price', 'kms_run', 'color', 'idv_id', 'age',
               'data_source', 'Popularity Index', 'ex_showroom_price','negotiation', 'margin',
               'percent']]

        dfm['real_price'] = dfm['quoted_price']-(dfm['quoted_price']*dfm['percent'])/100

        dfm['dep_percentage'] = ((dfm['ex_showroom_price']-dfm['real_price'])/dfm['ex_showroom_price'])*100

        dfma = dfm[['id', 'make', 'model', 'variant', 'city', 'fuel_type', 'transmission',
               'owners', 'quoted_price', 'kms_run', 'color', 'idv_id', 'age',
               'data_source', 'Popularity Index', 'ex_showroom_price','dep_percentage','percent','negotiation', 'margin']]


        dfma=dfma[(dfma.fuel_type=='petrol')|(dfma.fuel_type=='diesel')|(dfma.fuel_type=='lpg')|(dfma.fuel_type=='cng')]

        dfma=dfma[dfma.transmission!='nan']

        dfma.loc[dfma['age']==0.5,'age']=0.0

        dfma['check4']=stats.zscore(dfma['dep_percentage'])
        dfma=dfma[(dfma.check4<1.5)&(dfma.check4>-1.5)]

        k1 = dfma.groupby('age')

        df_with_z_score_age = pd.DataFrame(columns =['id', 'make', 'model', 'variant', 'city', 'fuel_type', 'transmission',
               'owners', 'quoted_price', 'kms_run', 'color', 'idv_id', 'age',
               'data_source', 'Popularity Index', 'ex_showroom_price',
               'dep_percentage', 'percent','dep_percentage_zs','kms_zs'])

        for age,age_df in k1:
            age_df['dep_percentage_zs'] = zscore(age_df['dep_percentage'])
            age_df['kms_zs'] = zscore(age_df['kms_run'])

            df_with_z_score_age = pd.concat([df_with_z_score_age,age_df])

        df_with_z_score_age = df_with_z_score_age[df_with_z_score_age['dep_percentage_zs']<1.5]
        df_with_z_score_age = df_with_z_score_age[df_with_z_score_age['dep_percentage_zs']>-1.5]
        df_with_z_score_age = df_with_z_score_age[df_with_z_score_age['kms_zs']<1.5]
        df_with_z_score_age = df_with_z_score_age[df_with_z_score_age['kms_zs']>-1.5]

        dfma=df_with_z_score_age.copy()

        dfma['mm'] = dfma['make'] +' '+ dfma['model']

        g_mm_count = dfma.groupby(['mm']).count().reset_index()

        mm_more_than_200 = g_mm_count[g_mm_count['make']>70]['mm']

        dfn3 = dfma[dfma['mm'].isin(mm_more_than_200)]

        return dfn3

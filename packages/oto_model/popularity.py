import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re

"""Incorporating business logic, popularity indexes were prepared for
various make, model and variants which would serve as an addition feature...

Here we are mapping the popularity with the transformed dataset...

The transformed data from the previous transformations are required to undergo some
other specfic transformation for matching with the popularity dataset..
 """

class DataCleanerPop(BaseEstimator, TransformerMixin):
    """Cleans the data for matching with the popularity dataset"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X['make1'] = X['make'].astype(str)
        X['model1'] = X['model'].astype(str)
        X['variant1'] = X['variant'].astype(str)
        X['make1'] = X['make1'].map(lambda x: x.replace('"',''))
        X['model1'] = X['model1'].map(lambda x: x.replace('"',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('"',''))

        #specific corrections
        def clean_string(s):
            p=re.compile('\d')
            iter1  = p.finditer(s)
            indices=[m.start(0) for m in iter1]
            l=list(s)
            for i in indices:
                if(i<len(l)-2):
                    if(l[i+1]=='l'and l[i-1]=='.'):
                        del l[i+1]
            s=''.join(l)
            return s

        X['variant1'] = X['variant1'].map(lambda x: x.replace(' ',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('(',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace(')',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('+','plus'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('-',' '))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('opt','o'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('lpg',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace(' ',''))
        X['variant1'] = X['variant1'].apply(clean_string)
        X['mmv@'] = X['make1']+''+X['model1']+''+X['variant1']
        X['product_description1'] = X['mmv@'].map(lambda x: ''.join(sorted(x)))
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace('-',' '))
        X['product_description1'] = X['product_description1'].map(lambda x:x.strip())

        return X

class PopCleaner(BaseEstimator, TransformerMixin):
    """Cleaning the popularity for matching with the data"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        def clean_string_2(s):
            p=re.compile('\d')
            iter1  = p.finditer(s)
            indices=[m.start(0) for m in iter1]
            l=list(s)
            for i in indices:
                if(i<len(l)-2):
                    if(l[i+1]=='l' and l[i-1]=='.' and ~(l[i+2].isalpha())):
                        del l[i+1]

            s=''.join(l)
            return s


        def string_clean(s):
            if bool(re.match(r"\d\d\d\d", s[-4:]))==True:
                return s[-4:]
            elif bool(re.match(r".*[1-2][09][0-9][0-9]", s))==True:
                pattern = re.compile(r"[2][09][0-9][0-9]")
                matches = pattern.findall(s)
                return matches[0]
            else:
                return '0'

        def string_clean_year_remove(s):
            if bool(re.match(r"\d\d\d\d", s[-4:]))==True:
                k = re.finditer(r"\d\d\d\d", s)
                indices = [m.start(0) for m in k]
                #return s[:indices[0]+1]+s[indices[0]+5:]
                return s[:len(s)-4]

            elif bool(re.match(r".*[1-2][09][0-9][0-9]", s))==True:
                k = re.finditer(r"[1-2][09][0-9][0-9]", s)
                indices = [m.start(0) for m in k]
                return s[:indices[0]]+s[indices[0]+4:]

            else:
                return s


        def clean_string_1(s):
                t = s.find('[')
                t1 = s.find(']')
                if(t==-1):
                    return s
                else:
                    s = s[:t]+s[t1+1:]
                    return s

        X['product_description1'] = X['MMV'].apply(clean_string_1)
        X['product_description1'] = X.product_description1.apply(string_clean_year_remove)
        X['product_description1'] = X['product_description1'].map(lambda x:x.lower())
        X['product_description1'] = X['product_description1'].astype(str)
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace('"',''))
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace('(',''))
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace(')',''))
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace('+','plus'))
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace('-',' '))
        X['product_description1'] = X['product_description1'].map(lambda x: x.replace('opt','o'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('volkswagenventohighlinediesel','volkswagenventohighlinediesel1.5'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('volkswagenventotrendlinediesel','volkswagenventotrendlinediesel1.5'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('hyundaicreta1.6sxdiesel','hyundaicreta1.6sxcrdi'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('hyundaicreta1.6sxodiesel','hyundaicreta1.6sxocrdi'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('hyundaii10era1.11rde2','hyundaii10era1.1irde2'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('renaultduster110psrxz','renaultduster110psrxzdiesel'))
        X['product_description1'] = X['product_description1'].map(lambda x:x.replace('marutisuzukialtok10k10vxi','arutisuzukialtok10vxi'))
        X['product_description1'] = X['product_description1'].apply(clean_string_2)
        X['product_description1'] = X['product_description1'].map(lambda x:''.join(sorted(x)))
        X['Make'] = X['Make'].map(lambda x:x.lower())
        X['Model'] = X['Model'].map(lambda x:x.lower())
        X['product_description1'] = X['product_description1'].map(lambda x: x.strip())
        X.columns = ['Popularity Index','MMV','make1','model1','product_description1']

        return X

class MergePop(BaseEstimator, TransformerMixin):
    """Merging cleaned dataset and popularity dataset"""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, data: pd.DataFrame, popularity:pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        dfm = pd.merge(data,popularity,how='left',on='product_description1')
        dfm_matched = dfm[dfm['Popularity Index'].notnull()]
        dfm_not_matched_id = dfm[dfm['Popularity Index'].isnull()].id
        df_not_matched = data[data['id'].isin(dfm_not_matched_id)]
        df_not_matched_match = pd.merge(df_not_matched,popularity,how='inner',on=['make1','model1'])
        df_not_matched_match = df_not_matched_match.drop_duplicates('id')
        df_not_matched_match.drop_duplicates('id')
        df_not_matched_match.loc[df_not_matched_match['make1']=='bmw','Popularity Index'] = 4
        df_not_matched_match.loc[df_not_matched_match['make1']=='mercedes benz','Popularity Index'] = 4
        df_not_matched_match.loc[df_not_matched_match['make1']=='audi','Popularity Index'] = 4
        dfm_matched = dfm_matched[['id', 'make', 'model', 'variant', 'city', 'fuel_type', 'transmission','owners', 'quoted_price', 'kms_run', 'color', 'idv_id', 'age','data_source','Popularity Index']]
        df_not_matched_match = df_not_matched_match[['id', 'make', 'model', 'variant', 'city', 'fuel_type', 'transmission','owners', 'quoted_price', 'kms_run', 'color', 'idv_id', 'age','data_source','Popularity Index']]
        df = pd.concat([dfm_matched,df_not_matched_match])

        return df

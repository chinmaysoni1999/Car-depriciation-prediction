import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re

class CarwaleCleaner(BaseEstimator, TransformerMixin):
    """Cleans the Carwale data for matching with the idv_id."""

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

        X = X.copy()
        X['mfg_year'] = X['mfg_year'].astype(str).astype(int)
        X = X[X['mfg_year']>=2009]
        X['make1'] = X['make'].map(lambda x: x.lower())
        X['model1'] = X['model'].map(lambda x: x.lower())
        X['variant1'] = X['variant'].map(lambda x: x.lower())
        X['make1'].astype(str)
        X['model1'].astype(str)
        X['variant1'].astype(str)

        #remove elements like [2008-2009] from the string
        def clean_string_1(s):
            t = s.find('[')
            if(t==-1):
                return s
            else:
                s = s[:t-1]
                return s

        X['model1'] = X['model1'].apply(clean_string_1)
        X['variant1'] = X['variant1'].apply(clean_string_1)
        X['variant1'] = X['variant1'].map(lambda x: x.replace('+','plus'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('-',' '))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('opt','o'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('(',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace(')',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('petrol','p'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('diesel','d'))


        #removes 'l' and spaces if alpha after number
        def clean_string_2(s):
            l=list(s)
            p=re.compile('\d')
            iter = p.finditer(s)
            indices=[m.start(0) for m in iter]
            if(len(l)-1 in indices):
                indices.remove(len(l)-1)
            ind=[i+1 for i in indices if l[i+1]=='l' and l[i-1]=='.']
            l1=[l[i] for i in list(range(len(l))) if i not in ind]
            s1=''.join(l1)
            iter  = p.finditer(s1)
            indices=[m.start(0) for m in iter]
            if(len(l1)-1 in indices):
                indices.remove(len(l1)-1)
            ind=[i+1 for i in indices if l1[i+1].isalpha()]
            b=[' ']*len(ind)
            d = dict(zip(ind, b))
            f=[t for k in [(d.get(i),j) for i,j in enumerate(l1)] for t in k if t is not None]
            s=''.join(f)
            return s

        #spaces if aplha before number
        def clean_string_3(s):
            l=list(s)
            p=re.compile('\d')
            iter1  = p.finditer(s)
            indices=[m.start(0) for m in iter1]
            ind=[i for i in indices if l[i-1].isalpha()]
            b=[' ']*len(ind)
            d = dict(zip(ind, b))
            f=[t for k in [(d.get(i),j) for i,j in enumerate(l)] for t in k if t is not None]
            s=''.join(f)
            return s

        X['variant1'] = X['variant1'].apply(clean_string_2)
        X['variant1'] = X['variant1'].apply(clean_string_3)
        X['mmv@'] = X['make1']+' '+X['model1']+' '+X['variant1']

        def sort_string(s):
            t = s.split(' ')
            t.sort()
            v=' '.join(t)
            return v

        X['mmv@'] = X['mmv@'].apply(sort_string)
        X['mmv@'] = X['mmv@'].map(lambda x:x.strip())

        return X

class IDVCarwaleCleaner(BaseEstimator, TransformerMixin):
    """Cleans the X data for matching with the X_id with the carwale data."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'IDVCarwaleCleaner':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        X['make1']=X['make'].astype(str)
        X['model1']=X['model'].astype(str)
        X['variant1']=X['variant'].astype(str)
        X['variant1'] = X['variant1'].map(lambda x: x.replace('+','plus'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('-',' '))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('opt','o'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('(',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace(')',''))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('petrol','p'))
        X['variant1'] = X['variant1'].map(lambda x: x.replace('diesel','d'))

        #removes 'l' spaces e & remove spaces
        def clean_string_2(s):
            p=re.compile('\d')
            iter1  = p.finditer(s)
            indices=[m.start(0) for m in iter1]
            l=list(s)
            for i in indices:
                if(i<len(l)-1):
                    if(l[i+1]=='l'):
                        del l[i+1]
            for i in indices:
                if(i<len(l)-1):
                    if(l[i+1].isalpha()):
                        l.insert(i+1,' ')
            for i in indices:
                if(i<len(l)):
                    if(l[i-1].isalpha()):
                        l.insert(i,' ')

            s=''.join(l)
            return s

        #spaces if aplha before number
        def clean_string_3(s):
            l=list(s)
            p=re.compile('\d')
            iter1  = p.finditer(s)
            indices=[m.start(0) for m in iter1]
            ind=[i for i in indices if l[i-1].isalpha()]
            b=[' ']*len(ind)
            d = dict(zip(ind, b))
            f=[t for k in [(d.get(i),j) for i,j in enumerate(l)] for t in k if t is not None]
            s=''.join(f)
            return s

        X['variant1'] = X['variant1'].apply(clean_string_2)
        X['variant1'] = X['variant1'].apply(clean_string_3)
        X['mmv@'] = X['make1']+' '+X['model1']+' '+X['variant1']

        def sort_string(s):
            t = s.split(' ')
            t.sort()
            v=' '.join(t)
            return v

        X['mmv@'] = X['mmv@'].apply(sort_string)
        X['mmv@'] = X['mmv@'].map(lambda x:x.strip())

        return X

class AddIdvId(BaseEstimator, TransformerMixin):
    """Matching idv_id for data using cleaned idv and carwale data."""

    def __init__(self,variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'AddIdvId':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, carwale: pd.DataFrame, idv: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        carwale = carwale.copy()
        idv = idv.copy()
        X = pd.merge(carwale,idv,on=['mmv@'],how='inner')
        X['mod']=X['modified_x'].map(lambda x: str(x)[:4])
        X['age']= X['mod'].astype(int)-X['mfg_year'].astype(int)
        X['month']=X['modified_x'].map(lambda x: str(x)[5:7]).astype(int)
        X.loc[(X.age==0)&(X['month']>6),'age']= 0.5
        X['data_source']='carwale'
        X = X[['id_x','make_y','model_y','variant_y','city','fuel_type','transmission','owners','quoted_price','kms_run','color','idv_id','age','data_source',]]
        X.columns=['id','make','model','variant','city','fuel_type','transmission','owners','quoted_price','kms_run','color','idv_id','age','data_source']
        X = X.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        X = X.drop_duplicates()

        return X

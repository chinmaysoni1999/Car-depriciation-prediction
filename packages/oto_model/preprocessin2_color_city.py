import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColorCleaner(BaseEstimator, TransformerMixin):
    """Cleans the data with varying shades of colors to their parent colors."""

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

        #some irregular data observed in the dataset
        original_color = ["bronze", "orange", "blue" ,"purple", "star dust", "brown", "silver", "green" ,"red" ,"white" ,"beige" ,"pearl" ,"gold" ,"maroon", "grey" ,"yellow", "black"]

        X.drop(X[X['color']=='???? ??????? ??????'].index,inplace=True)
        X.drop(X[X['color']=='??????? ???'].index,inplace=True)
        X.drop(X[X['color']=='???????? ????'].index,inplace=True)
        X.drop(X[X.color.isnull()].index,inplace=True)

        def color_coding(s):
            original_color = ["bronze", "orange", "blue" ,"purple", "star dust", "brown", "silver", "green" ,"red" ,"white" ,"beige" ,"pearl" ,"gold" ,"maroon", "grey" ,"yellow", "black"]
            for i in original_color:
                if i in s:
                    return i
            return s

        X['color'] = X['color'].apply(color_coding)

        #various specific color mappings
        bronze = ['urban titanium metallic', 'urban titanium' , 'urban titanium metallic', 'urban titanium metallic' , 'bronzo tan']
        bronze1 = ['bronze' for i in bronze]
        dic_bronze = dict(zip(bronze,bronze1))

        blue = ['chill','amethyst royal','Havana','misty lake','atlantis bule','arctic breeze','havanna','bluish titanium','artic','aroyal','sky','cyan']
        blue1 = ['blue' for i in blue]
        dic_blue = dict(zip(blue,blue1))

        purple = ['violet' , 'tryian wine','pink', 'royal orchid' , 'mysterious violet' ,'morello']
        purple1 = ['purple' for i in purple]
        dic_purple = dict(zip(purple,purple1))

        brown = ['bakers chocolate', 'fire brick' , 'rosso brunello' , 'chocolate' ,'mocca' ]
        brown1 = ['brown' for i in brown]
        dic_brown = dict(zip(brown,brown1))

        red = ['tuscan wine','ruby' , 'wine' , 'vitro' , 'megenta', 'burgandy' , 'burgundy' , 't wine']
        red1 = ['red' for i in red]
        dic_red = dict(zip(red,red1))

        white = [ 'fox trote' , 'real earth metallic' , 'saloon' , 'wite' , 'salon' , 'real earth' , 'milk' ]
        white1 = ['white' for i in white]
        dic_white = dict(zip(white,white1))

        beige = ['beize' , 'biege' , 'ashe beige']
        beige1 = ['beige' for i in beige]
        dic_beige = dict(zip(beige,beige1))

        maroon = [ 'meroon' , 'mehroon' ]
        maroon1 = ['maroon' for i in maroon]
        dic_maroon = dict(zip(maroon,maroon1))

        grey  = ['gray', 'champagne mica metallic', 'carbon steel','mettalic', 'modern steel metallic', 'metallic' ,'mettalic' , 'ember gray' , 'chill metallic' , 'premium' , 'azure gray' ,'metallic glistening gray' , 'graphite' , 'metallic magma gray' , 'granite',  'mystic gray'  , 'platanium' , 'heather mist' ]
        grey1 = ['grey' for i in grey]
        dic_grey = dict(zip(grey,grey1))

        yellow  = ['sunlight copper' , 'mushroom', 'metallic sunlight copper' , 'copper' , 'champagne', 'beach' , 's l copper' ,'camel' ,'cream' ,'taxi' ]
        yellow1 = ['yellow' for i in yellow]
        dic_yellow = dict(zip(yellow,yellow1))

        black  = ['dark gray melallic'  , 'carbon' ,'carbon flash' , 'polished metal metallic' , 'jatoba metallic']
        black1 = ['black' for i in black]
        dic_black = dict(zip(black,black1))

        dic_final_color = {**dic_bronze,**dic_blue,**dic_purple,**dic_brown,**dic_red,**dic_white,**dic_beige,**dic_maroon,**dic_grey,**dic_yellow,**dic_black}

        X['color'] = X['color'].astype(str)

        X['color'] = X['color'].map(lambda x:x.replace('canyon ridge','orange'))
        X['color'] = X['color'].map(lambda x:x.replace('mist sliver','silver'))
        X['color'] = X['color'].map(lambda x:x.replace('squeeze','green'))

        X['color'] = X['color'].map(lambda x:dic_final_color.get(x) if x in dic_final_color.keys() else x)
        X = X[X['color'].isin(original_color)]

        return X


class CityCleaner(BaseEstimator, TransformerMixin):
    """Cleans the data with varying areas of cities to their parent city or nearby city depending on the RTO registration."""

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

        original_cities = ['mumbai', 'bangalore', 'delhi', 'gurugram', 'chennai', 'hyderabad','pune']
        mumbai_d = ['mumbai','navi mumbai','thane','ulhasnagar','panvel','kalyan','vasai','dombivali','vapi','badlapur','bhiwandi','virar','badlapur']
        mumbai = ['mumbai' for i in mumbai_d]
        dic_m = dict(zip(mumbai_d,mumbai))

        delhi_d = ['new delhi']
        delhi = ['delhi' for i in delhi_d]
        dic_d = dict(zip(delhi_d,delhi))

        gurugram_d = ['faridabad','gurgaon']
        gurugram = ['gurugram' for i in gurugram_d]
        dic_g = dict(zip(gurugram_d,gurugram))

        dic_final = {**dic_m, **dic_d,**dic_g}
        X['city']=X['city'].astype(str)
        X['city'] = X['city'].map(lambda x: dic_final.get(x) if x in dic_final.keys() else x)
        X = X[X['city'].isin(original_cities)]

        return X

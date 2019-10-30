import preprocessors as pp1
import preprocessin2_color_city as pp2
from sklearn.pipeline import Pipeline
import popularity as pop
import valuation as val
import outlier_dep as od
import training_prep

carwale_pp = Pipeline([('total',pp1.CarwaleCleaner())])
idv_pp = Pipeline([('total',pp1.IDVCarwaleCleaner())])

"""Since we are merging two dataframes so our transform requires two positional arguments
which cannot be done incase of pipline"""

carwale_idv_merger = pp1.AddIdvId()

color_city_transform = Pipeline([('color',pp2.ColorCleaner()),('city',pp2.CityCleaner())])

data_for_pop = Pipeline([('datacleaner',pop.DataCleanerPop())])
popularity_cleaner = Pipeline([('pop_cleaner',pop.PopCleaner())])

"""Since we are merging two dataframes so our transform requires two positional arguments
which cannot be done incase of pipline"""

data_pop_merger = pop.MergePop()
ex_showroom_price = val.ExshowroomPriceCal()
outlier_dep = od.DepOutier()
training_prep = training_prep.TrainingPrep()

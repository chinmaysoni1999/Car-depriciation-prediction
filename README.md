# Car-depriciation-prediction

The project aims in developing a machine learning model for predicting the depreciation percentage of second hand vehicles

Initial data was scraped from various sources where second hand cars were listed for sales

Also various datasets were collected and prepared regarding margins of dealers(collected by establishing direct contact with dealers regarding cars of vaious make,model,variant), idv(Insured Declare Value),Popularity Index for vehicles  

Datasets were prepared with various preprocesing steps and trained on target value of depriciation factor

Although the pipeline prepared was subject to give batch predictions of scraped data, but here a web app is also built(using flask) through which users with minimum information of there vehicle could gain predictions for exact deprication(without any margins added) through the developed model

#Steps to run the project onto your system

1.) Create a virtual environment and install dependencies mentioned in requirements.txt, create trained_models directory in oto_model and in api directory

2.) Extract the rar datafiles in the datasets directory. Now run train_pipeline.py to train the model

3.) The model and preprocessing pickle files would be discovered in trained_models directory, copy the files into the trained_models directory in the api folder to get predictions via app

4.) Now set FLASK_APP=app

5.) flask run.....

#The app is ready to predict your old car's depriciation......!!

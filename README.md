# car-depriciation-prediction

The project aims in developing a machine learning model for predicting the depreciation percentage of second hand vehicles

Initial data was scraped from various sources where second hand cars were listed for sales

Also various datasets were collected and prepared regarding margins of dealers(collected by establishing direct contact with dealers regarding cars of vaious make,model,variant), idv(Insured Declare Value),Popularity Index for vehicles  

Datasets were prepared with various preprocesing steps and trained on target value of depriciation factor

Although the pipeline prepared was subject to give batch predictions of scraped data, but here a web app is also built(using flask) through which users with minimum information of there vehicle could gain predictions for exact deprication(without any margins added) through the developed model

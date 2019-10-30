import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from data_management import (load_dataset, load_pipeline)
from sklearn.externals import joblib
import config
import pandas as pd

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == "POST":
        make = request.form['make']
        model = request.form['model']
        variant = request.form['variant']
        color = request.form['color']
        city = request.form['city']
        age = request.form['age']
        owners = request.form['owners']
        fuel_type = request.form['fuel_type']
        kms_run = request.form['kms_run']
        transmission = request.form['transmission']
        ex_showroom_price = request.form['ex_showroom_price']
        data = load_dataset(file_name=config.CLEANED_POPULARITY)
        popularity = int(data.loc[(data['make']==str(make))&(data['model']==str(model))&(data['variant']==str(variant)),'Popularity Index'].iloc[0])
        _price_pipe=load_pipeline(file_name=config.LABEL_ENCO_DIC)
        make1 = _price_pipe['make'][make]
        model1 = _price_pipe['model'][model]
        variant1 = _price_pipe['variant'][variant]
        fuel_type1 = _price_pipe['fuel_type'][fuel_type]
        color1 = _price_pipe['color'][color]
        city1 = _price_pipe['city'][city]
        fuel_type1 = _price_pipe['fuel_type'][fuel_type]
        transmission1 = _price_pipe['transmission'][transmission]

        df = pd.DataFrame({'make':make1,
             'model':model1,
             'city':city1,
             'owners':int(owners),
             'kms_run':int(kms_run),
             'age':int(age),
             'Popularity_Index':popularity,
             'ex_showroom_price':int(ex_showroom_price),
             'fuel_type':fuel_type1,
             'transmission':int(transmission1),
             'color':color1},index=[0])

        _price_pipe=load_pipeline(file_name=config.TRAINED_MODEL)
        result = _price_pipe.predict(df)[0]
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

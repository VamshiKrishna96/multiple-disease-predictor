from flask import Flask, render_template, redirect, request
import numpy as np
import pickle

app = Flask(__name__)

def predict(values, dic):

    if len(values) == 27:
        model = pickle.load(open('models/cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


    elif len(values) == 8:
        model_2 = pickle.load(open('models/diabetes.pkl', 'rb'))
        values = np.asarray(values)
        return model_2.predict(values.reshape(1, -1))[0]

    elif len(values) == 13:
        model_3 = pickle.load(open('models/heart.pkl', 'rb'))
        values = np.asarray(values)
        return model_3.predict(values.reshape(1, -1))[0]
    
    elif len(values) == 18:
        model_4 = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model_4.predict(values.reshape(1, -1))[0]

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancer_page():
    return render_template('cancer.html')


@app.route("/diabetes", methods=['GET', 'POST'])
def diabetes_page():
    return render_template('diabetes.html')


@app.route("/heart", methods=['GET', 'POST'])
def heart_page():
    return render_template('heart.html')


@app.route("/kidney", methods=['GET', 'POST'])
def kidney_page():
    return render_template('kidney.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict_page():

    try:
        if request.method == 'POST':


            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)


    except:
        message = "Please Enter Valid Data"
        return render_template("home.html", message = message)



    return render_template('predict.html', pred = pred)
    


if __name__=="__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

def get_classification_model(model_file):
    try:
        with open(model_file, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def get_model(model_name):
    dir_path = os.path.abspath("models")
    
    if model_name == 'simpleNeuralNetwork':
        return get_classification_model(os.path.join(dir_path, "neural_network.pkl"))
    elif model_name == 'naiveBayes':
        return get_classification_model(os.path.join(dir_path, "naive_bayes.pkl"))
    elif model_name == 'kNN':
        return get_classification_model(os.path.join(dir_path, "knn.pkl"))
    elif model_name == 'std_scaler':
        return get_classification_model(os.path.join(dir_path, "std_scaler.pkl"))
    elif model_name == 'norm_scaler':
        return get_classification_model(os.path.join(dir_path, "norm_scaler.pkl"))
    else:
        return get_classification_model(os.path.join(dir_path, "Logistic_Regression.pkl"))


# Retriev Scalers
standard_scaler = get_model("std_scaler")
norm_scaler = get_model("norm_scaler")

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        model_name = request.form.get('model')
        
        age = int(request.form.get('age'))
        
        sex = 0
        if request.form.get('sex') == "Female":
            sex = 1

        cp = int(request.form.get('cp'))
        trestbps = int(request.form.get('trestbps'))
        chol = int(request.form.get('chol'))
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form.get('restecg'))
        thalach = int(request.form.get('thalach'))
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = int(request.form.get('slope'))
        ca = int(request.form.get('ca'))
        thal = int(request.form.get('thal'))
        
        model = get_model(model_name)

        user_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                  oldpeak, slope, ca, thal]) 
        
        reshaped_user_input = user_input.reshape(1, -1)

        if model_name == "logisticRegression" or model_name == 'kNN':
            reshaped_user_input = standard_scaler.transform(reshaped_user_input)

        elif model_name == 'naiveBayes' or model_name == 'simpleNeuralNetwork':
            reshaped_user_input = norm_scaler.transform(reshaped_user_input)

        # Make predictions
        result_2D = model.predict(reshaped_user_input)


        prediction = "No presence of a Heart Disease"
        y_hat = 0

        if model_name == "simpleNeuralNetwork":
            predicted_labels = (result_2D > 0.7).astype(int)
            y_hat = predicted_labels[0][0]
            if y_hat == 1:
                prediction = "Presence of a Heart Disease"
        else:
            y_hat = result_2D[0]
            if y_hat == 1:
                prediction = "Presence of a Heart Disease"

        input_summary = f"""
        age:{age}   | sex:{sex} | cp:{cp}  |trestbps:{trestbps} \n
        chol:{chol} | fbs:{fbs} | restecg:{restecg} | thalach:{thalach}, \n 
        exang:{exang} | oldpeak:{oldpeak} |slope:{slope} | ca:{ca} | thal:{thal}
        """
        
        result_dict = {
            "model" : model_name,
            "user_input": input_summary,
            "prediction": prediction
        }

        return render_template("index.html", results=result_dict)
    else:
        result_dict = {
            "model": "None",
            "user_input": "None",
            "prediction": "None"
        }
        return render_template("index.html", results=None)

if __name__ == '__main__':
    app.run(debug=True)

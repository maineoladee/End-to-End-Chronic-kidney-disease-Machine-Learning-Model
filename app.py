from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("ckd_RFF.pkl", "rb"))
scalar=pickle.load(open('scaling.pkl','rb'))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # Age_of_Patient
        Age_of = request.form["Age"]
        Age_of = int(Age_of)

        # Blood Pressure
        blood_press = request.form["Blood_Pressure"]
        blood_press = int(blood_press)

        # Specific gravity
        specific_grav = float(request.form["Specific_Gravity"])

        #Albumin
        Alb = int(request.form["Albumin"])

        #Sugar
        sug = int(request.form["Sugar"])

        #Red Blood Cells
        RBC = int(request.form["Red_Blood_Cells"])

        #Pus Cell
        pus_c = int(request.form["Pus_Cell"])

        #Pus Cell Clumps
        pus_cc = int(request.form["Pus_Cell_Clumps"])


        #Bacteria
        bact = int(request.form["Bacteria"])

        #Blood Glucose Random
        BGR = int(request.form["Blood_Glucose_Random"])

        #Blood Urea
        bu= float(request.form["Blood_Urea"])

        #serum creatinine
        serum = float(request.form["Serum_Creatinine"])

        #Sodium
        sod = float(request.form["Sodium"])

        #Potassium
        K = float(request.form["Potassium"])

        #hemoglobin
        hemo = float(request.form["Hemoglobin"])

        #Packed Cell Volume
        pcv = int(request.form["Packed_Cell_Volume"])

        #White Blood Cell Count
        wbcc = int(request.form["White_Blood_Cell_Count"])

        #Red Blood Cell Count
        rbcc = float(request.form["Red_Blood_Cell_Count"])

        #Hypertension
        hypten = int(request.form["Hypertension"])

        #diabetes mellitus
        diab_mell = int(request.form["Diabetes_Mellitus"])

        #Coronary Artery Disease
        cad = float(request.form["Coronary_Artery_Disease"])

        #Appetite
        appet = int(request.form["Appetite"])

        #Pedal Edema
        pe = int(request.form["Pedal_Edema"])

        #Anemia
        ane = int(request.form["Anemia"])

        




        
        prediction=model.predict(scalar.transform(np.array([[
            Age_of,
            blood_press,
            specific_grav,
            Alb,
            sug,
            RBC,
            pus_c,
            pus_cc,
            bact,
            BGR,
            bu,
            serum,
            sod,
            K,
            hemo,
            pcv,
            wbcc,
            rbcc,
            hypten,
            diab_mell,
            cad,
            appet,
            pe,
            ane,
            ]])))

        output=round(prediction[0],2)

        
        if output == 1:
            output = "Chronic Kidney Disease"
        else:
            output = "Not Chronic Kidney Disease"

        return render_template('home.html',prediction_text="Prediction is {}".format(output))


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


application =Flask(__name__)
app =application

# import ridge model or scaler  
ridge_model =pickle.load(open('models/ridge.pkl','rb'))
standard_scaler =pickle.load(open('models/scaler.pkl','rb'))


@app.route("/homepage")
def HomePage():
    return render_template("index.html")

@app.route("/homepage/predictdata",methods =["GET","POST"])
def predict_datapoint():
    if request.method =="POST" :
        Temperature =float(request.form.get("Temperature"))
        RH =float(request.form.get("RH"))
        Ws =float(request.form.get("Ws"))
        Rain =float(request.form.get("Rain"))
        DMC =float(request.form.get("DMC"))
        DC =float(request.form.get("DC"))
        ISI =float(request.form.get("ISI"))
        Classes =float(request.form.get("Classes"))
        Region =float(request.form.get("Region"))
        
        ### Scaled new data
        new_scaled_data= standard_scaler.transform([[Temperature,RH,Ws,Rain,DMC,DC,ISI,Classes,Region]])
        
        ### predict new data
        result =ridge_model.predict(new_scaled_data)
        return render_template("home.html",result=result[0])
    else:
        return render_template("home.html")




if __name__ == "__main__" :
    app.run(host="0.0.0.0")

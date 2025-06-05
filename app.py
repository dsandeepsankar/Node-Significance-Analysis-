from flask import Flask, render_template, request
import pickle
import numpy as np
from database import *
from sklearn.preprocessing import LabelEncoder
import joblib
app = Flask(__name__,static_url_path='/static')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
# Load the machine learning model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib 
# 
  
@app.route('/p')
def p():
    return render_template('index.html')

@app.route('/')
def m():
    return render_template('main.html')

@app.route('/l')
def l():
    return render_template('login.html')

@app.route('/h')
def h():
    return render_template('home.html')

@app.route('/r')
def r():
    return render_template('register.html')

@app.route('/m')
def menu():
    return render_template('menu.html')



@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/home.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input from form
    degree_centrality = float(request.form['degree_centrality'])
    betweenness_centrality = float(request.form['betweenness_centrality'])
    closeness_centrality = float(request.form['closeness_centrality'])
    eigenvector_centrality = float(request.form['eigenvector_centrality'])
    community_assignment = request.form['community_assignment']
    node_type = request.form['node_type']
    clustering_coefficient = float(request.form['clustering_coefficient'])

    # Load the dataset
    file_path = "node_significance_analysis_data.csv"  # Ensure this file is in the project directory
    df = pd.read_csv(file_path)

    # Drop identifier column
    df = df.drop(columns=["Node_ID"])

    # Encode categorical features
    label_encoders = {}
    categorical_cols = ["Community_Assignment", "Node_Type", "Node_Significance"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop("Node_Significance", axis=1)
    y = df["Node_Significance"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the best model (Gradient Boosting)
    best_model = GradientBoostingClassifier()
    best_model.fit(X_train, y_train)

    # Create input DataFrame for prediction
    input_data = pd.DataFrame([{
        "Degree_Centrality": degree_centrality,
        "Betweenness_Centrality": betweenness_centrality,
        "Closeness_Centrality": closeness_centrality,
        "Eigenvector_Centrality": eigenvector_centrality,
        "Community_Assignment": label_encoders["Community_Assignment"].transform([community_assignment])[0],
        "Node_Type": label_encoders["Node_Type"].transform([node_type])[0],
        "Clustering_Coefficient": clustering_coefficient
    }])

    # Predict
    predicted_encoded = best_model.predict(input_data)[0]
    predicted_significance = label_encoders["Node_Significance"].inverse_transform([predicted_encoded])[0]

    # Render result
    return render_template("result.html", op1=predicted_significance)



if __name__ == "__main__":
    app.run(debug=True, port=5112)
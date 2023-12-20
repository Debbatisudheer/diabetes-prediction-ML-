import os
from flask import Flask, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
import time

app = Flask(__name__)

# Load your dataset (replace 'diabetes.csv' with your actual file)
dataset = pd.read_csv('diabetes.csv')

# Assuming your dataset has features (X) and a target variable (y)
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_and_evaluate(model, model_name, X_train_scaled, y_train, X_test_scaled, y_test):
    # Create and train the model with scaled features
    if model_name == 'Logistic Regression':
        model_instance = LogisticRegression(max_iter=1000)
    elif model_name == 'Decision Tree':
        model_instance = DecisionTreeClassifier()
    elif model_name == 'Random Forest':
        model_instance = RandomForestClassifier()
    elif model_name == 'SVM':
        model_instance = SVC()
    elif model_name == 'KNN':
        model_instance = KNeighborsClassifier()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Measure the start time
    start_time = time.time()

    model_instance.fit(X_train_scaled, y_train)

    # Measure the end time
    end_time = time.time()

    # Make predictions on the scaled test set
    y_pred_scaled = model_instance.predict(X_test_scaled)

    # Evaluate the model
    accuracy_scaled = accuracy_score(y_test, y_pred_scaled) * 100  # Convert to percentage
    classification_rep = classification_report(y_test, y_pred_scaled)

    return accuracy_scaled, end_time - start_time

def train_all_models(X_train_scaled, y_train, X_test_scaled, y_test):
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN']
    results = {'Model': [], 'Accuracy': [], 'Execution Time': []}

    for model_name in models:
        accuracy, exec_time = train_and_evaluate(model_name, model_name, X_train_scaled, y_train, X_test_scaled, y_test)
        results['Model'].append(model_name)
        results['Accuracy'].append(accuracy)
        results['Execution Time'].append(exec_time)

    return results

@app.route('/')
def home():
    results = train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # Plot combined graph
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'green', 'orange', 'purple', 'red']  # Change these colors

    for model, accuracy, exec_time, color in zip(results['Model'], results['Accuracy'], results['Execution Time'], colors):
        label = f'{model}\nAccuracy: {accuracy:.2f}%\nExecution Time: {exec_time:.2f}s'
        ax.bar(model, accuracy, label=label, color=color)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Diabetes Prediction - Model Performance')
    ax.legend()

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Encode the image to base64
    image_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    return render_template('index.html', image_base64=image_base64)

if __name__ == '__main__':
    app.run(debug=True)










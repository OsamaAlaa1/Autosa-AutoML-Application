# Needed packages
from flask import Flask, render_template,request,send_file, jsonify
import os
import time 
import pandas as pd 
import zipfile

# helpful packages for exploration 
from ydata_profiling import ProfileReport
import pdfkit
import nbconvert

# transformation 
from sklearn.model_selection import train_test_split
from utilities import DataPreprocessor
import joblib

# training and fine tuning using using H2O automl and auto-keras
import h2o
from h2o.automl import H2OAutoML

app = Flask(__name__)


# needed global variables

# selected task (classification or regression)
task_option = None
is_uploaded = False

# selection for target column
selected_target_column = None

# home call
@app.route('/')
@app.route('/home')
@app.route('/index.html')
def home():
    return render_template('home.html')


# automation call 
@app.route('/auto-ml')
def auto_ml():
    return render_template('auto-ml.html')

# automation call 
@app.route('/automation')
def automation():
    return render_template('automation.html')


# automation call
@app.route('/upload', methods=['GET', 'POST'])
def upload():

    global is_uploaded

    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        # check there is a file 
        if uploaded_file.filename != '':
            
            # rename the uploaded file 
            file_path = os.path.join('data.csv')

            # flag for uploading 
            is_uploaded = True

            # Save the uploaded file to the specified path
            uploaded_file.save(file_path)

        
            return 'File uploaded successfully!'
        
        
@app.route('/get_columns', methods=['GET'])
def get_columns():
    dataset = pd.read_csv('data.csv')
    columns = dataset.columns.tolist()
    return jsonify(columns)

@app.route('/select_target', methods=['POST'])
def select_target():
    global selected_target_column
    selected_target_column = request.form.get('target_column')
    print(selected_target_column)
    return jsonify(status='success')


# exploration call  
@app.route('/explore', methods=['GET', 'POST'])
def explore():

    if is_uploaded == True: 
        
        # read data from csv and convert it into dataframe object
        df = pd.read_csv('data.csv')
        
        # get the profile  
        profile = ProfileReport(df, title='Data Profiling Report')
        # save the report 
        profile.to_file("report.html")


        
        # html = profile.to_html()
        # pdfkit.from_string(html, "report.pdf")
        #nbconvert.convert_notebook("report.html", "jupyter", output_path="report.ipynb")
    
    return "successfully Explored"


# report call
@app.route('/report')
def show_report():
    return send_file('report.html')


@app.route('/transform', methods=['GET', 'POST'])
def transform():

    if is_uploaded == True : 
        
        # read uploaded csv file 
        data = pd.read_csv('data.csv')
        
        # split the data 
        X = data.drop(columns=[selected_target_column])
        y = data[selected_target_column]

        # train test split of dataset
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state= 42)
        

        # Fit and transform the data using the preprocessing pipeline
        preprocessor = DataPreprocessor()
        transformed_data = preprocessor.fit_transform(X_train)
        

        # Create a temporary directory to store files
        temp_dir = 'transformer_plus_data'
        os.makedirs(temp_dir, exist_ok=True)

        # save the transformer 
        joblib.dump(preprocessor, 'transformer_plus_data/transformer.pkl')

        # Save the preprocessed data to a CSV file
        transformed_data = pd.DataFrame(transformed_data)
        transformed_data.to_csv('transformer_plus_data/transformed_data.csv', index=False)


    return "Data successfully Transformed"


@app.route('/fine_tune', methods=['GET', 'POST'])
def fine_tune():

    global selected_target_column
    global task_option


    # Initialize H2O
    h2o.init()

    # Load your dataset
    data = h2o.import_file("data.csv")

    # Define the target variable
    y = selected_target_column  # Replace with the name of your target column

    # Initialize and train H2O AutoML
    aml = H2OAutoML(max_models=10, seed=42)
    aml.train(y=y, training_frame=data)

    # Create a report file
    report_file_path = "model_report.txt"
    with open(report_file_path, "w") as report_file:

        # Save the best model
        best_model = aml.leader
        model_path = h2o.save_model(model=best_model, path="best_model", force=True)
        report_file.write(f"Best Model:\n")
        report_file.write(f"Model Name: {best_model.model_id}\n")
        report_file.write(f"Model Path: {model_path}\n")
        report_file.write("\n")

        # Show all the models tried by AutoML
        leaderboard = aml.leaderboard
        report_file.write("All Models Tried by AutoML:\n")
        for model in leaderboard:
            model_name = model[0]
            metrics = model[2:]
            report_file.write(f"Model: {model_name}\n")
            for metric in metrics:
                report_file.write(f"{metric[0]}: {metric[1]}\n")
            report_file.write("\n")

    # Shutdown H2O
    h2o.shutdown()


    return "successfully Explored"




@app.route('/download_zip')
def download_zip():


    temp_dir = 'transformer_plus_data'

    # Create a zip archive from the temporary directory
    zip_filename = 'transformer_plus_data.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

    # Return the zip archive as an attachment
    return send_file(zip_filename, as_attachment=True)


@app.route('/download_report')
def download_report():
    file_path = 'model_report.txt'
    return send_file(file_path, as_attachment=True)

@app.route('/download_model')
def download_model():
    
    temp_dir = 'best_model'
    # Create a zip archive from the temporary directory
    zip_filename = 'best_model.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))

    # Return the zip archive as an attachment
    return send_file(zip_filename, as_attachment=True)



@app.route('/store_selection', methods=['POST'])
def store_selection():
    data = request.get_json()
    global task_option 
    task_option = data.get('selection')
    print(task_option)
    return "Selected Successfully!"


if __name__ == '__main__':
    app.run(debug=True)

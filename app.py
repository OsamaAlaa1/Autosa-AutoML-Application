from flask import Flask, render_template,request,send_file
import os
import time 

app = Flask(__name__)

selected_option = None

# home call
@app.route('/')
@app.route('/home')
@app.route('/index.html')
def home():
    return render_template('home.html')

@app.route('/automation')
def automation():
    return render_template('automation.html')


# automation call
@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        # check there is a file 
        if uploaded_file.filename != '':
            
            # rename the uploaded file 
            file_path = os.path.join('data.csv')

            # Save the uploaded file to the specified path
            uploaded_file.save(file_path)

            return 'File uploaded successfully!'
        




@app.route('/explore', methods=['GET', 'POST'])
def explore():

    time.sleep(5)
    
    return "successfully Explored"


@app.route('/transform', methods=['GET', 'POST'])
def transform():

    time.sleep(5)
    
    return "successfully Explored"


@app.route('/fine_tune', methods=['GET', 'POST'])
def fine_tune():

    time.sleep(5)
    
    return "successfully Explored"


@app.route('/download_transform')
def download_transform():
    file_path = 'data.csv'
    return send_file(file_path, as_attachment=True)


@app.route('/download_report')
def download_report():
    file_path = 'data.csv'
    return send_file(file_path, as_attachment=True)


@app.route('/download_model')
def download_model():
    file_path = 'data.csv'
    return send_file(file_path, as_attachment=True)


@app.route('/store_selection', methods=['POST'])
def store_selection():
    data = request.get_json()
    global selected_option 
    selected_option = data.get('selection')
    print(selected_option)
    return "Selected Successfully!"

if __name__ == '__main__':
    app.run(debug=True)

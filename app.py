from flask import Flask, flash, request, redirect, url_for
import os
import main
import pandas as pd

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['text', 'csv']

def validate_files(f):
    return f.filename.split('.')[-1] in ALLOWED_EXTENSIONS

def train_model():
    train = pd.read_csv('./data/train.csv')
    y = train.pop('Survived')
    train = main.clean_data(train)
    train = main.one_hot_encode(train)
    model = main.build_model()
    model.fit(train, y)
    return model


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        if f and validate_files(f):
            model = train_model()
            test = pd.read_csv(f)
            ids = test['PassengerId']
            test = main.clean_data(test)
            test = main.one_hot_encode(test)
            result = pd.DataFrame({'PassengerId': ids, 'Survived': model.predict(test)})
            return result.to_html(index = False)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run()

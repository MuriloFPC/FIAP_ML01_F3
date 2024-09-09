from flask import Flask, render_template, request, send_file

import pickle
app = Flask(__name__)

app.model_nota_ch = None


@app.route('/', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    if request.method == 'POST' or request.method == 'GET':
        if app.model_nota_ch is None:
            app.model_nota_ch = import_model('nota_ch')
    return render_template('index.html')


def import_model(model_name):
    with open(f'./models_pickle/model_{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    app.run()
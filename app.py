from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle

app = Flask(__name__)

app.model_nota_ch = None
app.model_nota_cn = None
app.model_nota_lc = None
app.model_nota_mt = None
app.model_nota_re = None


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        errors = validate_form(request.form)
        if errors is not None:
            return errors

        if app.model_nota_ch is None:
            app.model_nota_ch = import_model('nota_ch')
            pass

        if app.model_nota_cn is None:
            #app.model_nota_cn = import_model('nota_cn')
            pass

        if app.model_nota_lc is None:
            #app.model_nota_lc = import_model('nota_lc')
            pass

        if app.model_nota_mt is None:
            #app.model_nota_mt = import_model('nota_mt')
            pass

        if app.model_nota_re is None:
            #app.model_nota_re = import_model('nota_redacao')
            pass

        df =  pd.DataFrame([request.form.to_dict()])
        #y_ch = app.model_nota_ch.predict(df)
        #y_cn = app.model_nota_cn.predict(df)
        #y_lc = app.model_nota_lc.predict(df)
        #y_mt = app.model_nota_mt.predict(df)
        #y_re = app.model_nota_re.predict(df)
        y_ch = 458.3378
        response = f"""
        <div class="alert alert-success">
            <h4>Nota ENEM prevista</h4>
            <p><strong>CH:</strong>{round(y_ch,2)}</p>
            <p><strong>CN:</strong>{round(y_ch,2)}</p>
            <p><strong>LC:</strong>{round(y_ch,2)} </p>
            <p><strong>MT:</strong>{round(y_ch,2)} </p>
            <p><strong>Redação:</strong>{round(y_ch,2)} </p>
        </div>
        """
        return response

def import_model(model_name):
    with open(f'./models_pickle/model_XGBoost_{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
def validate_form(form):
    list_errors = []
    form_keys = form.keys()

    dict_perguntas = {
        'Q001':1,
        'Q002':2,
        'Q003':3,
        'Q004':4,
        'Q005':5,
        'Q006':6,
        'Q007':7,
        'Q008':8,
        'Q009':9,
        'Q010':10,
        'Q011':11,
        'Q012':12,
        'Q013':13,
        'Q014':14,
        'Q015':15,
        'Q016':16,
        'Q017':17,
        'Q018':18,
        'Q019':19,
        'Q020':20,
        'Q021':21,
        'Q022':22,
        'Q023':23,
        'Q024':24,
        'Q025':25
    }

    for key in dict_perguntas.keys():
        if not form_keys.__contains__(key):
            list_errors.append(f'<p><strong>Pergunta {dict_perguntas[key]} sem resposta</strong></p>')

    if list_errors is not []:
        response = f"""
                <div class="alert alert-danger">
                    <h4>Erro no preenchimento do Form</h4>
                    {''.join(list_errors)}
                </div>
                """
        return response

    return None

if __name__ == '__main__':
    app.run()
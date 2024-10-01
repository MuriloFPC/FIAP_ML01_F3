import sys
import logging
import pandas as pd
import pickle
import xgboost as xgb
from flask import Flask, render_template, request


app = Flask(__name__)
app.model_carregado = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger('FlaskApp')


def import_model(model_name):
    with open(f'./models_pickle/model_XGBoost_{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def load_models():
    logger.info('Loading models...')
    global model_nota_ch, model_nota_cn, model_nota_lc, model_nota_mt, model_nota_re # noqa

    logger.info('Loading model nota_ch...')
    model_nota_ch = import_model('nota_ch')

    logger.info('Loading model nota_cn...')
    model_nota_cn = import_model('nota_cn')

    logger.info('Loading model nota_lc...')
    model_nota_lc = import_model('nota_lc')

    logger.info('Loading model nota_mt...')
    model_nota_mt = import_model('nota_mt')

    logger.info('Loading model nota_re...')
    model_nota_re = import_model('nota_redacao')

    setattr(app, 'model_carregado', True)
    logger.info('Models loaded')


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        errors = validate_form(request.form)
        if errors is not None:
            return errors

        if not app.model_carregado:
            return """
                    <div class="alert alert-danger">
                        <p><strong>Modelos sendo carregados, aguarde um momento</strong></p>  # noqa
                    </div>
                    """

        dict = request.form.to_dict()
        dict = {k: [int(v)] for k, v in dict.items()}

        df = pd.DataFrame.from_dict(dict)

        df = df[['Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006',
                 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012',
                 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018',
                 'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024',
                 'Q025', 'TP_ESCOLA', 'TP_ENSINO', 'TP_FAIXA_ETARIA']]

        # Criando o DMatrix
        df_dmatrix = xgb.DMatrix(df, enable_categorical=True)

        y_ch = round(model_nota_ch.predict(df_dmatrix)[0], 2)
        y_cn = round(model_nota_cn.predict(df_dmatrix)[0], 2)
        y_lc = round(model_nota_lc.predict(df_dmatrix)[0], 2)
        y_mt = round(model_nota_mt.predict(df_dmatrix)[0], 2)
        y_re = round(model_nota_re.predict(df_dmatrix)[0], 2)

        response = f"""
        <div class="alert alert-success">
            <h4>Nota ENEM prevista</h4>
            <p><strong>Nota Ciências Humanas: </strong>{y_ch:.2f}</p>
            <p><strong>Nota Ciências da Natureza: </strong>{y_cn:.2f}</p>
            <p><strong>Nota Linguagens e Códigos: </strong>{y_lc:.2f} </p>
            <p><strong>Nota Matemática: </strong>{y_mt:.2f} </p>
            <p><strong>Nota Redação: </strong>{y_re:.2f} </p>
        </div>
        """
        return response


def validate_form(form):
    list_errors = []
    form_keys = form.keys()

    dict_perguntas = {
        'Q001': 1,
        'Q002': 2,
        'Q003': 3,
        'Q004': 4,
        'Q005': 5,
        'Q006': 6,
        'Q007': 7,
        'Q008': 8,
        'Q009': 9,
        'Q010': 10,
        'Q011': 11,
        'Q012': 12,
        'Q013': 13,
        'Q014': 14,
        'Q015': 15,
        'Q016': 16,
        'Q017': 17,
        'Q018': 18,
        'Q019': 19,
        'Q020': 20,
        'Q021': 21,
        'Q022': 22,
        'Q023': 23,
        'Q024': 24,
        'Q025': 25,
        'TP_ESCOLA': 26,
        'TP_ENSINO': 27,
        'TP_FAIXA_ETARIA': 28
    }

    for key in dict_perguntas.keys():
        if not form_keys.__contains__(key):
            list_errors.append(f'<p><strong>Pergunta {dict_perguntas[key]} sem resposta</strong></p>')  # noqa

    if list_errors:
        response = f"""
                <div class="alert alert-danger">
                    <h4>Erro no preenchimento do Form</h4>
                    {''.join(list_errors)}
                </div>
                """
        return response

    return None


load_models()
if __name__ == '__main__':
    app.run()

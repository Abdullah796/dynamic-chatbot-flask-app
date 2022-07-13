import json

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from utils import retrieve_best_match, build_model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def hello_world():
    return jsonify({'message': 'Server app is running fine'})


@app.route('/asked-admission-cell', methods=['POST'])
@cross_origin()
def asked_admission_cell():
    print("asked_admission_cell start")
    dataset_path = "static\\datasets\\NED-ADMISSION-CELL-DATASET.csv"
    model_path = "static\\embeddings\\admission_cell_model.pkl"
    record = json.loads(request.data)
    raw_question = record["question"]
    print(f"asked_admission_cell: {raw_question}")
    result_dict = retrieve_best_match(dataset_path, model_path, raw_question)
    print(f"asked_admission_cell: result_dict {result_dict}");
    print("asked_admission_cell end")
    return jsonify(result_dict)


@app.route('/build-admission-cell', methods=['GET'])
def build_admission_cell():
    dataset_path = "static\\datasets\\NED-ADMISSION-CELL-DATASET.csv"
    model_path = "static\\embeddings\\admission_cell_model.pkl"
    build_model(dataset_path, model_path)
    return jsonify({"message": "ned admission cell model build"})


@app.route('/asked-covid-faq', methods=['POST'])
def asked_covid_faq():
    dataset_path = "static\\datasets\\covid_faq.csv"
    model_path = "static\\embeddings\\covid_faq_model.pkl"
    record = json.loads(request.data)
    raw_question = record["question"]
    result_dict = retrieve_best_match(dataset_path, model_path, raw_question)
    return jsonify(result_dict)


@app.route('/build-covid-faq', methods=['GET'])
def build_covid_faq():
    dataset_path = "static\\datasets\\covid_faq.csv"
    model_path = "static\\embeddings\\covid_faq_model.pkl"
    build_model(dataset_path, model_path)
    return jsonify({"message": "covid faq model build"})


if __name__ == '__main__':
    app.run()

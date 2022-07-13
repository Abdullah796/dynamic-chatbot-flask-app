import json

from flask import Flask, request, jsonify

from utils import retrieve_best_match, build_model

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({'message': 'Server app is running fine'})


@app.route('/asked-admission-cell', methods=['POST'])
def asked_admission_cell():
    dataset_path = "static\\datasets\\NED-ADMISSION-CELL-DATASET.csv"
    model_path = "static\\embeddings\\admission_cell_model.pkl"
    record = json.loads(request.data)
    raw_question = record["question"]
    result_dict = retrieve_best_match(dataset_path, model_path, raw_question)
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

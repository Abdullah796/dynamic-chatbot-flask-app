import pickle
import re

import numpy as np
import pandas as pd
from gensim import models
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics.pairwise import cosine_similarity


def pkl_to_data(path):
    print('start pkl_to_list function of PklService')
    np.set_printoptions(threshold=1000000000000000)
    file = open(path, 'rb')
    print('reads the contents of the file pkl')
    inf = pickle.load(file, encoding='iso-8859-1')
    print('end pkl_to_list function of PklService')
    file.close()
    return inf


def data_to_pkl(path, list_data):
    print('start list_to_pkl function of PklService')
    np.set_printoptions(threshold=1000000000000000)
    with open(path, 'wb') as fo:
        pickle.dump(list_data, fo)
    print('end list_to_pkl function of PklService')


def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence


def get_cleaned_sentences(df, stopwords=False):
    cleaned_sentences = []

    for index, row in df.iterrows():
        # print(index,row)
        cleaned = clean_sentence(row["questions"], stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences


def get_word_vec(word, w2vec_embedding_size, model):
    vec = [0] * w2vec_embedding_size
    try:
        vec = model[word]
    except:
        vec = [0] * w2vec_embedding_size
    return (vec)


def get_phrase_embedding(phrase, w2vec_embedding_size, model):
    vec = np.array([0] * w2vec_embedding_size)
    for word in phrase.split():
        vec = vec + np.array(get_word_vec(word, w2vec_embedding_size, model))
    return vec.reshape(1, -1)


def retrieve_and_print_faq_answer(question, question_embedding, sentence_embeddings, FAQdf, sentences):
    max_sim = -1
    index_sim = -1
    for index, faq_embedding in enumerate(sentence_embeddings):
        # sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0]
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        # print(index, sim, sentences[index])
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    return {
        "askedQuestion": question,
        "retrievedQuestion": FAQdf.iloc[index_sim, 0],
        "retrievedAnswer": FAQdf.iloc[index_sim, 1],
        "type": "chatbot"
    }


def retrieve_best_match(dataset_path, model_path, raw_question):
    question = clean_sentence(raw_question, stopwords=False)
    model = models.KeyedVectors.load_word2vec_format(
        'static/models/GoogleNews-vectors-negative300.bin', binary=True)
    print("model import successfully")
    w2vec_embedding = model['computer']
    w2vec_embedding_size = len(w2vec_embedding)
    question_embedding = get_phrase_embedding(question, w2vec_embedding_size, model)
    sent_embeddings = pkl_to_data(model_path)
    df = pd.read_csv(dataset_path)
    df.columns = ["questions", "answers"]
    cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
    result_dict = retrieve_and_print_faq_answer(question, question_embedding, sent_embeddings, df, cleaned_sentences)
    return result_dict


def build_model(dataset_path, model_path):
    df = pd.read_csv(dataset_path)
    df.columns = ["questions", "answers"]
    print(df)
    cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
    print(cleaned_sentences)

    print("\n")
    cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)
    print(cleaned_sentences_with_stopwords)
    model = None
    try:
        model = models.KeyedVectors.load_word2vec_format(
            'static/models/GoogleNews-vectors-negative300.bin', binary=True)
        print("model import successfully")
    except Exception as e:
        print("model can't import", e)
    w2vec_embedding = model['computer']
    w2vec_embedding_size = len(w2vec_embedding)
    print(f"w2vec_embedding: {w2vec_embedding}, w2vec_embedding_size: {w2vec_embedding_size}")
    sent_embeddings = []
    for sent in cleaned_sentences:
        sent_embeddings.append(get_phrase_embedding(sent, w2vec_embedding_size, model))
    data_to_pkl(model_path, sent_embeddings)

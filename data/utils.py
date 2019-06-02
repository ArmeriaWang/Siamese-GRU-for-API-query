import pickle
import string
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=2)


# 去除文本中的标点等符号
def remove_symbols(sentence):
    del_estr = string.punctuation
    replace = " " * len(del_estr)
    tran_tab = str.maketrans(del_estr, replace)
    sentence = sentence.translate(tran_tab)
    return sentence


def remove_stop_words(sentence):
    return [w for w in sentence if not w in stop_words]


def text_process(tstr):
    """ Tokenizing, stemming. Return a list of tokens. """
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(word) for sent in sent_tokenize(tstr) for word in word_tokenize(sent)]
    return tokens


def sent2matrix(w2v, embedding_dim, sentence):
    matrix = np.zeros((len(sentence), embedding_dim))
    for i, w in enumerate(sentence):
        matrix[i] = w2v.get(w, w2v.get('pad'))
    # l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(sentence), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm != 0)
    except RuntimeWarning:
        print(sentence)
    return matrix


def sentence_idf_vector(idf, sentence):
    idf_vector = np.zeros((1, len(sentence)))  # word embedding size is 100
    for i, word in enumerate(sentence):
        if word in idf:
            idf_vector[0][i] = idf[word]
    return idf_vector

import numpy as np


def load_vec_list_file(vec_file_name):
    fin = open(vec_file_name, 'rb')
    x = np.fromfile(fin, np.int32, 2)
    num_vecs = x[0]
    vec_dim = x[1]
    vec_list = list()
    for i in xrange(num_vecs):
        vec = np.fromfile(fin, np.float32, vec_dim)
        vec_list.append(vec)
    fin.close()
    return vec_list


def load_labels_file(labels_file_name):
    fin = open(labels_file_name, 'rb')
    num_labels = np.fromfile(fin, np.int32, 1)
    labels = np.fromfile(fin, np.int32, num_labels)
    fin.close()
    return labels


def load_bow_file(file_name):
    fin = open(file_name, 'rb')
    params = np.fromfile(fin, np.int32, 2)
    num_docs, num_words = params
    word_indices_list = list()
    word_cnts_list = list()
    for i in xrange(num_docs):
        num_words_in_doc = np.fromfile(fin, np.int32, 1)
        word_indices = np.fromfile(fin, np.int32, num_words_in_doc)
        word_cnts = np.fromfile(fin, np.int32, num_words_in_doc)
        word_indices_list.append(word_indices)
        word_cnts_list.append(word_cnts)
    fin.close()

    return word_indices_list, word_cnts_list, num_words

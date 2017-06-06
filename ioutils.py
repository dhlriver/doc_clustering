import numpy as np
from array import array
import os


def get_file_len(file_name):
    fin = open(file_name, 'rb')
    fin.seek(0, os.SEEK_END)
    file_len = fin.tell()
    fin.close()
    return file_len


def read_str_with_byte_len(fin):
    slen = np.fromfile(fin, '>i1', 1)
    if not slen:
        return ''

    s = array('B')
    s.fromfile(fin, slen)
    return s.tostring()


def write_str_with_byte_len(s, fout):
    slen = len(s)
    np.asarray([slen], np.int8).tofile(fout)
    fout.write(s)


def read_str_with_fixed_len(fin, slen):
    s = array('B')
    s.fromfile(fin, slen)
    pos = slen - 1
    while pos > -1 and s[pos] == 0:
        pos -= 1
    s = s[:pos + 1]
    return s.tostring().strip()


# TODO merge with load words dict
def load_entity_dict(file_name):
    fin = open(file_name, 'rb')
    entities = list()
    for line in fin:
        entities.append(line.strip().split('\t')[0])
    fin.close()
    return entities


def load_words_dict_to_list(file_name, skip_first_line=True):
    fin = open(file_name, 'rb')
    if skip_first_line:
        line = fin.readline()
        print line
    words = list()
    for line in fin:
        words.append(line.strip().split('\t')[0])
    fin.close()
    return words


def load_vec_list_file(vec_file_name):
    fin = open(vec_file_name, 'rb')
    num_vecs, dim = np.fromfile(fin, np.int32, 2)
    print '%d vectors, dim: %d' % (num_vecs, dim)
    # num_vecs = 1
    vecs = np.zeros((num_vecs, dim), np.float32)
    for i in xrange(num_vecs):
        vec = np.fromfile(fin, np.float32, dim)
        vecs[i][:] = vec

    fin.close()
    return vecs


def save_labels(labels, dst_file):
    fout = open(dst_file, 'wb')
    np.asarray([len(labels)], dtype=np.int32).tofile(fout)
    np.asarray(labels, dtype=np.int32).tofile(fout)
    fout.close()


def load_labels_file(labels_file_name):
    fin = open(labels_file_name, 'rb')
    num_labels = np.fromfile(fin, np.int32, 1)
    labels = np.fromfile(fin, np.int32, num_labels)
    fin.close()
    return labels


def load_bow_file(file_name, uint16_cnts=True):
    fin = open(file_name, 'rb')
    num_docs, num_words = np.fromfile(fin, np.int32, 2)
    word_indices_list = list()
    word_cnts_list = list()
    for i in xrange(num_docs):
        num_words_in_doc = np.fromfile(fin, np.int32, 1)
        # print num_words_in_doc
        word_indices = np.fromfile(fin, np.int32, num_words_in_doc)
        if uint16_cnts:
            word_cnts = np.fromfile(fin, np.uint16, num_words_in_doc)
        else:
            word_cnts = np.fromfile(fin, np.int32, num_words_in_doc)
        word_indices_list.append(word_indices)
        word_cnts_list.append(word_cnts)
    fin.close()

    return word_indices_list, word_cnts_list, num_words


def text_vecs_to_bin(text_vecs_file, dst_bin_file):
    num_vecs, vec_len = 0, 0
    fin = open(text_vecs_file, 'rb')
    fout = open(dst_bin_file, 'wb')
    np.asarray([0, 0], np.int32).tofile(fout)
    for line in fin:
        vals = line.strip().split(' ')
        vec_len = len(vals)
        num_vecs += 1
        vec = list()
        for val in vals:
            vec.append(float(val))
        np.asarray(vec, np.float32).tofile(fout)
    fin.close()

    print '%d vecs, dim: %d' % (num_vecs, vec_len)
    fout.seek(0)
    np.asarray([num_vecs, vec_len], np.int32).tofile(fout)
    fout.close()

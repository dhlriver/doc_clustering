import gensim
import numpy as np
from scipy.sparse import lil_matrix
from sklearn import svm
import ioutils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from textclustering import cluster_accuracy


def __text_vecs_to_bin(text_vecs_file, dst_bin_file):
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


def text_vecs_to_bin_file():
    # text_vecs_file = 'e:/dc/baselines/rsm-0.1/nyt-hidden-20.txt'
    # bin_file = 'e:/dc/nyt-world-full/processed/vecs/rsm-vecs-20.bin'
    text_vecs_file = 'e:/dc/baselines/drbm_release/nyt-hidden-30.txt'
    bin_file = 'e:/dc/nyt-world-full/processed/vecs/drbm-vecs-30.bin'
    __text_vecs_to_bin(text_vecs_file, bin_file)


def test():
    docs_file_name = 'e:/dc/tmp.txt'
    documents = gensim.models.doc2vec.TaggedLineDocument(docs_file_name)
    for s in documents:
        print s


def main():
    # test()
    text_vecs_to_bin_file()
    # bow_classification()


if __name__ == '__main__':
    main()

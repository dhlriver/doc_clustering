from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.sparse import lil_matrix
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans

from textclustering import cluster_accuracy, purity
import ioutils


def _word_cnts_to_bow_vecs(indices_list, cnts_list, num_words, idf_dict):
    num_docs = len(indices_list)
    lm = lil_matrix((num_docs, num_words), dtype=np.float32)
    for idx, (indices, cnts) in enumerate(zip(indices_list, cnts_list)):
        num_words_in_doc = sum(cnts)
        for ix in xrange(len(indices)):
            # lm[idx, indices[ix]] = idf_dict[indices[ix]] * cnts[ix] / float(num_words_in_doc)
            lm[idx, indices[ix]] = cnts[ix] / float(num_words_in_doc)
    return lm.tocsr()


def _get_idf_values(word_indices_list, word_cnts_list, num_words):
    idfs = np.zeros(num_words, np.float32)
    for word_indices in word_indices_list:
        for i in xrange(len(word_indices)):
            idfs[word_indices[i]] += 1

    num_docs = len(word_indices_list)
    for i in xrange(len(idfs)):
        idfs[i] = np.log(num_docs / float(idfs[i] + 1))

    return idfs


def _bow_svm(train_bow_file_name, train_label_file_name, test_bow_file_name,
             test_label_file_name):
    print 'loading file ...'
    train_word_indices_list, train_word_cnts_list, num_words = ioutils.load_bow_file(train_bow_file_name, False)
    test_word_indices_list, test_word_cnts_list, num_words = ioutils.load_bow_file(test_bow_file_name, False)
    print num_words, 'words'
    idfs = _get_idf_values(train_word_indices_list, train_word_cnts_list, num_words)
    print idfs

    print 'to sparse ...'
    train_cm = _word_cnts_to_bow_vecs(train_word_indices_list, train_word_cnts_list, num_words, idfs)
    test_cm = _word_cnts_to_bow_vecs(test_word_indices_list, test_word_cnts_list, num_words, idfs)
    # print train_cm[0]

    train_y = ioutils.load_labels_file(train_label_file_name)
    test_y = ioutils.load_labels_file(test_label_file_name)

    print 'training svm ...'
    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.LinearSVC()
    clf.fit(train_cm, train_y)
    print 'done.'

    y_pred = clf.predict(test_cm)
    ftmp = open('e:/dc/20ng_data/tmp_labels.txt', 'wb')
    for i in xrange(len(y_pred)):
        ftmp.write(str(y_pred[i]) + '\t' + str(test_y[i]) + '\n')
    ftmp.close()
    print 'accuracy', accuracy_score(test_y, y_pred)
    print 'precision', precision_score(test_y, y_pred, average='macro')
    print 'recall', recall_score(test_y, y_pred, average='macro')
    print 'f1', f1_score(test_y, y_pred, average='macro')


def __get_bow_vecs(dw_file):
    print 'loading file ...'
    word_indices_list, word_cnts_list, num_words = ioutils.load_bow_file(dw_file)
    print num_words, 'words'

    idfs = _get_idf_values(word_indices_list, word_cnts_list, num_words)

    print 'to sparse ...'
    bow_vecs = _word_cnts_to_bow_vecs(word_indices_list, word_cnts_list, num_words, idfs)
    return bow_vecs


def bow_kmeans(bow_vecs, gold_labels, num_clusters):
    print 'performing kmeans ...'
    model = KMeans(n_clusters=num_clusters, n_jobs=4, n_init=20)
    model.fit(bow_vecs)

    print len(gold_labels), 'samples'
    print 'NMI: %f' % normalized_mutual_info_score(gold_labels, model.labels_)
    print 'Purity: %f' % purity(gold_labels, model.labels_)
    print 'Accuracy: %f' % cluster_accuracy(gold_labels, model.labels_)


def bow_classification():
    train_bow_file_name = 'e:/dc/20ng_bydate/bin/train_docs_dw_net.bin'
    test_bow_file_name = 'e:/dc/20ng_bydate/bin/test_docs_dw_net.bin'
    train_label_file = 'e:/dc/20ng_bydate/train_labels.bin'
    test_label_file = 'e:/dc/20ng_bydate/test_labels.bin'
    _bow_svm(train_bow_file_name, train_label_file, test_bow_file_name, test_label_file)


def bow_clustering():
    num_clusters = 5
    dw_file = 'e:/dc/nyt-world-full/processed/bin/dw-40.bin'
    gold_labels_file = 'e:/dc/nyt-world-full/processed/doc-labels.bin'

    gold_labels = ioutils.load_labels_file(gold_labels_file)
    bow_vecs = __get_bow_vecs(dw_file)
    for num_clusters in [20]:
        print num_clusters, 'clusters'
        bow_kmeans(bow_vecs, gold_labels, num_clusters)

if __name__ == '__main__':
    # bow_classification()
    bow_clustering()

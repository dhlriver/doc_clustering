from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.sparse import lil_matrix
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans

from textclustering import cluster_accuracy, purity, rand_index, write_clustering_perf_to_csv
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
    uint16_cnts = True
    train_word_indices_list, train_word_cnts_list, num_words = ioutils.load_bow_file(train_bow_file_name, uint16_cnts)
    test_word_indices_list, test_word_cnts_list, num_words = ioutils.load_bow_file(test_bow_file_name, uint16_cnts)
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
    # ftmp = open('e:/data/emadr/20ng_data/tmp_labels.txt', 'wb')
    # for i in xrange(len(y_pred)):
    #     ftmp.write(str(y_pred[i]) + '\t' + str(test_y[i]) + '\n')
    # ftmp.close()
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

    # print len(gold_labels), 'samples'

    nmi_score = normalized_mutual_info_score(gold_labels, model.labels_)
    purity_score = purity(gold_labels, model.labels_)
    ri_score = rand_index(gold_labels, model.labels_)

    # print 'NMI: %f' % normalized_mutual_info_score(gold_labels, model.labels_)
    # print 'Purity: %f' % purity(gold_labels, model.labels_)
    # print 'Accuracy: %f' % cluster_accuracy(gold_labels, model.labels_)
    print 'NMI: %f Purity: %f Rand index: %f' % (nmi_score, purity_score, ri_score)
    return nmi_score, purity_score, ri_score


def __bow_classification():
    # train_bow_file_name = 'e:/dc/20ng_bydate/bin/train_docs_dw_net.bin'
    # test_bow_file_name = 'e:/dc/20ng_bydate/bin/test_docs_dw_net.bin'
    # train_label_file = 'e:/dc/20ng_bydate/train_labels.bin'
    # test_label_file = 'e:/dc/20ng_bydate/test_labels.bin'

    min_occurrence = 50
    train_bow_file_name = 'e:/data/emadr/nyt-world-full/processed/bin/dw-train-%d.bin' % min_occurrence
    test_bow_file_name = 'e:/data/emadr/nyt-world-full/processed/bin/dw-test-%d.bin' % min_occurrence
    train_label_file = 'e:/data/emadr/nyt-world-full/processed/bin/train-labels.bin'
    test_label_file = 'e:/data/emadr/nyt-world-full/processed/bin/test-labels.bin'
    _bow_svm(train_bow_file_name, train_label_file, test_bow_file_name, test_label_file)


def __bow_clustering():
    # num_clusters_list = [5, 10, 15, 20]
    num_clusters_list = [5]

    dw_file = 'e:/data/emadr/nyt-world-full/processed/bin/dw-50.bin'
    gold_labels_file = 'e:/data/emadr/nyt-world-full/processed/doc-labels.bin'
    result_file = 'd:/documents/lab/paper-data/plot/bow-results-ri-bak.csv'

    dw_file = 'e:/data/emadr/20ng_bydate/bin/dw-test-50.bin'
    gold_labels_file = 'e:/data/emadr/20ng_bydate/test_labels.bin'
    result_file = 'd:/documents/lab/paper-data/plot/bow-results-20ng.csv'

    perf_list = list()
    gold_labels = ioutils.load_labels_file(gold_labels_file)
    print len(gold_labels), gold_labels[:10]
    bow_vecs = __get_bow_vecs(dw_file)
    print bow_vecs.shape
    for num_clusters in num_clusters_list:
        print num_clusters, 'clusters'
        nmi_score, purity_score, ri_score = bow_kmeans(bow_vecs, gold_labels, num_clusters)
        perf_list.append((num_clusters, nmi_score, purity_score, ri_score))
    write_clustering_perf_to_csv('BoW', perf_list, result_file)

if __name__ == '__main__':
    # __bow_classification()
    __bow_clustering()

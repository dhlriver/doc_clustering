import numpy as np
import os
from sklearn.metrics.cluster import normalized_mutual_info_score

import dataarange
from textclassification import doc_classification_lr, doc_classification_svm
from textclustering import topic_model_clustering
import ioutils
from textclustering import cluster_accuracy, purity, write_clustering_perf_to_csv, rand_index


def __word_legal(word):
    if word == '</s>':
        return True
    if not word.islower():
        return False
    if '_' in word or '#' in word or '/' in word or '@' in word:
        return False
    return True


def __load_words_in_word_vec_file(word_vec_file):
    f = open(word_vec_file, 'r')
    words = set()
    for line in f:
        vals = line.strip().split(' ')
        words.add(vals[0])
    f.close()
    return words


def __match_docs_with_word_vecs():
    # docs_file = 'e:/data/emadr/20ng_bydate/tokenizedlc/docs-tokenized-lc-30.txt'
    # matched_docs_file = 'e:/data/emadr/20ng_bydate/tokenizedlc/docs-tokenized-lc-30-wv.txt'
    minoc = 70
    mc = 'world'
    docs_file = 'e:/data/emadr/nyt-less-docs/%s/tokenizedlc/docs-tokenized-lc-%d.txt' % (mc, minoc)
    matched_docs_file = 'e:/data/emadr/nyt-less-docs/%s/tokenizedlc/docs-tokenized-lc-%d-wv.txt' % (mc, minoc)
    word_vec_file = 'e:/data/common-res/googlenews-vecs-lc.txt'
    vocab = __load_words_in_word_vec_file(word_vec_file)
    print len(vocab), 'words loaded'
    f = open(docs_file, 'r')
    fout = open(matched_docs_file, 'wb')
    for idx, line in enumerate(f):
        words = line.strip().split(' ')
        words_line = ''
        for word in words:
            if word not in vocab:
                continue
            words_line += '%s ' % word
        if not words_line:
            print 'doc %d has no words' % idx
            words_line = '</s>'
        fout.write('%s\n' % words_line.strip())
    f.close()
    fout.close()


def __text_wordvec_to_bin():
    text_wordvec_file = 'e:/data/common-res/googlenews-vecs-lc.txt'
    dst_words_file = 'e:/data/common-res/googlenews-vecs-lc-words.txt'
    dst_bin_file = 'e:/data/common-res/googlenews-vecs-lc.bin'
    dim = -1
    f = open(text_wordvec_file, 'r')
    fout0 = open(dst_words_file, 'wb')
    fout1 = open(dst_bin_file, 'wb')
    np.zeros(2, np.int32).tofile(fout1)
    word_cnt = 0
    for line in f:
        vals = line.strip().split(' ')
        curdim = len(vals) - 1
        assert dim == -1 or curdim != dim
        dim = curdim
        fout0.write('%s\n' % vals[0])
        vec = np.zeros(curdim, np.float32)
        for i, v in enumerate(vals[1:]):
            vec[i] = float(v)
        vec.tofile(fout1)
        word_cnt += 1
    fout1.seek(0, )
    f.close()
    fout0.close()
    fout1.close()


def __filter_google_word_vecs():
    wordvec_file = 'e:/data/common-res/GoogleNews-vectors-negative300.txt'
    dst_word_vec_file = 'e:/data/common-res/googlenews-vecs-lc.txt'
    f = open(wordvec_file, 'r')
    fout = open(dst_word_vec_file, 'wb')
    head_line = f.next()
    print head_line.strip()
    word_cnt = 0
    for i, line in enumerate(f):
        vals = line.strip().split(' ')
        word = vals[0]
        if not __word_legal(word):
            continue
        fout.write(line)
        word_cnt += 1
        if i % 10000 == 0:
            print i, word_cnt
    f.close()
    fout.close()
    print word_cnt, 'words'


def __job_text_vecs_to_bin_file():
    # datadir = 'e:/data/emadr/nyt-all/business/rsm'
    # datadir = 'e:/data/emadr/20ng_bydate/lftm'
    mc = 'world'
    datadir = 'e:/data/emadr/nyt-less-docs/%s' % mc
    minoc = 100
    numtopics = 10
    inititers = 500
    iters = 50
    # text_vecs_file = os.path.join(datadir, 'LFLDA-%d-%d-%d.theta' % (numtopics, inititers, iters))
    text_vecs_file = os.path.join(datadir, 'lftm/LFLDA-%s-%d-%d-%d.theta' % (mc, numtopics, inititers, iters))
    binfile = os.path.join(datadir, 'lftm/lftm-vecs-%d.bin' % numtopics)
    # text_vecs_file = os.path.join(datadir, 'drbm-vecs-100-30.txt')
    # binfile = os.path.join(datadir, 'drbm-vecs.bin')
    # ioutils.text_vecs_to_bin(text_vecs_file, binfile)

    split_labels_file_name = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_vecs_file_name = os.path.join(datadir, 'lftm/train-lftm-vecs-%d.bin' % numtopics)
    test_vecs_file_name = os.path.join(datadir, 'lftm/test-lftm-vecs-%d.bin' % numtopics)

    dataarange.split_vecs(binfile, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)


def __classification():
    datadir = 'e:/data/emadr/20ng_bydate'
    # datadir = 'e:/data/emadr/nyt-less-docs/business'
    numtopics = 20
    all_vecs_file_name = os.path.join(datadir, 'lftm/lftm-vecs-%d.bin' % numtopics)
    # all_vecs_file_name = os.path.join(datadir, 'rsm/drbm-vecs.bin')
    # split_labels_file_name = os.path.join(datadir, 'bindata/data-split-labels.bin')
    split_labels_file_name = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_label_file = os.path.join(datadir, 'bindata/train-labels.bin')
    test_label_file = os.path.join(datadir, 'bindata/test-labels.bin')
    train_vecs_file_name = os.path.join(datadir, 'lftm/train-lftm-vecs-%d.bin' % numtopics)
    test_vecs_file_name = os.path.join(datadir, 'lftm/test-lftm-vecs-%d.bin' % numtopics)

    dataarange.split_vecs(all_vecs_file_name, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)
    # doc_classification_lr(train_vecs_file_name, train_label_file, test_vecs_file_name,
    #                       test_label_file, 0, -1)
    # doc_classification_svm(train_vecs_file_name, train_label_file, test_vecs_file_name,
    #                        test_label_file, 0, -1)


def __job_clustering():
    # datadir = 'e:/data/emadr/20ng_bydate/'
    datadir = 'e:/data/emadr/nyt-less-docs/world'
    numtopics = 20
    labels_file = os.path.join(datadir, 'bindata/test-labels.bin')
    topic_vecs_file = os.path.join(datadir, 'lftm/test-lftm-vecs-%d.bin' % numtopics)

    topic_vecs = ioutils.load_vec_list_file(topic_vecs_file)
    gold_labels = ioutils.load_labels_file(labels_file)
    sys_labels = topic_model_clustering(topic_vecs)

    nmi_score = normalized_mutual_info_score(gold_labels, sys_labels)
    purity_score = purity(gold_labels, sys_labels)
    # ri_score = rand_index(gold_labels, sys_labels)
    ri_score = 0

    print 'NMI: %f Purity: %f Rand index: %f' % (nmi_score, purity_score, ri_score)
    # print 'Accuracy: %f' % cluster_accuracy(labels, model.labels_)

    print '%f\t%f\t%f' % (nmi_score, purity_score, ri_score)


if __name__ == '__main__':
    # __filter_google_word_vecs()
    # __match_docs_with_word_vecs()

    # __job_text_vecs_to_bin_file()
    # __classification()
    __job_clustering()

import gensim
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import os
from itertools import izip

import ioutils
from textclustering import cluster_accuracy, purity, write_clustering_perf_to_csv, rand_index
from textclassification import doc_classification_lr, doc_classification_svm


def __train_lda_model(dict_file, mm_corpus_file, dst_model_file, num_topics=20):
    print '%d topics' % num_topics
    word_dict = gensim.corpora.Dictionary.load(dict_file)
    mm_corpus = gensim.corpora.MmCorpus(mm_corpus_file)
    # lda_model = gensim.models.ldamodel.LdaModel(mm_corpus, id2word=word_dict, num_topics=num_topics)
    lda_model = gensim.models.ldamodel.LdaModel(mm_corpus, id2word=word_dict, num_topics=num_topics, iterations=100,
                                                alpha=0.1)
    lda_model.save(dst_model_file)


def __eval_lda_clustering(lda_model, mm_corpus, gold_labels):
    # lda_model = gensim.models.ldamodel.LdaModel.load(model_file)
    sys_labels = list()
    for i, doc in enumerate(mm_corpus):
        topic_dist = lda_model[doc]
        # print topic_dist
        cluster_idx = 0
        max_dist = 0
        for tup in topic_dist:
            if tup[1] > max_dist:
                cluster_idx = tup[0]
                max_dist = tup[1]
        sys_labels.append(cluster_idx)
        if len(sys_labels) % 5000 == 0:
            print len(sys_labels)
        # if i > 10:
        #     break
    # print len(sys_labels)
    # print len(gold_labels)

    nmi_score = normalized_mutual_info_score(gold_labels, sys_labels)
    purity_score = purity(gold_labels, sys_labels)
    ri_score = rand_index(gold_labels, sys_labels)

    # print 'NMI: %f' % normalized_mutual_info_score(gold_labels, sys_labels)
    # print 'Purity: %f' % purity(gold_labels, sys_labels)
    # print 'Accuracy: %f' % cluster_accuracy(gold_labels, sys_labels)

    print 'NMI: %f Purity: %f Rand index: %f' % (nmi_score, purity_score, ri_score)
    return nmi_score, purity_score, ri_score


def __eval_lda_clustering_20ng():
    text_doc_file = 'e:/dc/20ng_bydate/twe/docs-nl.txt'
    dict_file = 'e:/dc/20ng_bydate/lda/all-docs.dict'
    mm_file = 'e:/dc/20ng_bydate/lda/all-docs.mm'
    lda_model_file = 'e:/dc/20ng_bydate/lda/lda-model'

    dataset_label_file = 'e:/dc/20ng_bydate/doc_split_labels.bin'
    test_label_file = 'e:/dc/20ng_bydate/test_labels.bin'

    # __text_file_to_mm_corpus(text_doc_file, dict_file, mm_file)

    __train_lda_model(dict_file, mm_file, lda_model_file)

    dataset_labels = ioutils.load_labels_file(dataset_label_file)
    lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_file)
    mm_corpus = gensim.corpora.MmCorpus(mm_file)
    sys_labels = list()
    for i, doc in enumerate(mm_corpus):
        if dataset_labels[i] == 0:
            continue

        topic_dist = lda_model[doc]
        # print topic_dist
        cluster_idx = 0
        max_dist = 0
        for tup in topic_dist:
            if tup[1] > max_dist:
                cluster_idx = tup[0]
                max_dist = tup[1]
        sys_labels.append(cluster_idx)
        if len(sys_labels) % 1000 == 0:
            print len(sys_labels)
        # if i > 10:
        #     break
    print len(sys_labels)
    gold_labels = ioutils.load_labels_file(test_label_file)
    print len(gold_labels)
    print normalized_mutual_info_score(gold_labels, sys_labels)
    print cluster_accuracy(gold_labels, sys_labels)


def __text_file_to_mm_corpus(text_docs_file, dict_file, mm_file):
    fin = open(text_docs_file, 'rb')
    text_corpus = gensim.corpora.TextCorpus(fin)
    print 'saving dict ...'
    text_corpus.dictionary.save(dict_file)
    print 'saving mm corpus ...'
    gensim.corpora.MmCorpus.save_corpus(mm_file, text_corpus)
    fin.close()


def __job_build_corpus_train():
    mcs = ['arts', 'business', 'sports', 'world']
    nums_topics = [4, 4, 11, 5]
    for min_occurrence in [30]:
        for mc, num_topics in izip(mcs, nums_topics):
            datadir = 'e:/data/emadr/nyt-less-docs/%s' % mc
            print mc, num_topics
            # num_topics = 100

            text_doc_file = os.path.join(datadir, 'tokenizedlc/docs-tokenized-lc-%d.txt' % min_occurrence)
            dict_file = os.path.join(datadir, 'lda/all-docs-%d.dict' % min_occurrence)
            mm_file = os.path.join(datadir, 'lda/all-docs-%d.mm' % min_occurrence)

            lda_model_file = os.path.join(datadir, 'lda/lda-model-%d-%d.bin' % (num_topics, min_occurrence))

            __text_file_to_mm_corpus(text_doc_file, dict_file, mm_file)

            __train_lda_model(dict_file, mm_file, lda_model_file, num_topics=num_topics)

            data_split_labels_file = os.path.join(datadir, 'bindata/dataset-split-labels.bin')

            train_vecs_file = os.path.join(datadir, 'lda/train-vecs-%d-%d.bin' % (num_topics, min_occurrence))
            val_vecs_file = os.path.join(datadir, 'lda/val-vecs-%d-%d.bin' % (num_topics, min_occurrence))
            test_vecs_file = os.path.join(datadir, 'lda/test-vecs-%d-%d.bin' % (num_topics, min_occurrence))

            __gen_lda_features(data_split_labels_file, mm_file, lda_model_file, train_vecs_file,
                               val_vecs_file, test_vecs_file)


def __job_text_file_to_mm_corpus():
    # datadir = 'e:/data/emadr/nyt-world-full/processed/'
    # datadir = 'e:/data/emadr/nyt-all/world/'
    datadir = 'e:/data/emadr/20ng_bydate/'
    min_occurrence = 30
    text_doc_file = os.path.join(datadir, 'tokenizedlc/docs-tokenized-lc-%d.txt' % min_occurrence)
    dict_file = os.path.join(datadir, 'lda/all-docs-%d.dict' % min_occurrence)
    mm_file = os.path.join(datadir, 'lda/all-docs-%d.mm' % min_occurrence)
    __text_file_to_mm_corpus(text_doc_file, dict_file, mm_file)


def __job_train_lda_model():
    datadir = 'e:/data/emadr/nyt-less-docs/world/'
    # datadir = 'e:/data/emadr/20ng_bydate/'
    for num_topics in [10, 15, 20]:
    # num_topics = 10
        min_occurrence = 30
        dict_file = os.path.join(datadir, 'lda/all-docs-%d.dict' % min_occurrence)
        mm_file = os.path.join(datadir, 'lda/all-docs-%d.mm' % min_occurrence)
        lda_model_file = os.path.join(datadir, 'lda/lda-model-%d-%d.bin' % (num_topics, min_occurrence))

        __train_lda_model(dict_file, mm_file, lda_model_file, num_topics=num_topics)

        data_split_labels_file = os.path.join(datadir, 'bindata/dataset-split-labels.bin')

        train_vecs_file = os.path.join(datadir, 'lda/train-vecs-%d-%d.bin' % (num_topics, min_occurrence))
        val_vecs_file = os.path.join(datadir, 'lda/val-vecs-%d-%d.bin' % (num_topics, min_occurrence))
        test_vecs_file = os.path.join(datadir, 'lda/test-vecs-%d-%d.bin' % (num_topics, min_occurrence))

        __gen_lda_features(data_split_labels_file, mm_file, lda_model_file, train_vecs_file,
                           val_vecs_file, test_vecs_file)


def __save_vecs(vecs, dst_file):
    fout = open(dst_file, 'wb')
    np.asarray([len(vecs), len(vecs[0])], np.int32).tofile(fout)
    for vec in vecs:
        vec.tofile(fout)
    fout.close()


def __gen_lda_features(data_split_labels_file, mm_file, lda_model_file, dst_train_vecs_file,
                       dst_val_vecs_file, dst_test_vecs_file):
    data_split_labels = ioutils.load_labels_file(data_split_labels_file)
    lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_file)
    mm_corpus = gensim.corpora.MmCorpus(mm_file)
    train_vecs, val_vecs, test_vecs = list(), list(), list()
    for i, (l, doc) in enumerate(izip(data_split_labels, mm_corpus)):
        topic_dist = lda_model[doc]
        vec = np.zeros(lda_model.num_topics, np.float32)
        for tup in topic_dist:
            vec[tup[0]] = tup[1]

        if l == 0:
            train_vecs.append(vec)
        elif l == 1:
            val_vecs.append(vec)
        else:
            test_vecs.append(vec)
        # print topic_dist
        # print vec
        if i % 1000 == 0:
            print i
            # break
    # print train_vecs[:5]
    __save_vecs(train_vecs, dst_train_vecs_file)
    __save_vecs(val_vecs, dst_val_vecs_file)
    __save_vecs(test_vecs, dst_test_vecs_file)


def __lda_classification():
    datadir = 'e:/data/emadr/nyt-less-docs/business/'
    # datadir = 'e:/data/emadr/20ng_bydate/'
    num_topics = 100
    min_occurrence = 30
    dict_file = os.path.join(datadir, 'lda/all-docs-%d.dict' % min_occurrence)
    mm_file = os.path.join(datadir, 'lda/all-docs-%d.mm' % min_occurrence)
    lda_model_file = os.path.join(datadir, 'lda/lda-model-%d-%d.bin' % (num_topics, min_occurrence))
    # data_split_labels_file = os.path.join(datadir, 'data-split-labels.bin')
    data_split_labels_file = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_labels_file = os.path.join(datadir, 'bindata/train-labels.bin')
    test_labels_file = os.path.join(datadir, 'bindata/test-labels.bin')

    train_vecs_file = os.path.join(datadir, 'lda/train-vecs-%d-%d.bin' % (num_topics, min_occurrence))
    val_vecs_file = os.path.join(datadir, 'lda/val-vecs-%d-%d.bin' % (num_topics, min_occurrence))
    test_vecs_file = os.path.join(datadir, 'lda/test-vecs-%d-%d.bin' % (num_topics, min_occurrence))

    # __gen_lda_features(data_split_labels_file, mm_file, lda_model_file, train_vecs_file,
    #                    val_vecs_file, test_vecs_file)
    # doc_classification_lr(train_vecs_file, train_labels_file, test_vecs_file,
    #                       test_labels_file, 0, -1)
    doc_classification_svm(train_vecs_file, train_labels_file, test_vecs_file,
                           test_labels_file, 0, -1)


def __lda_clustering_nyt():
    num_clusters_list = [5, 10, 15, 20]
    min_occurrence = 30
    datadir = 'e:/data/emadr/nyt-less-docs/world/'
    result_file = 'd:/documents/lab/paper-data/plot/lda-results-ri.csv'

    dict_file = os.path.join(datadir, 'lda/all-docs-%d.dict' % min_occurrence)
    mm_file = os.path.join(datadir, 'lda/all-docs-%d.mm' % min_occurrence)
    gold_label_file = 'e:/data/emadr/nyt-less-docs/world/bindata/test-labels.bin'

    # __text_file_to_mm_corpus(text_doc_file, dict_file, mm_file)

    perf_list = list()
    gold_labels = ioutils.load_labels_file(gold_label_file)
    word_dict = gensim.corpora.Dictionary.load(dict_file)
    mm_corpus = gensim.corpora.MmCorpus(mm_file)
    for num_clusters in num_clusters_list:
        print num_clusters, 'clusters'
        lda_model = gensim.models.ldamodel.LdaModel(mm_corpus, id2word=word_dict, num_topics=num_clusters)
        lda_model_file = 'e:/dc/nyt-world-full/processed/lda/lda-model-%d' % num_clusters
        lda_model.save(lda_model_file)

        nmi_score, purity_score, ri_score = __eval_lda_clustering(lda_model, mm_corpus, gold_labels)
        perf_list.append((num_clusters, nmi_score, purity_score, ri_score))

    write_clustering_perf_to_csv('LDA', perf_list, result_file)

    # print 'training lda ...'
    # __train_lda_model(dict_file, mm_file, lda_model_file, num_topics=10)

    # __eval_lda_clustering(lda_model_file, mm_file, gold_label_file)

    # print lda_model.print_topics(5)
    # print len(text_corpus.dictionary)
    # print lda_model.print_topics(5)


def __lda_clustering():
    num_topics = 20
    min_occurrence = 30
    # datadir = 'e:/data/emadr/20ng_bydate/'
    # labels_file = os.path.join(datadir, 'bindata/test-labels.bin')
    # topic_vecs_file = os.path.join(datadir, 'lda/test-vecs-%d-%d.bin' % (num_topics, min_occurrence))
    datadir = 'e:/data/emadr/nyt-less-docs/world'
    labels_file = os.path.join(datadir, 'bindata/test-labels.bin')
    topic_vecs_file = os.path.join(datadir, 'lda/test-vecs-%d-%d.bin' % (num_topics, min_occurrence))

    topic_vecs = ioutils.load_vec_list_file(topic_vecs_file)
    gold_labels = ioutils.load_labels_file(labels_file)
    sys_labels = list()
    for i, topic_vec in enumerate(topic_vecs):
        cluster_idx = 0
        max_dist = 0
        for j, v in enumerate(topic_vec):
            if v > max_dist:
                cluster_idx = j
                max_dist = v
        # print cluster_idx, max_dist
        sys_labels.append(cluster_idx)
        if len(sys_labels) % 5000 == 0:
            print len(sys_labels)

    nmi_score = normalized_mutual_info_score(gold_labels, sys_labels)
    purity_score = purity(gold_labels, sys_labels)
    # ri_score = rand_index(gold_labels, sys_labels)
    ri_score = 0

    print 'NMI: %f Purity: %f Rand index: %f' % (nmi_score, purity_score, ri_score)
    # print 'Accuracy: %f' % cluster_accuracy(labels, model.labels_)

    print '%f\t%f\t%f' % (nmi_score, purity_score, ri_score)


if __name__ == '__main__':
    # __job_build_corpus_train()
    # __text_file_to_mm_corpus()
    __job_train_lda_model()

    # __lda_clustering_nyt()
    # __lda_classification()
    # __lda_clustering()

import gensim
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np

import ioutils
from textclustering import cluster_accuracy, purity


def __text_file_to_mm_corpus(text_file, dst_dict_file, dst_mm_corpus_file):
    fin = open(text_file, 'rb')
    text_corpus = gensim.corpora.TextCorpus(fin)
    text_corpus.dictionary.save(dst_dict_file)
    gensim.corpora.MmCorpus.save_corpus(dst_mm_corpus_file, text_corpus)
    fin.close()


def __train_lda_model(dict_file, mm_corpus_file, dst_model_file, num_topics=20):
    print '%d topics' % num_topics
    word_dict = gensim.corpora.Dictionary.load(dict_file)
    mm_corpus = gensim.corpora.MmCorpus(mm_corpus_file)
    lda_model = gensim.models.ldamodel.LdaModel(mm_corpus, id2word=word_dict, num_topics=num_topics)
    lda_model.save(dst_model_file)


def __eval_lda_clustering_20ng():
    text_doc_file = 'e:/dc/20ng_bydate/twe/docs-nl.txt'
    dict_file = 'e:/dc/20ng_bydate/lda/all-docs.dict'
    mm_file = 'e:/dc/20ng_bydate/lda/all-docs.mm'
    lda_model_file = 'e:/dc/20ng_bydate/lda/lda-model'

    dataset_label_file = 'e:/dc/20ng_bydate/doc_split_labels.bin'
    test_label_file = 'e:/dc/20ng_bydate/test_labels.bin'

    __text_file_to_mm_corpus(text_doc_file, dict_file, mm_file)

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


def __eval_lda_clustering(model_file, mm_file, gold_label_file):
    lda_model = gensim.models.ldamodel.LdaModel.load(model_file)
    mm_corpus = gensim.corpora.MmCorpus(mm_file)
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
        if len(sys_labels) % 1000 == 0:
            print len(sys_labels)
        # if i > 10:
        #     break
    print len(sys_labels)
    gold_labels = ioutils.load_labels_file(gold_label_file)
    print len(gold_labels)
    print 'NMI: %f' % normalized_mutual_info_score(gold_labels, sys_labels)
    print 'Purity: %f' % purity(gold_labels, sys_labels)
    print 'Accuracy: %f' % cluster_accuracy(gold_labels, sys_labels)


def lda():
    text_doc_file = 'e:/dc/nyt-world-full/processed/docs_tokenized_lc.txt'
    dict_file = 'e:/dc/nyt-world-full/processed/lda/all-docs.dict'
    mm_file = 'e:/dc/nyt-world-full/processed/lda/all-docs.mm'
    lda_model_file = 'e:/dc/nyt-world-full/processed/lda/lda-model'
    gold_label_file = 'e:/dc/nyt-world-full/processed/doc-labels.bin'

    # __text_file_to_mm_corpus(text_doc_file, dict_file, mm_file)

    print 'training lda ...'
    __train_lda_model(dict_file, mm_file, lda_model_file, num_topics=10)

    __eval_lda_clustering(lda_model_file, mm_file, gold_label_file)

    # print lda_model.print_topics(5)
    # print len(text_corpus.dictionary)
    # print lda_model.print_topics(5)

if __name__ == '__main__':
    lda()

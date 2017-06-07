import gensim
import numpy as np
import os
import ioutils
import dataarange
from textclassification import doc_classification_svm, doc_classification_lr, get_scores_label_file


def save_doc2vec_vectors(model, dst_vecs_file_name):
    # model = gensim.models.doc2vec.Doc2Vec.load(model_file_name)
    # print type(model.docvecs[0])
    fout = open(dst_vecs_file_name, 'wb')
    num_vecs = len(model.docvecs)
    vec_len = len(model.docvecs[0])
    print num_vecs, vec_len
    np.asarray([num_vecs, vec_len], np.int32).tofile(fout)
    for i in xrange(len(model.docvecs)):
        model.docvecs[i].tofile(fout)
    fout.close()


def train_doc_vectors(docs_file_name, min_count=5, def_alpha=0.005, ns=0, niter=20, dm=0):
    documents = gensim.models.doc2vec.TaggedLineDocument(docs_file_name)
    print 'initing ...'
    # model = gensim.models.doc2vec.Doc2Vec(documents, alpha=def_alpha, min_count=1, hs=1, negative=5,
    #                                       size=400, workers=3, dm=0)
    # model = gensim.models.doc2vec.Doc2Vec(documents, alpha=def_alpha, min_count=1, hs=0, negative=10,
    #                                       size=100, workers=2, dm=1)
    if dm == 0:
        model = gensim.models.doc2vec.Doc2Vec(documents, min_count=min_count, alpha=def_alpha, size=100, workers=4, dm=dm,
                                              iter=niter)
    else:
        # def_alpha = 0.001
        model = gensim.models.doc2vec.Doc2Vec(documents, min_count=min_count, alpha=def_alpha, size=100, workers=4,
                                              dm=dm, negative=ns, iter=niter)
    print 'vocab size: ', len(model.vocab)
    print len(model.index2word)
    print len(model.docvecs), 'docs'

    # for i in xrange(5):
    #     print 'epoch', i, model.alpha
    #     model.train(documents)
    #     model.alpha *= 0.9

    # sims = model.docvecs.most_similar(0)
    # print sims
    return model


def __train_pv_20ng():
    docs_file = 'e:/data/emadr/20ng_bydate/tokenizedlc/docs-tokenized-lc-2.txt'
    method = 'pvdm'
    dst_vecs_file = 'e:/data/emadr/20ng_bydate/bindata/%s-vecs.bin' % method

    niters = 40
    def_alpha = 0.01
    min_count = 5
    ns = 0
    dm = 1 if method == 'pvdm' else 0
    model = train_doc_vectors(docs_file, min_count=min_count, def_alpha=def_alpha, ns=ns, dm=dm,
                              niter=niters)
    # dst_vecs_file = 'e:/data/emadr/20ng_bydate/bindata/pvdbow-vecs.bin'
    save_doc2vec_vectors(model, dst_vecs_file)

    split_labels_file_name = 'e:/data/emadr/20ng_bydate/bindata/dataset-split-labels.bin'
    train_vecs_file_name = 'e:/data/emadr/20ng_bydate/bindata/train-%s-vecs.bin' % method
    test_vecs_file_name = 'e:/data/emadr/20ng_bydate/bindata/test-%s-vecs.bin' % method
    dataarange.split_vecs(dst_vecs_file, split_labels_file_name, train_vecs_file_name, test_vecs_file_name,
                          train_label=0, test_label=2)


def __train_pv_nyt():
    method = 'pvdm'
    # data_dir = 'e:/data/emadr/nyt-world-full/processed/'
    data_dir = 'e:/data/emadr/nyt-less-docs/business/'
    docs_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-2.txt')
    dst_vecs_file = os.path.join(data_dir, 'bindata/%s-vecs.bin' % method)

    niters = 40
    def_alpha = 0.01
    min_count = 5
    ns = 0
    dm = 1 if method == 'pvdm' else 0
    model = train_doc_vectors(docs_file, min_count=min_count, def_alpha=def_alpha, ns=ns, dm=dm,
                              niter=niters)
    save_doc2vec_vectors(model, dst_vecs_file)

    split_labels_file_name = os.path.join(data_dir, 'bindata/dataset-split-labels.bin')
    train_vecs_file_name = os.path.join(data_dir, 'bindata/train-%s-vecs.bin' % method)
    test_vecs_file_name = os.path.join(data_dir, 'bindata/test-%s-vecs.bin' % method)
    dataarange.split_vecs(dst_vecs_file, split_labels_file_name, train_vecs_file_name, test_vecs_file_name,
                          train_label=0, test_label=2)


def __classification():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed/'
    # data_dir = 'e:/data/emadr/nyt-all/arts/'
    # data_dir = 'e:/data/emadr/nyt-all/business/'
    # data_dir = 'e:/data/emadr/nyt-less-docs/business/bindata/'
    data_dir = 'e:/data/emadr/20ng_bydate/bindata/'
    # method = 'pvdm'
    method = 'pvdbow'

    all_vecs_file = os.path.join(data_dir, '%s-vecs.bin' % method)
    split_labels_file_name = os.path.join(data_dir, 'dataset-split-labels.bin')
    train_label_file = os.path.join(data_dir, 'train-labels.bin')
    test_label_file = os.path.join(data_dir, 'test-labels.bin')
    train_vecs_file_name = os.path.join(data_dir, 'train-%s-vecs.bin' % method)
    test_vecs_file_name = os.path.join(data_dir, 'test-%s-vecs.bin' % method)
    dst_y_pred_file = os.path.join(data_dir, 'ypred-%s.bin' % method)

    dataarange.split_vecs(all_vecs_file, split_labels_file_name, train_vecs_file_name, test_vecs_file_name,
                          train_label=0, test_label=2)
    # doc_classification_lr(train_vecs_file_name, train_label_file, test_vecs_file_name,
    #                       test_label_file, 0, -1)
    y_pred_test = doc_classification_svm(train_vecs_file_name, train_label_file, test_vecs_file_name, 0, -1)
    get_scores_label_file(test_label_file, y_pred_test)
    ioutils.save_labels(y_pred_test, dst_y_pred_file)


def __job_train_classification():
    docs_file = 'e:/data/emadr/20ng_bydate/tokenizedlc/docs-tokenized-lc-2.txt'
    # method = 'pvdm'
    method = 'pvdbow'
    dm = 1 if method == 'pvdm' else 0
    # dst_vecs_file = 'e:/data/emadr/20ng_bydate/bindata/pvdbow-vecs.bin'
    dst_vecs_file = 'e:/data/emadr/20ng_bydate/bindata/%s-vecs.bin' % method

    dst_result_file = 'e:/data/emadr/20ng_bydate/pvdm-results.txt'

    split_labels_file_name = 'e:/data/emadr/20ng_bydate/bindata/dataset-split-labels.bin'
    train_vecs_file_name = 'e:/data/emadr/20ng_bydate/bindata/train-%s-vecs.bin' % method
    test_vecs_file_name = 'e:/data/emadr/20ng_bydate/bindata/test-%s-vecs.bin' % method

    data_dir = 'e:/data/emadr/20ng_bydate/bindata/'
    train_label_file = os.path.join(data_dir, 'train-labels.bin')
    test_label_file = os.path.join(data_dir, 'test-labels.bin')

    # min_counts = [2, 5, 10, 20]
    # def_alphas = [0.1, 0.01, 0.001]
    # nss = [0, 5, 10, 15]
    min_counts = [5]
    def_alphas = [0.01]
    nss = [0]
    niters = 100
    fout = open(dst_result_file, 'wb')
    for min_count in min_counts:
        for def_alpha in def_alphas:
            for ns in nss:
                model = train_doc_vectors(docs_file, min_count=min_count, def_alpha=def_alpha, ns=ns, dm=dm,
                                          niter=niters)
                save_doc2vec_vectors(model, dst_vecs_file)
                dataarange.split_vecs(dst_vecs_file, split_labels_file_name, train_vecs_file_name, test_vecs_file_name,
                                      train_label=0, test_label=2)

                y_pred_test = doc_classification_svm(train_vecs_file_name, train_label_file,
                                                     test_vecs_file_name, 0, -1)
                acc, prec, recall, f1 = get_scores_label_file(test_label_file, y_pred_test)
                print '%d\t%f\t%d\t%d' % (min_count, def_alpha, ns, niters)
                fout.write('%d\t%f\t%d\n' % (min_count, def_alpha, ns))
                fout.write('%f\t%f\t%f\t%f\n' % (acc, prec, recall, f1))
                fout.flush()
    fout.close()

if __name__ == '__main__':
    # __job_train_classification()
    # __train_pv_20ng()
    # __train_pv_nyt()
    __classification()

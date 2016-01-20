import gensim
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics import f1_score
from sklearn import svm
import io_utils


def get_idf_values(word_indices_list, word_cnts_list, num_words):
    idfs = np.zeros(num_words, np.float32)
    for word_indices in word_indices_list:
        for i in xrange(len(word_indices)):
            idfs[word_indices[i]] += 1

    num_docs = len(word_indices_list)
    for i in xrange(len(idfs)):
        idfs[i] = np.log(num_docs / float(idfs[i] + 1))

    return idfs


def bow_svm(train_bow_file_name, train_label_file_name, test_bow_file_name,
            test_label_file_name):
    print 'loading file ...'
    train_word_indices_list, train_word_cnts_list, num_words = io_utils.load_bow_file(train_bow_file_name)
    test_word_indices_list, test_word_cnts_list, num_words = io_utils.load_bow_file(test_bow_file_name)
    print num_words, 'words'
    idfs = get_idf_values(train_word_indices_list, train_word_cnts_list, num_words)
    print idfs

    def to_csr_matrix(indices_list, cnts_list):
        num_docs = len(indices_list)
        lm = lil_matrix((num_docs, num_words), dtype=np.float32)
        for idx, (indices, cnts) in enumerate(zip(indices_list, cnts_list)):
            num_words_in_doc = sum(cnts)
            for ix in xrange(len(indices)):
                lm[idx, indices[ix]] = idfs[indices[ix]] * cnts[ix] / float(num_words_in_doc)
        return lm.tocsr()

    print 'to sparse ...'
    train_cm = to_csr_matrix(train_word_indices_list, train_word_cnts_list)
    test_cm = to_csr_matrix(test_word_indices_list, test_word_cnts_list)
    # print train_cm[0]

    train_y = io_utils.load_labels_file(train_label_file_name)
    test_y = io_utils.load_labels_file(test_label_file_name)

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
    print f1_score(test_y, y_pred, average='micro')
    print f1_score(test_y, y_pred, average='macro')


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


def train_doc_vectors(docs_file_name):
    documents = gensim.models.doc2vec.TaggedLineDocument(docs_file_name)
    def_alpha = 0.075
    print 'initing ...'
    model = gensim.models.doc2vec.Doc2Vec(documents, alpha=def_alpha, min_count=1, hs=0, negative=5,
                                          size=200, workers=4, dm=0)
    print 'vocab size: ', len(model.vocab)
    print len(model.index2word)
    print len(model.docvecs), 'docs'

    for i in xrange(10):
        print 'epoch', i, model.alpha
        model.train(documents)

    # sims = model.docvecs.most_similar(0)
    # print sims
    return model


def train_20ng_doc_vectors():
    docs_file_name = 'e:dc/20ng_bydate/doc_text_data.txt'
    # docs_file_name = 'e:dc/20ng_data/all_docs_wi.txt'
    dst_vecs_file_name = 'e:dc/20ng_bydate/vecs/dbow_doc_vec_100.bin'
    model = train_doc_vectors(docs_file_name)
    save_doc2vec_vectors(model, dst_vecs_file_name)


def bow_classification():
    train_bow_file_name = 'e:/dc/20ng_bydate/train_docs_dw_net.bin'
    test_bow_file_name = 'e:/dc/20ng_bydate/test_docs_dw_net.bin'
    train_label_file = 'e:/dc/20ng_bydate/train_labels.bin'
    test_label_file = 'e:/dc/20ng_bydate/test_labels.bin'
    bow_svm(train_bow_file_name, train_label_file, test_bow_file_name, test_label_file)


def test():
    docs_file_name = 'e:/dc/tmp.txt'
    documents = gensim.models.doc2vec.TaggedLineDocument(docs_file_name)
    for s in documents:
        print s


def main():
    # test()
    train_20ng_doc_vectors()
    # bow_classification()


if __name__ == '__main__':
    main()

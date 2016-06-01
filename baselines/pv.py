import gensim
import numpy as np
import ioutils


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
    def_alpha = 0.005
    print 'initing ...'
    # model = gensim.models.doc2vec.Doc2Vec(documents, alpha=def_alpha, min_count=1, hs=1, negative=5,
    #                                       size=400, workers=3, dm=0)
    # model = gensim.models.doc2vec.Doc2Vec(documents, alpha=def_alpha, min_count=1, hs=0, negative=10,
    #                                       size=100, workers=2, dm=1)
    model = gensim.models.doc2vec.Doc2Vec(documents, min_count=2, alpha=def_alpha, size=100, workers=4, dm=0)
    print 'vocab size: ', len(model.vocab)
    print len(model.index2word)
    print len(model.docvecs), 'docs'

    for i in xrange(20):
        print 'epoch', i, model.alpha
        model.train(documents)
        # model.alpha *= 0.9

    # sims = model.docvecs.most_similar(0)
    # print sims
    return model


def train_20ng_doc_vectors():
    docs_file = 'e:dc/20ng_bydate/doc_text_data.txt'
    # docs_file_name = 'e:dc/20ng_data/all_docs_wi.txt'
    dst_vecs_file = 'e:dc/20ng_bydate/vecs/dbow_doc_vecs.bin'
    model = train_doc_vectors(docs_file)
    save_doc2vec_vectors(model, dst_vecs_file)


def train_pv_nyt():
    docs_file = 'e:/dc/nyt-world-full/processed/docs_tokenized_lc.txt'
    dst_vecs_file = 'e:/dc/nyt-world-full/processed/vecs/pvdbow-vecs.bin'
    model = train_doc_vectors(docs_file)
    save_doc2vec_vectors(model, dst_vecs_file)

if __name__ == '__main__':
    train_20ng_doc_vectors()
    # train_pv_nyt()

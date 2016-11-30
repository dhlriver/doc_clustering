import gzip
import os
import numpy as np
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import scipy.spatial
import timeit
from eltune import ELTune


def load_el_data(dataset):
    print 'loading', dataset
    fin = open(dataset, 'rb')
    num_docs, vec_dim = np.fromfile(fin, np.int32, 2)
    doc_vecs = np.empty((num_docs, vec_dim), dtype=theano.config.floatX)
    vec_cnds = list()
    commonness = list()
    gold_indices = list()
    for i in xrange(num_docs):
        doc_vec = np.fromfile(fin, np.float32, vec_dim)
        # if i < 5:
        #     print doc_vec
        doc_vecs[i][:] = doc_vec

        num_mentions = np.fromfile(fin, np.int32, 1)
        vecs_mentions = list()
        commonness_mentions = list()
        doc_gold_indices = np.zeros(num_mentions, np.int32)
        for j in xrange(num_mentions):
            num_candidates, doc_gold_indices[j] = np.fromfile(fin, np.int32, 2)

            cur_commonness = np.fromfile(fin, np.float32, num_candidates)
            cur_commonness = cur_commonness.astype(theano.config.floatX)
            commonness_mentions.append(cur_commonness)

            vecs = np.empty((num_candidates, vec_dim), dtype=theano.config.floatX)
            for k in xrange(num_candidates):
                vecs[k][:] = np.fromfile(fin, np.float32, vec_dim)
                # if i < 2 and j < 2 and k < 5:
                #     print vecs[k]
            vecs_mentions.append(vecs)

        commonness.append(commonness_mentions)
        vec_cnds.append(vecs_mentions)
        gold_indices.append(doc_gold_indices)

    fin.close()
    print 'done'

    return doc_vecs, gold_indices, commonness, vec_cnds


def load_el_data_for_theano_train(datasets, num_crpt_vecs=5):
    def load_data_sets(dataset):
        doc_vecs, gold_indices, commonness, vec_cnds = load_el_data(dataset)
        mention_vecs = list()
        gold_vecs = list()
        crpt_vecs = list()
        gold_cmns = list()
        crpt_cmns = list()
        for i, indices in enumerate(gold_indices):  # docs
            doc_vec = doc_vecs[i]
            for j, idx in enumerate(indices):  # mentions
                if len(vec_cnds[i][j]) <= 1:
                    continue
                gold_vec = vec_cnds[i][j][idx]
                for k in xrange(min(num_crpt_vecs, len(vec_cnds[i][j]))):
                    if k != idx:
                        mention_vecs.append(doc_vec)
                        gold_vecs.append(gold_vec)
                        crpt_vecs.append(vec_cnds[i][j][k])
                        gold_cmns.append(commonness[i][j][idx])
                        crpt_cmns.append(commonness[i][j][k])
        return mention_vecs, gold_vecs, crpt_vecs, gold_cmns, crpt_cmns

    all_mentions_vecs = list()
    all_gold_vecs = list()
    all_crpt_vecs = list()
    all_gold_cmns = list()
    all_crpt_cmns = list()
    for ds in datasets:
        tmp_mention_vecs, tmp_gold_vecs, tmp_crpt_vecs, tmp_gold_cmns, tmp_crpt_cmns = load_data_sets(ds)
        all_mentions_vecs += tmp_mention_vecs
        all_gold_vecs += tmp_gold_vecs
        all_crpt_vecs += tmp_crpt_vecs
        all_gold_cmns += tmp_gold_cmns
        all_crpt_cmns += tmp_crpt_cmns

    all_mentions_vecs = theano.shared(np.asarray(all_mentions_vecs, dtype=theano.config.floatX), borrow=True)
    all_gold_vecs = theano.shared(np.asarray(all_gold_vecs, dtype=theano.config.floatX), borrow=True)
    all_crpt_vecs = theano.shared(np.asarray(all_crpt_vecs, dtype=theano.config.floatX), borrow=True)
    all_gold_cmns = theano.shared(np.asarray(all_gold_cmns, dtype=theano.config.floatX), borrow=True)
    all_crpt_cmns = theano.shared(np.asarray(all_crpt_cmns, dtype=theano.config.floatX), borrow=True)
    return all_mentions_vecs, all_gold_vecs, all_crpt_vecs, all_gold_cmns, all_crpt_cmns


def load_el_data_for_theano_test(dataset, max_num_candidates=40):
    doc_vecs, gold_indices, commonness, vec_cnds = load_el_data(dataset)
    num_mentions = 0
    for indices in gold_indices:
        num_mentions += len(indices)

    dim = len(doc_vecs[0])
    target_mentions_vecs = np.zeros((num_mentions, dim), dtype=theano.config.floatX)
    target_commonness = np.zeros((num_mentions, max_num_candidates), dtype=theano.config.floatX)
    target_cnd_vecs = np.ones((num_mentions, max_num_candidates, dim), dtype=theano.config.floatX)
    target_indices = np.zeros(num_mentions, dtype=np.int32)
    num_candidates = np.zeros(num_mentions, dtype=np.int32)
    cur_mention_idx = 0
    for i, indices in enumerate(gold_indices):  # docs
        for j in xrange(len(indices)):  # mentions
            target_indices[cur_mention_idx] = indices[j]
            target_mentions_vecs[cur_mention_idx] = doc_vecs[i]

            num_candidates[cur_mention_idx] = len(vec_cnds[i][j])

            if num_candidates[cur_mention_idx] > 30:
                num_candidates[cur_mention_idx] = 30

            cur_num_candidates = num_candidates[cur_mention_idx]

            cmn_sum = sum(commonness[i][j][:cur_num_candidates])
            for k in xrange(cur_num_candidates):
                commonness[i][j][k] /= cmn_sum

            # target_commonness[cur_mention_idx][:len(commonness[i][j])] = commonness[i][j]
            target_commonness[cur_mention_idx][:cur_num_candidates] = commonness[i][j][:cur_num_candidates]

            # if num_candidates[cur_mention_idx] > 1:
            #     cnt = 0
            #     for k in xrange(1, len(commonness[i][j])):
            #         if commonness[i][j][0] / commonness[i][j][k] > 10:
            #             break
            #         cnt += 1
            #     num_candidates[cur_mention_idx] = cnt

            for k in xrange(len(vec_cnds[i][j])):
                target_cnd_vecs[cur_mention_idx][k][:] = vec_cnds[i][j][k]
            # for l in xrange(len(vec_cnds[i][j]), max_num_candidates):
            #     target_cnd_vecs[cur_mention_idx][l][:] = np.ones(dim, dtype=theano.config.floatX)
            cur_mention_idx += 1

    target_indices = theano.shared(np.asarray(target_indices, dtype=np.int32), borrow=True)
    target_mentions_vecs = theano.shared(np.asarray(target_mentions_vecs, dtype=theano.config.floatX), borrow=True)
    target_commonness = theano.shared(np.asarray(target_commonness, dtype=theano.config.floatX), borrow=True)
    target_cnd_vecs = theano.shared(np.asarray(target_cnd_vecs, dtype=theano.config.floatX), borrow=True)

    return target_indices, target_mentions_vecs, target_commonness, target_cnd_vecs, num_candidates


def load_data(dataset):
    print '... loading data'
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def train_el():
    year = 2009
    method = 3
    num_mentions = 1020.0 if year == 2010 else 1675.0
    if year == 2009:
        datasets_train = [# 'e:/dc/el/dwe_train/tac_2010_train_%d_wl.bin' % method,
                          'e:/data/emadr/el/tac/2010/eval/eval_%d_wl.bin' % method]
        data_set_test = 'e:/data/emadr/el/tac/2009/eval/eval_%d_wl.bin' % method
    elif year == 2010:
        # datasets_train = ['e:/dc/el/dwe_train/tac_2010_train_%d_wl.bin' % method,
        #                   'e:/dc/el/dwe_train/tac_2009_eval_%d_wl.bin' % method]
        # datasets_train = ['e:/dc/el/dwe_train/tac_2010_train_%d_wl.bin' % method]
        # datasets_train = ['e:/dc/el/dwe_train/tac_2009_eval_%d_wl.bin' % method,
        # 'e:/dc/el/dwe_train/tac_2014_eval_wl.bin']]
        datasets_train = ['e:/data/emadr/el/tac/2009/eval/eval_%d_wl.bin' % method]
        data_set_test = 'e:/data/emadr/el/tac/2010/eval/eval_%d_wl.bin' % method
        # datasets_train = ['e:/data/emadr/el/datasetbin/2009/eval_%d_wl.bin' % method]
        # data_set_test = 'e:/data/emadr/el/datasetbin/2010/eval_%d_wl.bin' % method
    else:
        datasets_train = ['e:/data/emadr/el/datasetbin/tac_2009_eval_%d_wl.bin' % method]
        data_set_test = 'e:/data/emadr/el/datasetbin/tac_2014_eval_wl.bin'

    # data_set_train = 'e:/dc/el/dwe_train/tac_2014_train_wl.bin'
    # data_set_test = 'e:/dc/el/dwe_train/tac_2014_eval_wl.bin'
    # datasets_train = ['e:/dc/el/dwe_train/tac_2010_train_wl.bin',
    #                   'e:/dc/el/dwe_train/tac_2014_train_wl.bin',
    #                   'e:/dc/el/dwe_train/tac_2010_eval_wl.bin']
    # datasets_train = ['e:/dc/el/dwe_train/tac_2010_train_wl.bin']
    # datasets_train = ['e:/dc/el/dwe_train/tac_2014_train_wl.bin',
    #                   'e:/dc/el/dwe_train/tac_2009_eval_2_wl.bin']
    # data_set_test = 'e:/dc/el/dwe_train/tac_2010_eval_wl.bin'

    # data_set_test = 'e:/dc/el/dwe_train/tac_2014_eval_wl.bin'

    # doc_vecs, gold_indices, vec_cnds = load_el_data(data_set_file)
    # fh_cnt = 0
    # for indices in gold_indices:
    #     fh_cnt += np.equal(indices, np.zeros(len(indices), np.int32)).sum()
    # print fh_cnt

    max_num_candidates = 50
    test_indices, test_mentions_vecs, test_commonness, test_cnd_vecs, num_candidates = load_el_data_for_theano_test(
            data_set_test, max_num_candidates)
    mask_matrix = ELTune.get_mask_matrix(num_candidates, max_num_candidates)
    # print num_hit.eval()
    train_mention_vecs, train_gold_vecs, train_crpt_vecs, train_gold_cmns, train_crpt_cmns = \
        load_el_data_for_theano_train(datasets_train)

    print 'building models ...'

    dim = 100
    ndim = 200
    rng = np.random.RandomState(1234)
    batch_size = 5
    l2_reg = 0.001
    learning_rate = 0.001

    n_train_batches = train_mention_vecs.get_value(borrow=True).shape[0] // batch_size
    print n_train_batches, 'train batches'

    elt = ELTune(rng, dim, ndim, batch_size)

    batch_index = T.lscalar()
    mention_vecs = T.matrix('mv')
    gold_vecs = T.matrix('gv')
    crpt_vecs = T.matrix('cv')
    gold_cmns = T.fvector('gc')
    crpt_cmns = T.fvector('cc')
    loss = elt.loss(mention_vecs, gold_vecs, crpt_vecs, gold_cmns, crpt_cmns, l2_reg)

    gparams = [T.grad(loss, param) for param in elt.params]
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(elt.params, gparams)]

    train_model = theano.function(
        inputs=[batch_index],
        outputs=loss,
        updates=updates,
        givens={
            mention_vecs: train_mention_vecs[batch_index * batch_size: (batch_index + 1) * batch_size],
            gold_vecs: train_gold_vecs[batch_index * batch_size: (batch_index + 1) * batch_size],
            crpt_vecs: train_crpt_vecs[batch_index * batch_size: (batch_index + 1) * batch_size],
            # gold_cmns: train_gold_cmns[batch_index * batch_size: (batch_index + 1) * batch_size],
            # crpt_cmns: train_crpt_cmns[batch_index * batch_size: (batch_index + 1) * batch_size]
        }
    )

    num_hit = elt.num_hits(test_mentions_vecs, test_cnd_vecs, test_commonness, test_indices, mask_matrix)
    test_model = theano.function([], num_hit)

    print 'done'

    epoch = 0
    n_epochs = 70
    max_correct = 0
    while epoch < n_epochs:
        epoch += 1
        sum_cost = 0
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            sum_cost += minibatch_avg_cost
        cur_correct = test_model()
        print epoch, cur_correct, cur_correct / num_mentions
        if cur_correct > max_correct:
            max_correct = cur_correct
        print sum_cost
    print 'max', max_correct

if __name__ == '__main__':
    train_el()

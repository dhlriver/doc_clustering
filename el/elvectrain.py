import gzip
import os
import numpy as np
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import scipy.spatial
import ioutils
from eltune import ELTune
from itertools import izip


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

    target_indices_var = theano.shared(np.asarray(target_indices, dtype=np.int32), borrow=True)
    target_mentions_vecs = theano.shared(np.asarray(target_mentions_vecs, dtype=theano.config.floatX), borrow=True)
    target_commonness = theano.shared(np.asarray(target_commonness, dtype=theano.config.floatX), borrow=True)
    target_cnd_vecs = theano.shared(np.asarray(target_cnd_vecs, dtype=theano.config.floatX), borrow=True)

    return target_indices_var, target_mentions_vecs, target_commonness, target_cnd_vecs, num_candidates, target_indices


def __read_mention_data(fin, vec_dim):
    qid = ioutils.read_str_with_byte_len(fin)
    gold_id = ioutils.read_str_with_byte_len(fin)
    num_candidates = np.fromfile(fin, np.int32, 1)
    eids = [ioutils.read_str_with_byte_len(fin) for _ in xrange(num_candidates)]
    commonnesses = np.fromfile(fin, np.float32, num_candidates)
    vecs = [np.fromfile(fin, np.float32, vec_dim) for _ in xrange(num_candidates)]
    return qid, gold_id, eids, commonnesses, vecs


def __load_dataset(data_file):
    f = open(data_file, 'rb')
    num_docs, vec_dim = np.fromfile(f, np.int32, 2)
    print num_docs, vec_dim
    docs = list()
    for i in xrange(num_docs):
        doc_vec = np.fromfile(f, np.float32, vec_dim)
        num_mentions = np.fromfile(f, np.int32, 1)
        mentions = list()
        for j in xrange(num_mentions):
            qid, gold_id, eids, commonnesses, vecs = __read_mention_data(f, vec_dim)
            print qid
            print gold_id, eids
            mentions.append((qid, gold_id, eids, commonnesses, vecs))
        docs.append((doc_vec, mentions))
    f.close()
    return docs


def __load_gold_el(edl_file):
    f = open(edl_file, 'r')
    gold_el_result = dict()
    for line in f:
        vals = line.strip().split('\t')
        if len(vals) < 7:
            continue
        gold_el_result[vals[1]] = vals[4]
    f.close()
    return gold_el_result


def __load_x_mention(fin, vecdim):
    qid = ioutils.read_str_with_byte_len(fin)
    num_candidates = np.fromfile(fin, '>i4', 1)
    eids = [ioutils.read_str_with_byte_len(fin) for _ in xrange(num_candidates)]
    commonnesses = np.fromfile(fin, '>f4', num_candidates)
    vecs = [np.fromfile(fin, '>f4', vecdim) for _ in xrange(num_candidates)]
    return qid, eids, commonnesses, vecs


def __load_docs_info(xdatafile):
    f = open(xdatafile, 'rb')
    num_docs, vecdim = np.fromfile(f, '>i4', 2)
    print '%d documents, vec dimention: %d' % (num_docs, vecdim)
    docs = list()
    for i in xrange(num_docs):
        docid = ioutils.read_str_with_byte_len(f)
        # print docid
        docvec = np.fromfile(f, '>f4', vecdim)
        num_mentions = np.fromfile(f, '>i4', 1)
        # print num_mentions
        mentions = list()
        for j in xrange(num_mentions):
            qid, kbids, commonnesses, vecs = __load_x_mention(f, vecdim)
            # print qid, kbids
            # print commonnesses
            # print kbids[2]
            mentions.append((qid, kbids, commonnesses, vecs))
        docs.append((docid, docvec, mentions))
        # if i == 5:
        #     break
    f.close()
    return docs, vecdim


def __get_legal_kbids(kbids, keep_nil):
    if keep_nil:
        return range(len(kbids)), kbids

    indices, legal_kbids = list(), list()
    for i, kbid in enumerate(kbids):
        if kbid.startswith('E'):
            indices.append(i)
            legal_kbids.append(kbid)
    return indices, legal_kbids


# separate those mentions that need ranking (with more than one candidates) and those that do not need ranking
def __separate_mentions(docs_info, keep_nil):
    link_result_trivial = dict()  # for those with no candidates or only one candidate
    qids = list()
    kbids_list = list()
    mention_side_vecs = list()
    candidate_side_vecs_list = list()
    commonnesses_list = list()
    for doc in docs_info:
        docid, docvec, mentions = doc
        for mention in mentions:
            (qid, kbids, commonnesses, vecs) = mention

            indices, legal_kbids = __get_legal_kbids(kbids, keep_nil)

            if len(legal_kbids) == 0:
                link_result_trivial[qid] = 'NIL'
                continue

            if len(legal_kbids) == 1:
                link_result_trivial[qid] = legal_kbids[0]
                continue

            legal_vecs, legal_commonnesses = list(), list()
            for idx in indices:
                legal_vecs.append(vecs[idx])
                legal_commonnesses.append(commonnesses[idx])

            qids.append(qid)
            mention_side_vecs.append(docvec)
            kbids_list.append(legal_kbids)
            candidate_side_vecs_list.append(legal_vecs)
            commonnesses_list.append(legal_commonnesses)

            # else:
            #     qids.append(qid)
            #     mention_side_vecs.append(docvec)
            #     kbids_list.append(kbids)
            #     candidate_side_vecs_list.append(vecs)
            #     commonnesses_list.append(commonnesses)
    candidates_info = (kbids_list, candidate_side_vecs_list, commonnesses_list)
    return link_result_trivial, qids, mention_side_vecs, candidates_info


def __build_x_for_testing(xdatafile, max_num_candidates, keep_nil):
    docs_info, dim = __load_docs_info(xdatafile)
    link_result_trivial, qids, mention_side_vecs, candidates_info = __separate_mentions(docs_info, keep_nil)
    # print qids

    kbids_list, candidate_side_vecs_list, commonnesses_list = candidates_info
    num_mentions = len(qids)

    for i, kbids in enumerate(kbids_list):
        kbids_list[i] = kbids[:min(len(kbids), max_num_candidates)]
    # candidate_side_vecs_list = candidate_side_vecs_list[:min(len(candidate_side_vecs_list), max_num_candidates)]
    # commonnesses_list = commonnesses_list[:min(len(commonnesses_list), max_num_candidates)]

    np_mentions_vecs = np.zeros((num_mentions, dim), dtype=theano.config.floatX)
    np_commonnesses = np.zeros((num_mentions, max_num_candidates), dtype=theano.config.floatX)
    np_cnd_vecs = np.ones((num_mentions, max_num_candidates, dim), dtype=theano.config.floatX)
    # np_num_candidates = np.zeros(num_mentions, dtype=np.int32)
    # print len(mention_side_vecs), len(candidate_side_vecs_list), len(commonnesses_list)
    # print commonnesses_list[0]
    for i, (mention_vec, candidate_vecs, commonnesses) in enumerate(izip(mention_side_vecs,
                                                                         candidate_side_vecs_list, commonnesses_list)):
        np_mentions_vecs[i] = mention_vec
        cur_num_candidates = min(len(commonnesses), max_num_candidates)
        np_commonnesses[i][:cur_num_candidates] = commonnesses[:cur_num_candidates]
        for j in xrange(min(len(candidate_vecs), max_num_candidates)):
            np_cnd_vecs[i][j][:] = candidate_vecs[j]

    mentions_vecs_t = theano.shared(np.asarray(np_mentions_vecs, dtype=theano.config.floatX), borrow=True)
    commonnesses_t = theano.shared(np.asarray(np_commonnesses, dtype=theano.config.floatX), borrow=True)
    cnd_vecs_t = theano.shared(np.asarray(np_cnd_vecs, dtype=theano.config.floatX), borrow=True)
    theano_vars = (mentions_vecs_t, commonnesses_t, cnd_vecs_t)

    return link_result_trivial, qids, kbids_list, theano_vars


def __prepare_training_data(xdatafile, gold_edl_file, num_crpt_vecs):
    gold_el_result = __load_gold_el(gold_edl_file)

    docs_info, vecdim = __load_docs_info(xdatafile)
    link_result_trivial, qids, mention_side_vecs, candidates_info = __separate_mentions(docs_info, True)
    kbids_list, candidate_side_vecs_list, commonnesses_list = candidates_info

    sel_mention_vecs, sel_gold_vecs, sel_crpt_vecs, sel_gold_cmns, sel_crpt_cmns = [], [], [], [], []
    for qid, kbids, cand_vecs, commonnesses in izip(qids, kbids_list, candidate_side_vecs_list, commonnesses_list):
        gold_kbid = gold_el_result[qid]
        try:
            gold_idx = kbids.index(gold_kbid)
            num_candidates = len(kbids)
            for i in xrange(min(num_crpt_vecs, num_candidates)):
                if kbids[i] == gold_kbid:
                    continue
                sel_mention_vecs.append(mention_side_vecs[i])
                sel_gold_vecs.append(cand_vecs[gold_idx])
                sel_gold_cmns.append(commonnesses[gold_idx])
                sel_crpt_vecs.append(cand_vecs[i])
                sel_crpt_cmns.append(commonnesses[i])
        except ValueError:
            pass
    return sel_mention_vecs, sel_gold_vecs, sel_crpt_vecs, sel_gold_cmns, sel_crpt_cmns


def __build_training_data_for_theano(xdatafile, gold_edl_file, num_crpt_vecs):
    r = __prepare_training_data(xdatafile, gold_edl_file, num_crpt_vecs)
    sel_mention_vecs, sel_gold_vecs, sel_crpt_vecs, sel_gold_cmns, sel_crpt_cmns = r

    mention_vecs_var = theano.shared(np.asarray(sel_mention_vecs, dtype=theano.config.floatX), borrow=True)
    gold_vecs_var = theano.shared(np.asarray(sel_gold_vecs, dtype=theano.config.floatX), borrow=True)
    crpt_vecs_var = theano.shared(np.asarray(sel_crpt_vecs, dtype=theano.config.floatX), borrow=True)
    gold_cmns_var = theano.shared(np.asarray(sel_gold_cmns, dtype=theano.config.floatX), borrow=True)
    crpt_cmns_var = theano.shared(np.asarray(sel_crpt_cmns, dtype=theano.config.floatX), borrow=True)
    return mention_vecs_var, gold_vecs_var, crpt_vecs_var, gold_cmns_var, crpt_cmns_var


def __build_train_model(xdatafile, gold_edl_file, num_crpt_vecs, elt, batch_size,
                        l2_reg, learning_rate):
    r = __build_training_data_for_theano(xdatafile, gold_edl_file, num_crpt_vecs)
    train_mention_vecs, train_gold_vecs, train_crpt_vecs, train_gold_cmns, train_crpt_cmns = r

    n_train_batches = train_mention_vecs.get_value(borrow=True).shape[0] // batch_size
    print n_train_batches, 'train batches'

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

    return train_model, n_train_batches


def __build_test_model(xdatafile, max_num_candidates, elt, keep_nil):
    link_result_trivial, qids, kbids_list, cand_theano_vars = __build_x_for_testing(xdatafile,
                                                                                    max_num_candidates, keep_nil)
    mentions_vecs, commonnesses, cnd_vecs = cand_theano_vars

    nums_candidates = [len(kbids) for kbids in kbids_list]
    mask_matrix = ELTune.get_mask_matrix(nums_candidates, max_num_candidates)

    y_pred, sims = elt.y_pred(mentions_vecs, cnd_vecs, commonnesses, mask_matrix)
    model_y_pred = theano.function([], y_pred)
    model_sims = theano.function([], sims)
    return model_y_pred, link_result_trivial, qids, kbids_list, model_sims


# def __num_hits(y_pred, y_true):
#     cnt = 0
#     for y0, y1 in izip(y_pred, y_true):
#         if y0 == y1:
#             cnt += 1
#     return cnt

def __kbid_hit(sys_kbid, gold_kbid, use_eid):
    if use_eid and sys_kbid.startswith('m.') and gold_kbid.startswith('NIL'):
        return True
    if gold_kbid.startswith('NIL') and sys_kbid.startswith('NIL'):
        return True
    if sys_kbid == gold_kbid:
        return True
    return False


def __nerl_perf(result_triv, qids, kbids_list, y_pred, gold_el_result, keep_nil, use_eid=True):
    # print result_triv
    # print gold_el_result
    result_rank = dict()
    for qid, kbids, y in izip(qids, kbids_list, y_pred):
        if y >= len(kbids):
            print y, len(kbids)
        result_rank[qid] = kbids[y]
        # result_rank[qid] = kbids[0]

    triv_hit_cnt, hit_cnt, num_mentions = 0, 0, 0
    for qid, gold_kbid in gold_el_result.iteritems():
        if not keep_nil and gold_kbid.startswith('NIL'):
            continue
        num_mentions += 1

        sys_kbid = result_triv.get(qid, '')
        if sys_kbid:
            if __kbid_hit(sys_kbid, gold_kbid, use_eid):
                triv_hit_cnt += 1
                hit_cnt += 1
            continue
        sys_kbid = result_rank.get(qid, '')
        if not sys_kbid:
            print 'error! %s not found' % qid
        if __kbid_hit(sys_kbid, gold_kbid, use_eid):
            hit_cnt += 1

    print '%d, %d of %d; triv: %f' % (triv_hit_cnt, hit_cnt, num_mentions, float(triv_hit_cnt) / num_mentions)
    return float(hit_cnt) / num_mentions


def __train_el_new():
    year = 2010
    method = 3
    train_data_file, train_gold_file = '', ''
    val_data_file, test_data_file = '', ''
    val_gold_file, test_gold_file = '', ''
    if year == 2011:
        train_data_file = 'e:/data/emadr/el/tac/2009/eval/el-2009-eval-3.bin'
        train_gold_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/mentions.tab'
        val_data_file = 'e:/data/emadr/el/tac/2010/eval/el-2010-eval-3.bin'
        val_gold_file = 'e:/data/el/LDC2015E19/data/2010/eval/data/mentions.tab'
        test_data_file = 'e:/data/emadr/el/tac/2011/eval/el-2011-eval-3.bin'
        test_gold_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/mentions.tab'
    elif year == 2010:
        train_data_file = 'e:/data/emadr/el/tac/2011/eval/el-2011-eval-3.bin'
        train_gold_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/mentions.tab'
        val_data_file = 'e:/data/emadr/el/tac/2009/eval/el-2009-eval-3.bin'
        val_gold_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/mentions.tab'
        test_data_file = 'e:/data/emadr/el/tac/2010/eval/el-2010-eval-3.bin'
        test_gold_file = 'e:/data/el/LDC2015E19/data/2010/eval/data/mentions.tab'
    elif year == 2009:
        train_data_file = 'e:/data/emadr/el/tac/2010/eval/el-2010-eval-3.bin'
        train_gold_file = 'e:/data/el/LDC2015E19/data/2010/eval/data/mentions.tab'
        val_data_file = 'e:/data/emadr/el/tac/2011/eval/el-2011-eval-3.bin'
        val_gold_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/mentions.tab'
        test_data_file = 'e:/data/emadr/el/tac/2009/eval/el-2009-eval-3.bin'
        test_gold_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/mentions.tab'

    print 'building models ...'

    num_crpt_vecs = 5
    max_num_candidates = 30
    keep_nil = False

    dim = 100
    ndim = 200
    rng = np.random.RandomState(1234)
    batch_size = 5
    l2_reg = 0.001
    learning_rate = 0.001

    elt = ELTune(rng, dim, ndim, batch_size)
    train_model, n_train_batches = __build_train_model(train_data_file, train_gold_file, num_crpt_vecs, elt,
                                                       batch_size, l2_reg, learning_rate)

    val_gold_result = __load_gold_el(val_gold_file)
    test_gold_result = __load_gold_el(test_gold_file)
    # val_model_y_pred, val_link_result_triv, val_qids, val_kbids_list = __build_test_model(val_data_file,
    #                                                                                       max_num_candidates,
    #                                                                                       elt, keep_nil)
    # test_model_y_pred, test_link_result_triv, test_qids, test_kbids_list = __build_test_model(test_data_file,
    #                                                                                           max_num_candidates,
    #                                                                                           elt, keep_nil)
    val_model_y_pred, val_link_result_triv, val_qids, val_kbids_list, msims = __build_test_model(val_data_file,
                                                                                                 max_num_candidates,
                                                                                                 elt, keep_nil)
    test_model_y_pred, test_link_result_triv, test_qids, test_kbids_list, msims = __build_test_model(test_data_file,
                                                                                                     max_num_candidates,
                                                                                                     elt, keep_nil)

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

        cur_val_y_pred = val_model_y_pred()
        val_perf = __nerl_perf(val_link_result_triv, val_qids, val_kbids_list, cur_val_y_pred,
                               val_gold_result, keep_nil)
        # cur_sims = msims()
        # print cur_sims

        cur_test_y_pred = test_model_y_pred()
        for yp in cur_test_y_pred[:100]:
            print yp,
        print
        test_perf = __nerl_perf(test_link_result_triv, test_qids, test_kbids_list, cur_test_y_pred,
                                test_gold_result, keep_nil)

        print 'epoch: %d, val: %f, test: %f' % (epoch, val_perf, test_perf)
        # print epoch, cur_correct, cur_correct / num_mentions
        # if cur_correct > max_correct:
        #     max_correct = cur_correct
        print sum_cost
    print 'max', max_correct


def __train_el():
    year = 2010
    method = 3

    datasets_train = []
    dataset_val, dataset_test = '', ''
    num_mentions_val, num_mentions_test = 1.0, 1.0
    if year == 2009:
        datasets_train = [# 'e:/dc/el/dwe_train/tac_2010_train_%d_wl.bin' % method,
                          'e:/data/emadr/el/tac/2010/eval/eval_%d_wl.bin' % method]
        dataset_val = 'e:/data/emadr/el/tac/2011/eval/eval_%d_wl.bin' % method
        dataset_test = 'e:/data/emadr/el/tac/2009/eval/eval_%d_wl.bin' % method
        num_mentions_val = 2250.0
        num_mentions_test = 1675.0
    elif year == 2010:
        datasets_train = ['e:/data/emadr/el/tac/2009/eval/eval_%d_wl.bin' % method]
        dataset_val = 'e:/data/emadr/el/tac/2011/eval/eval_%d_wl.bin' % method
        dataset_test = 'e:/data/emadr/el/tac/2010/eval/eval_%d_wl.bin' % method
        num_mentions_val = 2250.0
        num_mentions_test = 1020.0
    elif year == 2011:
        datasets_train = ['e:/data/emadr/el/tac/2009/eval/eval_%d_wl.bin' % method]
        dataset_val = 'e:/data/emadr/el/tac/2010/eval/eval_%d_wl.bin' % method
        dataset_test = 'e:/data/emadr/el/tac/2011/eval/eval_%d_wl.bin' % method
        num_mentions_val = 1020.0
        num_mentions_test = 2250.0
    # else:
    #     datasets_train = ['e:/data/emadr/el/datasetbin/tac_2009_eval_%d_wl.bin' % method]
    #     dataset_val = 'e:/data/emadr/el/tac/2010/eval/eval_%d_wl.bin' % method
    #     dataset_test = 'e:/data/emadr/el/datasetbin/tac_2014_eval_wl.bin'

    max_num_candidates = 50
    val_indices_var, val_mentions_vecs, val_commonness, val_cnd_vecs, val_num_candidates, val_y = \
        load_el_data_for_theano_test(dataset_val, max_num_candidates)
    test_indices_var, test_mentions_vecs, test_commonness, test_cnd_vecs, test_num_candidates, test_y = \
        load_el_data_for_theano_test(dataset_test, max_num_candidates)

    mask_matrix_test = ELTune.get_mask_matrix(test_num_candidates, max_num_candidates)
    mask_matrix_val = ELTune.get_mask_matrix(val_num_candidates, max_num_candidates)
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

    val_y_pred = elt.y_pred(val_mentions_vecs, val_cnd_vecs, val_commonness, mask_matrix_val)
    val_model_y_pred = theano.function([], val_y_pred)

    test_y_pred = elt.y_pred(test_mentions_vecs, test_cnd_vecs, test_commonness, mask_matrix_test)
    test_model_y_pred = theano.function([], test_y_pred)

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

        cur_val_y_pred = val_model_y_pred()
        cur_correct_val = __num_hits(cur_val_y_pred, val_y)

        cur_y_pred = test_model_y_pred()
        cur_correct_test = __num_hits(cur_y_pred, test_y)

        print 'epoch: %d, val: %d %f, test: %d %f' % (epoch, cur_correct_val, cur_correct_val / num_mentions_val,
                                                      cur_correct_test, cur_correct_test / num_mentions_test)
        # print epoch, cur_correct, cur_correct / num_mentions
        # if cur_correct > max_correct:
        #     max_correct = cur_correct
        print sum_cost
    print 'max', max_correct

if __name__ == '__main__':
    __train_el_new()
    # __train_el()
    # __load_dataset('e:/data/emadr/el/tac/bindata/2010-eval-m3-wl.bin')
    # __build_x_for_testing('e:/data/emadr/el/tac/2011/eval/el-2011-eval-3.bin')
    # __build_data_for_training('e:/data/emadr/el/tac/2011/eval/el-2011-eval-3.bin',
    #                           'e:/data/el/LDC2015E19/data/2011/eval/data/mentions.tab', 5)

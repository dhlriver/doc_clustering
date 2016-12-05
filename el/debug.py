import gzip
import os
import numpy as np
import theano
import theano.tensor as T
import neunet
import timeit
import re
from elutils import load_gold_el, load_docs_info, load_eid_wid_file
from mention import Mention


def filter_errors():
    fin = open('e:/el/error_list_09.txt', 'rb')
    err_values = list()
    for line in fin:
        vals = line.strip().split('\t')
        ch = vals[0][-1]
        if not ch.isdigit():
            continue
        if vals[1].startswith('NIL'):
            continue
        err_values.append(vals)
    fin.close()

    err_values.sort(key=lambda cur_val: cur_val[1])
    fout = open('e:/el/error_list_09_filtered_0.txt', 'wb')
    for err_val in err_values:
        fout.write('%s\t%s\t%s\t%s\n' % (err_val[0], err_val[1], err_val[2], err_val[3]))
    fout.close()


def show_errors():
    def load_queries(file_name):
        fin = open(file_name, 'rb')
        text = fin.read()
        fin.close()
        ps = r'<query id="(.*?)">\s*<name>(.*?)</name>\s*<docid>(.*?)</docid>'
        miter = re.finditer(ps, text)
        qdict = dict()
        for m in miter:
            qdict[m.group(1)] = (m.group(2), m.group(3))
        return qdict

    def load_file(file_name):
        fin = open(file_name, 'rb')
        tmp_err_list = list()
        for line in fin:
            vals = line.strip().split('\t')
            tmp_err_list.append(vals)
        fin.close()
        return tmp_err_list

    err_list = load_file('e:/el/error_list_09_filtered_0.txt')
    query_dict = load_queries(r'D:\data\el\LDC2015E19\data\2009\eval\tac_kbp_2'
                              r'009_english_entity_linking_evaluation_queries.xml')
    for err_val in err_list:
        name = query_dict[err_val[0]]
        print err_val[0], err_val[1], err_val[2], name[0], name[1]


def __get_legal_kbids(kbids, keep_nil):
    if keep_nil:
        return range(len(kbids)), kbids

    indices, legal_kbids = list(), list()
    for i, kbid in enumerate(kbids):
        if kbid.startswith('E'):
            indices.append(i)
            legal_kbids.append(kbid)
    return indices, legal_kbids


def __el_stat():
    # data_file = 'e:/data/emadr/el/tac/2009/eval/el-2009-eval-3.bin'
    # gold_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/mentions.tab'
    data_file = 'e:/data/emadr/el/tac/2011/eval/el-2011-eval-3.bin'
    gold_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/mentions.tab'
    eid_wid_file = 'e:/data/el/res/eid_wid_ord_eid.txt'
    keep_nil = False
    only_show_not_in_candidate = False

    eid_wid_dict = load_eid_wid_file(eid_wid_file)

    # gold_el_result = load_gold_el(gold_file)
    mentions = Mention.load_edl_file(gold_file)
    qid_mention_dict = Mention.group_mentions_by_qid(mentions)
    docs_info, dim = load_docs_info(data_file)

    error_list = list()
    num_mentions = 0
    nil_hit_cnt, id_hit_cnt = 0, 0
    for doc in docs_info:
        docid, docvec, mentions = doc
        for mention in mentions:
            (qid, kbids, commonnesses, vecs) = mention

            gold_mention = qid_mention_dict[qid]
            gold_id = gold_mention.kbid
            gold_id_is_nil = gold_id.startswith('NIL')
            if not keep_nil and gold_id_is_nil:
                continue
            num_mentions += 1

            indices, legal_kbids = __get_legal_kbids(kbids, keep_nil)

            if gold_id_is_nil and (len(legal_kbids) == 0 or legal_kbids[0].startswith('m.')):
                nil_hit_cnt += 1
                continue

            first_kbid = legal_kbids[0] if legal_kbids else 'NIL'

            if first_kbid == gold_id:
                id_hit_cnt += 1
                continue

            error_list.append((qid, gold_mention.name, gold_id, legal_kbids))

    error_list.sort(key=lambda x: x[1])
    for e in error_list:
        qid, name, gold_id, legal_kbids = e
        gold_wid = eid_wid_dict.get(gold_id, -1)
        in_candidates = gold_id in legal_kbids

        if only_show_not_in_candidate and in_candidates:
            continue

        # if not in_candidates:
        #     print 'not found'
        print '%s\t%s\t%s_%d' % (qid, name, gold_id, gold_wid)

        # for eid in legal_kbids:
        #     wid = eid_wid_dict.get(eid, -1)
        #     print '\t%s_%d' % (eid, wid),
        # print

    print 'INKB: %f' % (float(id_hit_cnt) / num_mentions)
    print 'TOTAL: %f' % (float(id_hit_cnt + nil_hit_cnt) / num_mentions)


def __test():
    f = open('e:/data/emadr/el/tac/2011/eval/doc_vecs_3.bin', 'rb')
    num, dim = np.fromfile(f, '<i4', 2)
    for i in xrange(5):
        vec = np.fromfile(f, '<f4', dim)
        print vec
    f.close()

if __name__ == '__main__':
    # filter_errors()
    # show_errors()
    # __test()
    __el_stat()

import gzip
import os
import numpy as np
import theano
import theano.tensor as T
import neunet
import timeit
import re
from elutils import load_gold_el, load_docs_info, load_eid_wid_file, load_mid_eid_file
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


def __measure_perf():
    # sys_edl_file = 'e:/data/el/LDC2015E19/data/2011/eval/output/sys-link-sm-0.tab'
    # gold_edl_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/mentions-name-expansion.tab'
    sys_edl_file = 'e:/data/el/LDC2015E19/data/2010/eval/output/sys-link-sm-0.tab'
    gold_edl_file = 'e:/data/el/LDC2015E19/data/2010/eval/data/mentions.tab'
    # sys_edl_file = 'e:/data/el/LDC2015E19/data/2009/eval/output/sys-link-sm-0.tab'
    # gold_edl_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/mentions.tab'
    mid_eid_file = 'e:/data/edl/res/prog-gen/mid-to-eid.bin'
    # mid_eid_file = 'e:/data/edl/res/prog-gen/mid-to-eid-ac.bin'
    mid_eid_dict = load_mid_eid_file(mid_eid_file)
    sys_el_result = load_gold_el(sys_edl_file)
    # filter_nil = False
    filter_nil = True

    f = open(gold_edl_file, 'r')
    gold_el_result = dict()
    for line in f:
        vals = line.strip().split('\t')
        if len(vals) < 7:
            continue
        gold_el_result[vals[1]] = vals
    f.close()

    hit_cnt, mention_cnt = 0, 0
    for qid, kbid in sys_el_result.iteritems():
        eid = mid_eid_dict.get(kbid[2:], 'NIL')
        gold_result = gold_el_result[qid]
        gold_id = gold_result[4]
        if filter_nil and gold_id.startswith('NIL'):
            continue

        mention_cnt += 1

        if not filter_nil and eid.startswith('NIL') and gold_id.startswith('NIL'):
            hit_cnt += 1
            continue
        if eid == gold_id:
            hit_cnt += 1
            continue
        print eid, gold_result
        # print eid, gold_id
    print float(hit_cnt) / mention_cnt


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
    data_file = 'e:/data/emadr/el/tac/2011/eval/el-2011-eval-name-exp-3.bin'
    gold_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/mentions-name-expansion.tab'
    eid_wid_file = 'e:/data/el/res/eid_wid_ord_eid.txt'
    keep_nil = True
    only_show_not_in_candidate = False

    eid_wid_dict = load_eid_wid_file(eid_wid_file)

    # gold_el_result = load_gold_el(gold_file)
    mentions = Mention.load_edl_file(gold_file)
    qid_mention_dict = Mention.group_mentions_by_qid(mentions)
    docs_info, dim = load_docs_info(data_file)

    error_list = list()
    num_mentions, nil_mentions = 0, 0
    nil_hit_cnt, id_hit_cnt = 0, 0
    for doc in docs_info:
        docid, docvec, mentions = doc
        for mention in mentions:
            (qid, kbids, commonnesses, vecs) = mention

            gold_mention = qid_mention_dict[qid]
            gold_id = gold_mention.kbid
            gold_id_is_nil = gold_id.startswith('NIL')
            if gold_id_is_nil:
                nil_mentions += 1
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

            error_list.append((qid, docid, gold_mention.name, gold_id, legal_kbids))

    error_list.sort(key=lambda x: x[2])
    for e in error_list:
        qid, docid, name, gold_id, legal_kbids = e
        gold_wid = eid_wid_dict.get(gold_id, -1)
        in_candidates = gold_id in legal_kbids

        if only_show_not_in_candidate and in_candidates:
            continue

        # if not in_candidates:
        #     print 'not found'
        print '%s\t%s\t%s\t%s_%d' % (qid, docid, name, gold_id, gold_wid)

        # for eid in legal_kbids:
        #     wid = eid_wid_dict.get(eid, -1)
        #     print '\t%s_%d' % (eid, wid),
        # print

    print id_hit_cnt, num_mentions
    print 'INKB: %f' % (float(id_hit_cnt) / (num_mentions - nil_mentions))
    print 'TOTAL: %f' % (float(id_hit_cnt + nil_hit_cnt) / num_mentions)


def __test():
    def load_exp_file(filename):
        exp_dict = dict()
        f = open(filename, 'r')
        for line in f:
            vals = line.strip().split('\t')
            exp_dict[vals[0]] = vals[1]
        f.close()
        return exp_dict

    exp_dict0 = load_exp_file('e:/data/el/exp0.txt')
    exp_dict1 = load_exp_file('e:/data/el/exp1.txt')
    cnt = 0
    for n, en in exp_dict0.iteritems():
        if n not in exp_dict1:
            if ' ' not in n and not n.isupper() and en.count(' ') == 1:
                print '%s\t%s' % (n, en)
                cnt += 1
    print cnt

if __name__ == '__main__':
    # filter_errors()
    # show_errors()
    __measure_perf()
    # __test()
    # __el_stat()

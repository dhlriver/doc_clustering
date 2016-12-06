import ioutils
import numpy as np


def load_mid_eid_file(filename):
    f = open(filename, 'rb')
    num = np.fromfile(f, '>i4', 1)
    print num
    mid_eid_dict = dict()
    for i in xrange(num):
        mid = ioutils.read_str_with_fixed_len(f, 8)
        eid = ioutils.read_str_with_fixed_len(f, 8)
        # print mid, eid
        mid_eid_dict[mid] = eid
    f.close()
    return mid_eid_dict


def __load_mention_info(fin, vecdim):
    qid = ioutils.read_str_with_byte_len(fin)
    num_candidates = np.fromfile(fin, '>i4', 1)
    eids = [ioutils.read_str_with_byte_len(fin) for _ in xrange(num_candidates)]
    commonnesses = np.fromfile(fin, '>f4', num_candidates)
    vecs = [np.fromfile(fin, '>f4', vecdim) for _ in xrange(num_candidates)]
    return qid, eids, commonnesses, vecs


def load_docs_info(xdatafile):
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
            qid, kbids, commonnesses, vecs = __load_mention_info(f, vecdim)
            # print qid, kbids
            # print commonnesses
            # print kbids[2]
            mentions.append((qid, kbids, commonnesses, vecs))
        docs.append((docid, docvec, mentions))
        # if i == 5:
        #     break
    f.close()
    return docs, vecdim


def load_gold_el(edl_file, include_name=False):
    f = open(edl_file, 'r')
    gold_el_result = dict()
    for line in f:
        vals = line.strip().split('\t')
        if len(vals) < 7:
            continue
        if include_name:
            gold_el_result[vals[1]] = (vals[4], vals[2])
        else:
            gold_el_result[vals[1]] = vals[4]
    f.close()
    return gold_el_result


def load_eid_wid_file(eid_wid_file):
    eid_wid_dict = dict()
    f = open(eid_wid_file, 'r')
    for line in f:
        vals = line.strip().split('\t')
        eid_wid_dict[vals[0]] = int(vals[1])
    f.close()

    return eid_wid_dict

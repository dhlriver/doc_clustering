import numpy as np
from itertools import izip
import random
from sklearn import svm

from mention import Mention
import ioutils


def __assgin_different_id_to_all_nils(mentions):
    nil_idx = 1
    for m in mentions:
        if not m.kbid.startswith('NIL'):
            continue
        m.kbid = 'NIL%05d' % nil_idx
        nil_idx += 1


def __belong_to_group(mention, mention_group):
    for m in mention_group:
        if m.name == mention.name:
            return True
    return False


def __print_mention_group(mention_group):
    for m in mention_group:
        print '%s\t%s' % (m.name, m.docid)


def __get_nil_mentions(mentions):
    nil_mentions = list()
    for m in mentions:
        if m.kbid.startswith('NIL'):
            nil_mentions.append(m)
    return nil_mentions


def __gen_positive_samples(kbid_mentions):
    pos_groups = list()
    for kbid, mentions in kbid_mentions.iteritems():
        docid_mentions = dict()
        for m in mentions:
            mentions_same_doc = docid_mentions.get(m.docid, list())
            if not mentions_same_doc:
                docid_mentions[m.docid] = mentions_same_doc
            mentions_same_doc.append(m)
        if len(docid_mentions) < 2:
            continue
        # print kbid
        cur_sample = list()
        for docid, doc_mentions in docid_mentions.iteritems():
            cur_sample.append(doc_mentions)
        pos_groups.append(cur_sample)

    pos_samples = list()
    for pg in pos_groups:
        l = len(pg)
        for i in xrange(l):
            for j in xrange(i + 1, l):
                pos_samples.append((pg[i], pg[j]))

    return pos_samples


def __gen_neg_samples(kbid_mentions, num_neg_samples):
    mention_groups_dict = dict()
    for kbid, mentions in kbid_mentions.iteritems():
        for m in mentions:
            tmpid = kbid + '_' + m.docid
            # print tmpid
            cur_group = mention_groups_dict.get(tmpid, list())
            if not cur_group:
                mention_groups_dict[tmpid] = cur_group
            cur_group.append(m)

    neg_samples = list()
    mention_groups = mention_groups_dict.values()
    for i in xrange(num_neg_samples):
        mgx = random.randint(0, len(mention_groups) - 1)
        mgy = mgx
        while mgy == mgx:
            mgy = random.randint(0, len(mention_groups) - 1)
        neg_samples.append((mention_groups[mgx], mention_groups[mgy]))

    return neg_samples


def __merge_samples(pos_samples, neg_samples):
    all_samples = list()
    for sample in pos_samples:
        all_samples.append((sample, 1))
    for sample in neg_samples:
        all_samples.append((sample, 0))
    random.shuffle(all_samples)
    return all_samples


def __get_longest_name(mentions):
    max_len = 0
    name = ''
    for m in mentions:
        if len(m.name) > max_len:
            max_len = m.name
            name = m.name
    return name


def __get_features(sample):
    features = list()

    mga = sample[0]
    mgb = sample[1]
    longest_name_a = __get_longest_name(mga)
    longest_name_b = __get_longest_name(mgb)
    f = 1 if longest_name_a == longest_name_b else 0
    features.append(f)
    # print longest_name_a
    # print longest_name_b
    # print features
    return features


def __gen_training_data(edl_file):
    mentions = Mention.load_edl_file(edl_file)
    nil_mentions = __get_nil_mentions(mentions)
    kbid_mentions = Mention.group_mentions_by_kbid(nil_mentions)
    pos_samples = __gen_positive_samples(kbid_mentions)
    neg_samples = __gen_neg_samples(kbid_mentions, len(pos_samples))

    data_x = list()
    data_y = list()
    all_samples = __merge_samples(pos_samples, neg_samples)
    for sample, y, in all_samples:
        sample_x = __get_features(sample)
        data_x.append(sample_x)
        data_y.append(y)
        # print

    for x, y in izip(data_x, data_y):
        print x, y

    return data_x, data_y


def all_to_all(edl_file, dst_edl_file):
    mentions = Mention.load_edl_file(edl_file)
    __assgin_different_id_to_all_nils(mentions)
    Mention.save_as_edl_file(mentions, dst_edl_file)


def __read_mention_from_linking_info_file(fin):
    qid = ioutils.read_str_with_byte_len(fin)
    num_candidates = np.fromfile(fin, '>i4', 1)
    # print num_candidates
    for i in xrange(num_candidates):
        np.fromfile(fin, 'b', 8)
        np.fromfile(fin, '>f4', 1)
        np.fromfile(fin, '>f8', 1)
        np.fromfile(fin, '>f4', 1)
    return qid


def __apply_coref(edl_file, linking_info_file, dst_edl_file):
    coref_dict = dict()
    f = open(linking_info_file, 'rb')
    while True:
        docid = ioutils.read_str_with_byte_len(f)
        if not docid:
            break
        num_mentions = np.fromfile(f, '>i4', 1)
        is_nested = np.fromfile(f, 'b', num_mentions)
        corefs = np.fromfile(f, '>i4', num_mentions)
        qids = list()
        for i in xrange(num_mentions):
            qid = __read_mention_from_linking_info_file(f)
            qids.append(qid)
        for coref_id, qid in izip(corefs, qids):
            if coref_id > 0:
                coref_dict[qid] = qids[coref_id]
    f.close()

    mentions = Mention.load_edl_file(edl_file)
    qid_mentions = Mention.group_mentions_by_qid(mentions)
    __assgin_different_id_to_all_nils(mentions)
    print qid_mentions['EDL14_ENG_0052'].kbid
    for m in mentions:
        if not m.kbid.startswith('NIL'):
            continue
        coref_qid = coref_dict.get(m.mention_id, '')
        if coref_qid:
            print m.mention_id, coref_qid, m.name, qid_mentions[coref_qid].kbid
            m.kbid = qid_mentions[coref_qid].kbid

    Mention.save_as_edl_file(mentions, dst_edl_file)


def job_all_to_all():
    edl_file = 'e:/data/el/LDC2015E20/data/eval/output/emadr-result.tab'
    dst_edl_file = 'e:/data/el/LDC2015E20/data/eval/output/emadr-result-ata.tab'
    all_to_all(edl_file, dst_edl_file)


def __job_coref():
    edl_file = 'e:/data/el/LDC2015E20/data/eval/output/emadr-result.tab'
    linking_info_file = 'e:/data/el/LDC2015E20/data/eval/output/cmn-tfidf-sys-raw.bin'
    dst_edl_file = 'e:/data/el/LDC2015E20/data/eval/output/emadr-result-coref.tab'
    __apply_coref(edl_file, linking_info_file, dst_edl_file)


def __job_gen_training_data():
    # edl_file = 'e:/data/el/LDC2015E20/data/eval/output/emadr-result-coref.tab'
    edl_file = 'd:/data/el/LDC2015E20/data/eval/data/mentions-raw.tab'
    __gen_training_data(edl_file)


def __coref_train():
    edl_file_train = 'd:/data/el/LDC2015E20/data/eval/data/mentions-raw.tab'
    edl_file_test = 'd:/data/el/LDC2015E20/data/eval/data/mentions-raw.tab'

    train_x, train_y = __gen_training_data(edl_file_train)
    test_x, test_y = __gen_training_data(edl_file_test)

    clf = svm.SVC(kernel='sigmoid')
    print 'train ...'
    clf.fit(train_x, train_y)

    # test_x = train_x
    # test_y = train_y
    pred_y = clf.predict(test_x)
    cnt = 0
    for y1, y2 in izip(pred_y, test_y):
        if y1 == y2:
            cnt += 1
    print cnt, len(test_y), float(cnt) / len(test_y)


if __name__ == '__main__':
    # job_all_to_all()
    # __job_coref()
    # __job_gen_training_data()
    __coref_train()

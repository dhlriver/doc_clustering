import numpy as np
from itertools import izip
from scipy.spatial.distance import cosine
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from elutils import load_docs_info
from mention import Mention


def __load_nil_detection_x_train(data_file):
    docs_info, dim = load_docs_info(data_file)
    qid_x_list = list()
    for doc in docs_info:
        docid, docvec, mentions = doc
        for m in mentions:
            qid, kbids, commonnesses, vecs = m
            # print qid
            # print kbids
            # print commonnesses
            for i, (kbid, commonness, vec) in enumerate(izip(kbids, commonnesses, vecs)):
                if not kbid.startswith('E'):
                    continue
                first_candidate = 1 if i == 0 else -1
                # first_candidate = i
                qid_x_list.append((qid, kbid, first_candidate, commonness, cosine(docvec, vec)))
                break
                # print '\t', cosine(docvec, vec)
            # qid_x_list.append((qid, ))
    return qid_x_list


def __build_training_data(qid_x_list, edl_file):
    mentions = Mention.load_edl_file(edl_file)
    qid_mentions = Mention.group_mentions_by_qid(mentions)
    train_x = list()
    train_y = list()
    for tup in qid_x_list:
        qid, kbid, first_candidate, commonness, dist = tup
        # print qid, kbid, first_candidate, commonness, dist
        m = qid_mentions[qid]

        if (not m.kbid.startswith('NIL')) and m.kbid != kbid:
            continue

        y = 0 if m.kbid.startswith('NIL') else 1
        # train_x.append([first_candidate, commonness, dist])
        train_x.append([first_candidate, commonness])
        # train_x.append([first_candidate])
        train_y.append(y)
    return train_x, train_y


def __nil_detection():
    train_data_file = 'e:/data/emadr/el/tac/2009/eval/el-2009-eval-expansion-nloc-3.bin'
    train_edl_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/mentions-expansion-nloc.tab'
    test_data_file = 'e:/data/emadr/el/tac/2010/eval/el-2010-eval-expansion-nloc-3.bin'
    test_edl_file = 'e:/data/el/LDC2015E19/data/2010/eval/data/mentions-expansion-nloc.tab'
    qid_x_list = __load_nil_detection_x_train(train_data_file)
    train_x, train_y = __build_training_data(qid_x_list, train_edl_file)
    text_qid_x_list = __load_nil_detection_x_train(test_data_file)
    test_x, test_y = __build_training_data(text_qid_x_list, test_edl_file)
    # clf = svm.LinearSVC()
    clf = svm.SVC(kernel='sigmoid')
    # clf = LogisticRegression(C=1000)
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
    __nil_detection()

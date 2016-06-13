import numpy as np
import sklearn.cluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from munkres import Munkres
from itertools import izip

import ioutils


def __get_label_positions(labels):
    label_pos_dict = dict()
    for i, label in enumerate(labels):
        pos_list = label_pos_dict.get(label, list())
        if not pos_list:
            label_pos_dict[label] = pos_list
        pos_list.append(i)
    return label_pos_dict


def __num_match_pos(gold_pos_list, sys_pos_list):
    len0, len1 = len(gold_pos_list), len(sys_pos_list)
    pos = 0
    result = 0
    for i in xrange(len0):
        while pos < len1 and sys_pos_list[pos] < gold_pos_list[i]:
            pos += 1
        if pos < len1 and sys_pos_list[pos] == gold_pos_list[i]:
            result += 1
    return result


def __init_cost_matrix(gold_label_pos_dict, sys_label_pos_dict):
    cost_matrix = [[0] * len(gold_label_pos_dict) for _ in xrange(len(sys_label_pos_dict))]
    for i, (gold_label, gold_pos) in enumerate(gold_label_pos_dict.iteritems()):
        for j, (sys_label, sys_pos) in enumerate(sys_label_pos_dict.iteritems()):
            num_matches = __num_match_pos(gold_pos, sys_pos)
            cost_matrix[j][i] = -num_matches

    return cost_matrix


def purity(gold_labels, sys_labels):
    sys_label_set, gold_label_set = set(), set()
    for l in sys_labels:
        sys_label_set.add(l)
    for l in gold_labels:
        gold_label_set.add(l)

    sum_hit = 0
    for k in sys_label_set:
        hit_cnt_dict = dict()
        for lg, ls in izip(gold_labels, sys_labels):
            if k == ls:
                cnt = hit_cnt_dict.get(lg, 0)
                hit_cnt_dict[lg] = cnt + 1
        max_hit_num = max(hit_cnt_dict.values())
        # print k, max_hit_num
        sum_hit += max_hit_num
    return float(sum_hit) / len(gold_labels)


def __label_indices_dict(labels):
    labels_indices = dict()
    for i, label in enumerate(labels):
        indices_list = labels_indices.get(label, None)
        if not indices_list:
            indices_list = labels_indices[label] = list()
        indices_list.append(i)
    return labels_indices


def __count_true_positive_pairs(gold_label_indices, sys_labels):
    tp, fn = 0, 0
    for _, indices in gold_label_indices.iteritems():
        num_indices = len(indices)
        for i in xrange(num_indices):
            for j in xrange(i + 1, num_indices):
                if sys_labels[indices[i]] == sys_labels[indices[j]]:
                    tp += 1
                else:
                    fn += 1
    return tp, fn


def __count_false_positive_pairs(sys_label_indices, gold_labels):
    fp = 0
    for _, indices in sys_label_indices.iteritems():
        num_indices = len(indices)
        for i in xrange(num_indices):
            for j in xrange(i + 1, num_indices):
                if gold_labels[indices[i]] != gold_labels[indices[j]]:
                    fp += 1
    return fp


def rand_index(gold_labels, sys_labels):

    gold_label_indices = __label_indices_dict(gold_labels)
    sys_label_indices = __label_indices_dict(sys_labels)
    tp, fn = __count_true_positive_pairs(gold_label_indices, sys_labels)
    # print tp, fn
    fp = __count_false_positive_pairs(sys_label_indices, gold_labels)

    num_docs = len(gold_labels)
    total_num_pairs = num_docs * (num_docs - 1) / 2
    tn = total_num_pairs - tp - fn - fp

    # print float(tp + tn) / total_num_pairs

    # tp, tn, fp, fn = 0, 0, 0, 0
    # for i in xrange(num_docs):
    #     if (i + 1) % 1000 == 0:
    #         print i + 1
    #     for j in xrange(i + 1, num_docs):
    #         if gold_labels[i] == gold_labels[j] and sys_labels[i] == sys_labels[j]:
    #             tp += 1
    #         if gold_labels[i] != gold_labels[j] and sys_labels[i] != sys_labels[j]:
    #             tn += 1
    #         if gold_labels[i] != gold_labels[j] and sys_labels[i] == sys_labels[j]:
    #             fp += 1
    #         if gold_labels[i] == gold_labels[j] and sys_labels[i] != sys_labels[j]:
    #             fn += 1

    # print tp, tn, fp, fn
    # print float(tp + tn) / total_num_pairs

    return float(tp + tn) / total_num_pairs


def cluster_accuracy(gold_labels, sys_labels):
    gold_label_pos = __get_label_positions(gold_labels)
    sys_label_pos = __get_label_positions(sys_labels)

    cost_matrix = __init_cost_matrix(gold_label_pos, sys_label_pos)
    # print cost_matrix
    m = Munkres()
    indices = m.compute(cost_matrix)
    num_correct = 0
    for r, c, in indices:
        num_correct -= cost_matrix[r][c]
        # print r, c, -cost_matrix[r][c]
    # print num_correct
    # print float(num_correct) / len(gold_labels)
    return float(num_correct) / len(gold_labels)


def write_labels(labels, dst_file_name):
    fout = open(dst_file_name, 'wb')
    cnt = 0
    for label in labels:
        fout.write(str(cnt) + '\t' + str(label) + '\n')
        cnt += 1
    fout.close()


def cluster_and_eval(vec_list, labels, num_clusters):
    if len(labels) < len(vec_list):
        vec_list = vec_list[-len(labels):]

    cl_data = np.asarray(vec_list)
    # print cl_data

    # model = sklearn.cluster.AgglomerativeClustering(n_clusters=5,
    #                                                 linkage="average", affinity="cosine")
    model = sklearn.cluster.KMeans(n_clusters=num_clusters, n_jobs=4, n_init=50)
    model.fit(cl_data)
    # print estimator.labels_
    # print labels[0:100]
    # print model.labels_

    nmi_score = normalized_mutual_info_score(labels, model.labels_)
    purity_score = purity(labels, model.labels_)
    ri_score = rand_index(labels, model.labels_)

    # print len(labels), 'samples'
    print 'NMI: %f Purity: %f Rand index: %f' % (nmi_score, purity_score, ri_score)
    # print 'Accuracy: %f' % cluster_accuracy(labels, model.labels_)

    return nmi_score, purity_score, ri_score


def clustering(doc_vec_file_name, labels_file_name, num_clusters):
    fin = open(doc_vec_file_name, 'rb')
    num_vecs, vec_len = np.fromfile(fin, dtype=np.int32, count=2)
    print '%d vecs, dim: %d' % (num_vecs, vec_len)
    vec_list = list()
    for i in xrange(num_vecs):
        vec = np.fromfile(fin, np.float32, vec_len)
        # vec = numpy.random.uniform(0, 1, vec_len).astype(numpy.float32)
        # vec = vec[100:]
        # vec = vec[:50]
        # vec /= numpy.linalg.norm(vec)
        vec_list.append(vec)
    fin.close()

    fin = open(labels_file_name, 'rb')
    x = np.fromfile(fin, dtype=np.int32, count=1)
    # gold_labels = numpy.fromfile(fin, dtype=numpy.int8, count=x[0])
    gold_labels = np.fromfile(fin, dtype=np.int32, count=x[0])
    fin.close()

    return cluster_and_eval(vec_list, gold_labels, num_clusters)


def write_clustering_perf_to_csv(method, perf_list, dst_file):
    fout = open(dst_file, 'wb')
    fout.write('K,NMI.%s,Purity.%s,Rand_Index.%s\n' % (method, method, method))
    for perf in perf_list:
        fout.write('%d,%f,%f,%f\n' % (perf[0], perf[1], perf[2], perf[3]))
    fout.close()


def cluster_nyt():
    num_clusters_list = [5, 10, 15, 20]
    method = 'RSM'

    result_file = 'd:/documents/lab/paper-data/plot/%s-results-ri.csv' % method.lower()

    labels_file_name = 'e:/dc/nyt-world-full/processed/test/doc-labels.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/de-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/de-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/glove-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw2-vecs-ner.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw2-vecs-ner-200.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw4-vecs-015.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw5-vecs-ner.bin'
    doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/rsm-vecs-20.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/drbm-vecs-30.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/pvdm-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/pvdbow-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/nvdm-nyt.bin'

    # doc_vec_file_name = 'e:/dc/20ng_bydate/vecs/test-dedw-vecs.bin'
    # labels_file_name = 'e:/dc/20ng_bydate/test_labels.bin'

    perf_list = list()
    # for num_clusters in [5, 10, 15, 20]:
    vec_list = ioutils.load_vec_list_file(doc_vec_file_name)
    labels = ioutils.load_labels_file(labels_file_name)
    for num_clusters in num_clusters_list:
        print '%d clusters' % num_clusters
        # nmi_score, purity_score, ri_score = clustering(doc_vec_file_name, labels_file_name, num_clusters)
        nmi_score, purity_score, ri_score = cluster_and_eval(vec_list, labels, num_clusters)
        perf_list.append((num_clusters, nmi_score, purity_score, ri_score))
        # break
    write_clustering_perf_to_csv(method, perf_list, result_file)


def main():
    cluster_nyt()
    # gold_labels = np.random.randint(0, 5, 10000)
    # gold_labels = [0, 0, 0, 1, 1, 1]
    # sys_labels = np.random.randint(0, 5, 10000)
    # print rand_index(gold_labels, sys_labels)

if __name__ == '__main__':
    main()

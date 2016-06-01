import numpy
import sklearn.cluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from munkres import Munkres
from itertools import izip


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

    cl_data = numpy.asarray(vec_list)
    # print cl_data

    # model = sklearn.cluster.AgglomerativeClustering(n_clusters=5,
    #                                                 linkage="average", affinity="cosine")
    model = sklearn.cluster.KMeans(n_clusters=num_clusters, n_jobs=4, n_init=50)
    model.fit(cl_data)
    # print estimator.labels_
    # print labels[0:100]
    # print model.labels_

    print len(labels), 'samples'
    print 'NMI: %f' % normalized_mutual_info_score(labels, model.labels_)
    print 'Purity: %f' % purity(labels, model.labels_)
    print 'Accuracy: %f' % cluster_accuracy(labels, model.labels_)


def clustering(doc_vec_file_name, labels_file_name, num_clusters):
    fin = open(doc_vec_file_name, 'rb')
    num_vecs, vec_len = numpy.fromfile(fin, dtype=numpy.int32, count=2)
    print '%d vecs, dim: %d' % (num_vecs, vec_len)
    vec_list = list()
    for i in xrange(num_vecs):
        vec = numpy.fromfile(fin, numpy.float32, vec_len)
        # vec = numpy.random.uniform(0, 1, vec_len).astype(numpy.float32)
        # vec = vec[100:]
        # vec = vec[:50]
        # vec /= numpy.linalg.norm(vec)
        vec_list.append(vec)
    fin.close()

    fin = open(labels_file_name, 'rb')
    x = numpy.fromfile(fin, dtype=numpy.int32, count=1)
    # gold_labels = numpy.fromfile(fin, dtype=numpy.int8, count=x[0])
    gold_labels = numpy.fromfile(fin, dtype=numpy.int32, count=x[0])
    fin.close()

    cluster_and_eval(vec_list, gold_labels, num_clusters)


def cluster_nyt():
    num_clusters = 5

    labels_file_name = 'e:/dc/nyt-world-full/processed/doc-labels.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/de-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/de-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/glove-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw2-vecs-ner.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw2-vecs-ner-200.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw4-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/dedw5-vecs-ner.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/rsm-vecs-20.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/drbm-vecs-30.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/pvdm-vecs.bin'
    doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/pvdbow-vecs.bin'
    # doc_vec_file_name = 'e:/dc/nyt-world-full/processed/vecs/nvdm-nyt.bin'

    # doc_vec_file_name = 'e:/dc/20ng_bydate/vecs/test-dedw-vecs.bin'
    # labels_file_name = 'e:/dc/20ng_bydate/test_labels.bin'
    for num_clusters in [5, 10, 15, 20]:
        print '%d clusters' % num_clusters
        clustering(doc_vec_file_name, labels_file_name, num_clusters)
        # break


def main():
    cluster_nyt()
    # gold_labels = [0, 0, 0, 1, 1, 1]
    # sys_labels = [0, 1, 2, 0, 0, 1]
    # purity(gold_labels, sys_labels)

if __name__ == '__main__':
    main()

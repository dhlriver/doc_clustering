import numpy
import sklearn.cluster
from sklearn.metrics.cluster import normalized_mutual_info_score


def write_labels(labels, dst_file_name):
    fout = open(dst_file_name, 'wb')
    cnt = 0
    for label in labels:
        fout.write(str(cnt) + '\t' + str(label) + '\n')
        cnt += 1
    fout.close()


def cluster_and_test(vec_list, labels):
    if len(labels) < len(vec_list):
        vec_list = vec_list[-len(labels):]

    cl_data = numpy.asarray(vec_list)
    # print cl_data

    # model = sklearn.cluster.AgglomerativeClustering(n_clusters=5,
    #                                                 linkage="average", affinity="cosine")
    model = sklearn.cluster.KMeans(n_clusters=20, n_jobs=4, n_init=60)
    model.fit(cl_data)
    # print estimator.labels_
    # print labels[0:100]
    # print model.labels_

    print len(labels), 'samples'
    print normalized_mutual_info_score(labels, model.labels_)


def clustering_for_doc_vec_label_file(doc_vec_label_file_name, result_file_name):
    fin = open(doc_vec_label_file_name, 'rb')
    x = numpy.fromfile(fin, numpy.int32, 2)
    print x
    num_labels = x[0]
    vec_len = x[1]
    labels = numpy.fromfile(fin, numpy.int8, num_labels)
    # print labels[-10:-1]
    vec_list = list()
    for i in xrange(num_labels):
        vec = numpy.fromfile(fin, numpy.float32, vec_len)
        vec /= numpy.linalg.norm(vec)
        vec_list.append(vec)
    fin.close()

    # cluster_and_test(vec_list, labels)

    vecs_beg = len(vec_list) - 20000
    nv = 20000
    end_idx = len(vec_list)
    while vecs_beg + nv <= end_idx:
        print vecs_beg, vecs_beg + nv
        cluster_and_test(vec_list[vecs_beg:vecs_beg + nv], labels[vecs_beg:vecs_beg + nv])
        vecs_beg += nv
        # break


def clustering(doc_vec_file_name, labels_file_name):
    fin = open(doc_vec_file_name, 'rb')
    x = numpy.fromfile(fin, dtype=numpy.int32, count=2)
    num_vecs = x[0]
    vec_len = x[1]
    print 'dim', vec_len
    vec_list = list()
    for i in xrange(num_vecs):
        vec = numpy.fromfile(fin, numpy.float32, vec_len)
        # vec = vec[100-64:]
        # vec = vec[64:]
        # vec = vec[:64]
        # if i < 5:
        #     print vec
        vec /= numpy.linalg.norm(vec)
        vec_list.append(vec)
    fin.close()

    fin = open(labels_file_name, 'rb')
    x = numpy.fromfile(fin, dtype=numpy.int32, count=1)
    gold_labels = numpy.fromfile(fin, dtype=numpy.int8, count=x[0])
    fin.close()

    cluster_and_test(vec_list, gold_labels)


def cluster_nyt():
    labels_file_name = 'e:/dc/nyt/labels_f2012.bin'
    doc_vec_file_name = 'e:/dc/nyt/vecs/doc_vec_lo_f2012_joint_128.bin'
    clustering(doc_vec_file_name, labels_file_name)


def cluster_20ng():
    labels_file_name = 'e:/dc/20ng_data/all_doc_labels.bin'
    doc_vec_file_name = 'e:/dc/20ng_data/vecs/doc_vec_joint_128.bin'
    clustering(doc_vec_file_name, labels_file_name)


def cluster_20ng_train():
    labels_file_name = 'e:/dc/20ng_data/split/train_labels.bin'
    doc_vec_file_name = 'e:/dc/20ng_data/vecs/train_doc_vec_joint_100.bin'
    clustering(doc_vec_file_name, labels_file_name)


def cluster_20ng_test():
    labels_file_name = 'e:/dc/20ng_data/split/test_labels.bin'
    doc_vec_file_name = 'e:/dc/20ng_data/vecs/test_doc_vec_joint_100.bin'
    clustering(doc_vec_file_name, labels_file_name)


def main():
    # vec = numpy.random.rand(1, 10)
    # nm = numpy.linalg.norm(vec[0])
    # vec[0] = vec[0] / nm
    # print numpy.inner(vec[0], vec[0])
    # test_clustering()
    # doc_vec_label_file_name = '/media/dhl/Data/dc/nyt/docvecs_dbow_100_label_list_rm.bin'
    # doc_vec_label_file_name = '/media/dhl/Datadc/nyt/doc_vec_300_label_list.bin'
    # doc_vec_label_file_name = 'e:/dc/nyt/es_vec_label_list.bin'
    # doc_vec_label_file_name = 'e:/dc/nyt/merged_vec_label_list.bin'
    # result_file_name = 'e:/dc/nyt/es_cluster_result.txt'
    # result_file_name = '/media/dhl/Data/dc/nyt/mv_cluster_result.txt'
    # clustering_for_doc_vec_label_file(doc_vec_label_file_name, result_file_name)

    # cluster_nyt()
    # cluster_20ng()
    # cluster_20ng_train()
    cluster_20ng_test()


if __name__ == '__main__':
    # la = [1, 2, 3, 4, 5]
    # lb = [1, 1, 1, 1, 2]
    # print normalized_mutual_info_score(la, lb)
    main()

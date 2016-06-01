import numpy as np
import ioutils
import random

label_list = ['africa', 'americas', 'asia', 'europe', 'middleeast']


def get_label_id(label):
    if label == 'null':
        return 0
    idx = 1
    for l in label_list:
        if label == l:
            return idx
        idx += 1
    return -1


def filter_vecs(full_vec_list_file_name, label_list_file_name, dst_file_name):
    fin0 = open(full_vec_list_file_name, 'rb')
    fin1 = open(label_list_file_name, 'rb')

    x = np.fromfile(fin0, '<i4', count=2)
    print x
    vec_len = x[1]
    vec_list = list()
    doc_cnt = 0
    for i in xrange(x[0]):
        vec = np.fromfile(fin0, '<f4', vec_len)
        line = fin1.readline().strip()
        vals = line.split('\t')
        label_id = get_label_id(vals[1])

        if label_id == -1:
            print 'label:', vals[1]
            continue
        if label_id == 0:
            continue

        vec_list.append(vec)
        doc_cnt += 1

    fout = open(dst_file_name, 'wb')
    np.array([len(vec_list), vec_len], np.int32).tofile(fout)
    for vec in vec_list:
        vec.tofile(fout)
    fout.close()


def filter_doc_entity_file(entity_list_file_name, full_doc_entity_file_name, label_list_file_name, dst_text_file_name,
                           dst_bin_file_name):
    fin = open(entity_list_file_name, 'rb')
    line = fin.readline()
    num_entities = int(line.strip())
    print num_entities, 'entities'
    fin.close()

    fin0 = open(full_doc_entity_file_name, 'rb')
    fin1 = open(label_list_file_name, 'rb')

    adj_list = list()
    weights_list = list()
    flg = False
    for line_idx, line0 in enumerate(fin0):
        if (line_idx + 1) % 5000 == 0:
            print line_idx + 1

        line1 = fin1.readline()
        if not line1:
            print 'file1 end earlier.'
            break

        vals0 = line0.strip().split(' ')
        vals1 = line1.strip().split('\t')
        if vals0[0] != vals1[0]:
            print 'doc not equal!'
            break

        if vals0[0].startswith('2012'):
            flg = True

        if not flg:
            continue

        label_id = get_label_id(vals1[1])
        if label_id <= 0:
            continue

        num_vertices = int(vals0[1])
        adj_vertices = np.zeros(num_vertices, dtype=np.int32)
        weights = np.zeros(num_vertices, dtype=np.int32)
        for i in xrange(num_vertices):
            adj_vertices[i] = int(vals0[i * 2 + 2])
            weights[i] = int(vals0[i * 2 + 3])
        adj_list.append(adj_vertices)
        weights_list.append(weights)
        # fout.write(line0)
    fin0.close()
    fin1.close()

    fout = open(dst_text_file_name, 'wb')
    fout.write(str(len(adj_list)) + ' ' + str(num_entities) + '\n')
    for i in xrange(len(adj_list)):
        fout.write(str(len(adj_list[i])))
        for j in xrange(len(adj_list[i])):
            fout.write(' ' + str(adj_list[i][j]) + ' ' + str(weights_list[i][j]))
        fout.write('\n')
    fout.close()

    fout = open(dst_bin_file_name, 'wb')
    np.array([len(adj_list), num_entities], dtype=np.int32).tofile(fout)
    for i in xrange(len(adj_list)):
        np.array([len(adj_list[i])], dtype=np.int32).tofile(fout)
        adj_list[i].tofile(fout)
        weights_list[i].tofile(fout)
    fout.close()


def split_vectors(all_vec_file_name, all_labels_file_name, dst_train_vec_file_name,
                  dst_train_labels_file_name, dst_test_vec_file_name, dst_test_labels_file_name):
    all_vec_list = ioutils.load_vec_list_file(all_vec_file_name)
    all_labels = ioutils.load_labels_file(all_labels_file_name)

    train_vec_list = list()
    train_labels = list()
    test_vec_list = list()
    test_labels = list()
    for vec, label in zip(all_vec_list, all_labels):
        rand_val = random.randint(1, 10)
        if rand_val == 1:
            test_vec_list.append(vec)
            test_labels.append(label)
        else:
            train_vec_list.append(vec)
            train_labels.append(label)

    print len(train_labels), 'training samples'
    print len(test_labels), 'testing samples'

    def save_vecs(vec_list, dst_file_name):
        fout = open(dst_file_name, 'wb')
        np.asarray([len(vec_list), len(vec_list[0])], np.int32).tofile(fout)
        for cur_vec in vec_list:
            cur_vec.tofile(fout)
        fout.close()

    def save_labels(labels_list, dst_file_name):
        fout = open(dst_file_name, 'wb')
        np.asarray([len(labels_list)], np.int32).tofile(fout)
        np.asarray(labels_list, np.int32).tofile(fout)
        fout.close()

    save_vecs(train_vec_list, dst_train_vec_file_name)
    save_labels(train_labels, dst_train_labels_file_name)
    save_vecs(test_vec_list, dst_test_vec_file_name)
    save_labels(test_labels, dst_test_labels_file_name)


def averaging_vecs(doc_word_file, word_vecs_file, dst_file):
    word_vecs = ioutils.load_vec_list_file(word_vecs_file)
    dim = len(word_vecs[0])

    fin = open(doc_word_file, 'rb')
    num_left, num_right = np.fromfile(fin, np.int32, 2)
    print num_left, num_right
    fout = open(dst_file, 'wb')
    np.asarray([num_left, dim], np.int32).tofile(fout)
    for i in xrange(num_left):
        num_vertices = np.fromfile(fin, np.int32, 1)

        indices = np.fromfile(fin, np.int32, num_vertices)
        weights = np.fromfile(fin, np.uint16, num_vertices)

        vec = np.zeros(dim, np.float32)
        sum_weights = np.sum(weights)
        for idx, w in zip(indices, weights):
            vec += word_vecs[idx] * w
        vec /= sum_weights
        vec.tofile(fout)

        if i % 10000 == 10000 - 1:
            print i + 1
        # break  # TODO
    fout.close()
    fin.close()


def do_filter_vecs():
    vec_list_file_name = 'e:/dc/nyt/vecs/es_doc_vec_64.bin'
    label_list_file_name = 'e:/dc/nyt/doc_label_list.txt'
    dst_file_name = 'e:/dc/nyt/vecs/es_doc_vec_64_lo.bin'
    filter_vecs(vec_list_file_name, label_list_file_name, dst_file_name)


def do_filter_doc_entity_file():
    entity_list_file_name = 'e:/dc/nyt/entity_name_id_list.txt'
    doc_entity_file_name = 'e:/dc/nyt/doc_entities.txt'
    label_list_file_name = 'e:/dc/nyt/doc_label_list.txt'
    dst_file_name = 'e:/dc/nyt/doc_entities_lo_f2012_tmp.txt'
    dst_bin_file_name = 'e:/dc/nyt/doc_entities_lo_f2012.bin'
    filter_doc_entity_file(entity_list_file_name, doc_entity_file_name, label_list_file_name,
                           dst_file_name, dst_bin_file_name)


def split_20ng_vecs():
    all_vecs_file_name = 'e:/dc/20ng_data/vecs/all_doc_vec_joint_200.bin'
    all_labels_file_name = 'e:/dc/20ng_data/all_doc_labels.bin'
    train_vecs_file_name = 'e:/dc/20ng_data/split/train_doc_vec_joint_200.bin'
    train_labels_file_name = 'e:/dc/20ng_data/split/train_labels.bin'
    test_vecs_file_name = 'e:/dc/20ng_data/split/test_doc_vec_joint_200.bin'
    test_labels_file_name = 'e:/dc/20ng_data/split/test_labels.bin'
    split_vectors(all_vecs_file_name, all_labels_file_name, train_vecs_file_name, train_labels_file_name,
                  test_vecs_file_name, test_labels_file_name)


def gen_20ng_doc_vecs_by_averaging():
    doc_word_file = 'e:/dc/20ng_bydate/all_docs_dw_net_short.bin'
    word_vecs_file = 'e:/dc/20ng_bydate/vecs/word_vecs.bin'
    dst_file = 'e:/dc/20ng_bydate/vecs/doc_vecs_av.bin'
    averaging_vecs(doc_word_file, word_vecs_file, dst_file)


def gen_el_doc_vecs_by_averaging():
    # doc_word_file = 'e:/dc/el/tac_2014_train_docs_bow.bin'
    doc_word_file = 'e:/dc/el/wiki_bow.bin'
    word_vecs_file = 'e:/dc/el/vecs/word_vecs.bin'
    dst_file = 'e:/dc/el/vecs/wiki_vecs_av.bin'
    averaging_vecs(doc_word_file, word_vecs_file, dst_file)


def merge_vecs():
    year = 2010
    part = 'train'
    method = 3

    # vecs_file0 = 'e:/dc/el/vecs/tac_' + file_tag + '_entity_vecs.bin'
    # vecs_file1 = 'e:/dc/el/vecs/tac_' + file_tag + '_dw_vecs.bin'
    # dst_file = 'e:/dc/el/vecs/tac_' + file_tag + '_vecs.bin'

    vecs_file0 = 'e:/dc/el/vecs/%d/%s_%d_de_vecs.bin' % (year, part, method)
    vecs_file1 = 'e:/dc/el/vecs/%d/%s_%d_dw_vecs.bin' % (year, part, method)
    dst_file = 'e:/dc/el/vecs/%d/%s_%d_vecs.bin' % (year, part, method)

    vecs0 = ioutils.load_vec_list_file(vecs_file0)
    vecs1 = ioutils.load_vec_list_file(vecs_file1)
    if len(vecs0) != len(vecs1):
        print 'number of vectors not equal!'
        return

    fout = open(dst_file, 'wb')
    np.asarray([len(vecs0), len(vecs0[0]) + len(vecs1[0])], np.int32).tofile(fout)
    for i in xrange(len(vecs0)):
        vecs0[i].tofile(fout)
        vecs1[i].tofile(fout)
    fout.close()


def main():
    # do_filter_vecs()
    # do_filter_doc_entity_file()
    # split_20ng_vecs()
    # gen_20ng_doc_vecs_by_averaging()
    # gen_el_doc_vecs_by_averaging()
    merge_vecs()


if __name__ == '__main__':
    main()

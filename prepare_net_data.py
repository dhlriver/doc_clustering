import numpy
import os
import random


def split_20ng_data(dir_path, train_path_list_file, train_label_list_file,
                    test_path_list_file, test_label_list_file):
    fout0 = open(train_path_list_file, 'wb')
    fout1 = open(test_path_list_file, 'wb')
    train_label_cnts = list()
    train_doc_cnt = 0
    test_label_cnts = list()
    test_doc_cnt = 0
    for f in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, f)
        if os.path.isdir(cur_path):
            train_cnt = 0
            test_cnt = 0
            for f1 in os.listdir(cur_path):
                cur_path1 = os.path.join(cur_path, f1)
                if os.path.isfile(cur_path1):
                    rnd_val = random.randint(1, 10)
                    if rnd_val == 1:
                        fout1.write(cur_path1 + '\n')
                        test_cnt += 1
                    else:
                        fout0.write(cur_path1 + '\n')
                        train_cnt += 1
            train_doc_cnt += train_cnt
            test_doc_cnt += test_cnt
            train_label_cnts.append(train_cnt)
            test_label_cnts.append(test_cnt)
    fout0.close()
    fout1.close()

    def write_label_file(num_docs, label_cnts, dst_file_name):
        fout = open(dst_file_name, 'wb')
        numpy.array([num_docs], dtype=numpy.int32).tofile(fout)
        for label, label_cnt in enumerate(label_cnts):
            labels = numpy.ones(label_cnt, dtype=numpy.int32) * label
            labels.tofile(fout)
        fout.close()

    write_label_file(train_doc_cnt, train_label_cnts, train_label_list_file)
    write_label_file(test_doc_cnt, test_label_cnts, test_label_list_file)


def load_entity_dict(dict_file_name):
    entity_dict = dict()
    fin = open(dict_file_name, 'rb')
    for idx, line in enumerate(fin):
        vals = line.split('\t')
        entity_dict[vals[0]] = idx
    fin.close()
    return entity_dict


def gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name,
                             dst_file_name):
    entity_dict = load_entity_dict(entity_dict_file_name)
    entity_list_docs = list()
    entity_cnt_list_docs = list()
    fin = open(raw_doc_entity_file_name, 'rb')
    for line in fin:
        vals = line.split('\t')
        # tmp_entity_list = list()
        tmp_doc_entity_dict = dict()
        for val in vals:
            idx = entity_dict.get(val, -1)
            if idx != -1:
                cnt = tmp_doc_entity_dict.get(idx, 0)
                tmp_doc_entity_dict[idx] = cnt + 1
                # tmp_entity_list.append(idx)
        entity_list_docs.append(numpy.array(tmp_doc_entity_dict.keys(), dtype=numpy.int32))
        entity_cnt_list_docs.append(numpy.array(tmp_doc_entity_dict.values(), dtype=numpy.int32))
        # break
    fin.close()

    fout = open(dst_file_name, 'wb')
    numpy.array([len(entity_list_docs)], dtype=numpy.int32).tofile(fout)
    numpy.array([len(entity_dict)], dtype=numpy.int32).tofile(fout)
    for entity_list, entity_cnt_list in zip(entity_list_docs, entity_cnt_list_docs):
        numpy.array([len(entity_list)], dtype=numpy.int32).tofile(fout)
        entity_list.tofile(fout)
        entity_cnt_list.tofile(fout)
        # numpy.ones(len(entity_list), dtype=numpy.int32).tofile(fout)
    fout.close()


def gen_20ng_path_label_file(dir_path, dst_path_list_file, dst_label_file):
    fout0 = open(dst_path_list_file, 'wb')
    label_cnts = list()
    doc_cnt = 0
    for f in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, f)
        if os.path.isdir(cur_path):
            cnt = 0
            for f1 in os.listdir(cur_path):
                cur_path1 = os.path.join(cur_path, f1)
                if os.path.isfile(cur_path1):
                    fout0.write(cur_path1 + '\n')
                    cnt += 1
            doc_cnt += cnt
            label_cnts.append(cnt)
    fout0.close()

    fout1 = open(dst_label_file, 'wb')
    numpy.array([doc_cnt], dtype=numpy.int32).tofile(fout1)
    for label, label_cnt in enumerate(label_cnts):
        labels = numpy.ones(label_cnt, dtype=numpy.int32) * label
        labels.tofile(fout1)
    fout1.close()


def do_gen_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def do_gen_train_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_data/split/train_entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_data/split/train_doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_data/split/train_doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def do_gen_test_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_data/split/train_entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_data/split/test_doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_data/split/test_doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def do_split_20ng_data():
    dir_path = 'e:/dc/20_newsgroups'
    train_path_list_file = 'e:/dc/20ng_data/train_file_list.txt'
    train_label_list_file = 'e:/dc/20ng_data/train_labels.bin'
    test_path_list_file = 'e:/dc/20ng_data/test_file_list.txt'
    test_label_list_file = 'e:/dc/20ng_data/test_labels.bin'
    split_20ng_data(dir_path, train_path_list_file, train_label_list_file,
                    test_path_list_file, test_label_list_file)


def do_gen_20ng_path_label_file():
    dir_path = 'e:/dc/20_newsgroups'
    dst_path_list_file = 'e:/dc/20ng_data/all_doc_path_list.txt'
    dst_label_file = 'e:/dc/20ng_data/all_doc_labels.bin'
    gen_20ng_path_label_file(dir_path, dst_path_list_file, dst_label_file)
    # fin = open(dst_label_file, 'rb')
    # num_docs = numpy.fromfile(fin, dtype=numpy.int32, count=1)
    # print num_docs
    # labels = numpy.fromfile(fin, dtype=numpy.int32, count=num_docs)
    # print labels
    # fin.close()


def main():
    # do_gen_20ng_path_label_file()
    # do_split_20ng_data()
    do_gen_20ng_doc_entity_list()
    # do_gen_train_20ng_doc_entity_list()
    # do_gen_test_20ng_doc_entity_list()


def test():
    fin = open('e:/dc/20ng_data/doc_entities.bin', 'rb')
    head_vals = numpy.fromfile(fin, numpy.int32, 2)
    print head_vals
    num_entities = numpy.fromfile(fin, numpy.int32, 1)
    print num_entities
    entities = numpy.fromfile(fin, numpy.int32, num_entities)
    cnts = numpy.fromfile(fin, numpy.int32, num_entities)
    print entities
    print cnts
    fin.close()

if __name__ == '__main__':
    # test()
    main()

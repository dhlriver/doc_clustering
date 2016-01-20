import os
import numpy as np
import random
import io_utils


def split_vecs(all_vecs_file_name, split_labels_file_name,
               dst_train_vecs_file_name, dst_test_vecs_file_name):
    all_vec_list = io_utils.load_vec_list_file(all_vecs_file_name)
    split_labels = io_utils.load_labels_file(split_labels_file_name)

    train_vec_list = list()
    test_vec_list = list()
    for vec, split_label in zip(all_vec_list, split_labels):
        if split_label == 1:
            test_vec_list.append(vec)
        else:
            train_vec_list.append(vec)

    print len(train_vec_list), 'training samples'
    print len(test_vec_list), 'testing samples'

    def save_vecs(vec_list, dst_file_name):
        fout = open(dst_file_name, 'wb')
        np.asarray([len(vec_list), len(vec_list[0])], np.int32).tofile(fout)
        for cur_vec in vec_list:
            cur_vec.tofile(fout)
        fout.close()

    save_vecs(train_vec_list, dst_train_vecs_file_name)
    save_vecs(test_vec_list, dst_test_vecs_file_name)


def index_20ng_dataset(dir_path, dst_path_list_file, dst_class_labels_file,
                       dst_split_labels_file, dst_train_class_labels_file,
                       dst_test_class_labels_file):
    fout = open(dst_path_list_file, 'wb')
    split_labels = list()
    train_labels = list()
    test_labels = list()
    all_labels = list()
    doc_cnt = 0
    cur_class_label = 0
    for f in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, f)
        if os.path.isdir(cur_path):
            for f1 in os.listdir(cur_path):
                cur_path1 = os.path.join(cur_path, f1)
                if os.path.isfile(cur_path1):
                    fout.write(cur_path1 + '\n')
                    doc_cnt += 1

                    all_labels.append(cur_class_label)
                    rand_val = random.randint(1, 10)
                    if rand_val <= 4:
                        split_labels.append(1)  # test
                        test_labels.append(cur_class_label)
                    else:
                        split_labels.append(0)  # train
                        train_labels.append(cur_class_label)

            cur_class_label += 1
    fout.close()

    def save_labels(labels, dst_file_name):
        fout_labels = open(dst_file_name, 'wb')
        np.asarray([len(labels)], dtype=np.int32).tofile(fout_labels)
        np.asarray(labels, dtype=np.int32).tofile(fout_labels)
        fout_labels.close()

    save_labels(split_labels, dst_split_labels_file)
    save_labels(all_labels, dst_class_labels_file)
    save_labels(train_labels, dst_train_class_labels_file)
    save_labels(test_labels, dst_test_class_labels_file)


def index_20ng_bydate(dir_path, dst_path_list_file, dst_class_labels_file,
                      dst_split_labels_file, dst_train_class_labels_file,
                      dst_test_class_labels_file):
    split_labels = list()
    train_labels = list()
    test_labels = list()
    all_labels = list()

    train_files_path = os.path.join(dir_path, '20news-bydate-train')
    test_files_path = os.path.join(dir_path, '20news-bydate-test')

    fout = open(dst_path_list_file, 'wb')

    def write_paths(tmp_dir_path, split_label):
        cur_class_label = 0
        for f in os.listdir(tmp_dir_path):
            cur_path = os.path.join(tmp_dir_path, f)
            if os.path.isdir(cur_path):
                for f1 in os.listdir(cur_path):
                    cur_path1 = os.path.join(cur_path, f1)
                    if os.path.isfile(cur_path1):
                        fout.write(cur_path1 + '\n')

                        all_labels.append(cur_class_label)
                        split_labels.append(split_label)
                        if split_label == 0:  # test
                            train_labels.append(cur_class_label)
                        else:
                            test_labels.append(cur_class_label)

                cur_class_label += 1

    write_paths(train_files_path, 0)
    write_paths(test_files_path, 1)

    fout.close()

    def save_labels(labels, dst_file_name):
        fout_labels = open(dst_file_name, 'wb')
        np.asarray([len(labels)], dtype=np.int32).tofile(fout_labels)
        np.asarray(labels, dtype=np.int32).tofile(fout_labels)
        fout_labels.close()

    save_labels(split_labels, dst_split_labels_file)
    save_labels(all_labels, dst_class_labels_file)
    save_labels(train_labels, dst_train_class_labels_file)
    save_labels(test_labels, dst_test_class_labels_file)

import data_set_20ng
import numpy as np
import doc_classification


def split_vectors():
    all_vecs_file_name = 'e:/dc/20ng_data/vecs/all_doc_vec_joint_400.bin'
    split_labels_file_name = 'e:/dc/20ng_data/split/doc_split_labels.bin'
    # file_name_head = 'e:/dc/20ng_data/vecs/all'
    # train_vecs_file_name = 'train' + all_vecs_file_name[len(file_name_head):]
    # test_vecs_file_name = 'test' + all_vecs_file_name[len(file_name_head):]
    train_vecs_file_name = 'e:/dc/20ng_data/split/train_doc_vec_joint_400.bin'
    test_vecs_file_name = 'e:/dc/20ng_data/split/test_doc_vec_joint_400.bin'
    data_set_20ng.split_vecs(all_vecs_file_name, split_labels_file_name,
                             train_vecs_file_name, test_vecs_file_name)


def make_dataset_info():
    dir_path = 'e:/dc/20_newsgroups'
    dst_path_list_file = 'e:/dc/20ng_data/all_doc_path_list.txt'
    dst_split_labels_file = 'e:/dc/20ng_data/split/doc_split_labels.bin'
    dst_class_labels_file = 'e:/dc/20ng_data/all_doc_class_labels.bin'
    dst_train_labels_file = 'e:/dc/20ng_data/split/train_labels.bin'
    dst_test_labels_file = 'e:/dc/20ng_data/split/test_labels.bin'
    data_set_20ng.index_20ng_dataset(dir_path, dst_path_list_file, dst_class_labels_file,
                                     dst_split_labels_file, dst_train_labels_file,
                                     dst_test_labels_file)


def make_bydate_dataset_info():
    dir_path = 'e:/dc/20news-bydate'
    dst_path_list_file = 'e:/dc/20ng_bydate/all_doc_path_list.txt'
    dst_class_labels_file = 'e:/dc/20ng_bydate/all_doc_class_labels.bin'
    dst_split_labels_file = 'e:/dc/20ng_bydate/doc_split_labels.bin'
    dst_train_labels_file = 'e:/dc/20ng_bydate/train_labels.bin'
    dst_test_labels_file = 'e:/dc/20ng_bydate/test_labels.bin'
    data_set_20ng.index_20ng_bydate(dir_path, dst_path_list_file, dst_class_labels_file,
                                    dst_split_labels_file, dst_train_labels_file,
                                    dst_test_labels_file)


def split_and_classify():
    # all_vecs_file_name = 'e:/dc/20ng_bydate/vecs/doc_vec_cpp_100.bin'
    # train_vecs_file_name = 'e:/dc/20ng_bydate/vecs/train_doc_vec_cpp_100.bin'
    # test_vecs_file_name = 'e:/dc/20ng_bydate/vecs/test_doc_vec_cpp_100.bin'

    # all_vecs_file_name = 'e:/dc/20ng_bydate/vecs/dbow_doc_vec_100.bin'
    # train_vecs_file_name = 'e:/dc/20ng_bydate/vecs/train_dbow_doc_vec_100.bin'
    # test_vecs_file_name = 'e:/dc/20ng_bydate/vecs/test_dbow_doc_vec_100.bin'

    # all_vecs_file_name = 'e:/dc/20ng_bydate/vecs/all_doc_vec_joint_oml_100.bin'
    # train_vecs_file_name = 'e:/dc/20ng_bydate/vecs/train_doc_vec_joint_oml_100.bin'
    # test_vecs_file_name = 'e:/dc/20ng_bydate/vecs/test_doc_vec_joint_oml_100.bin'

    # all_vecs_file_name = 'e:/dc/20ng_bydate/vecs/all_doc_vec_joint_oml_200.bin'
    # train_vecs_file_name = 'e:/dc/20ng_bydate/vecs/train_doc_vec_joint_oml_200.bin'
    # test_vecs_file_name = 'e:/dc/20ng_bydate/vecs/test_doc_vec_joint_oml_200.bin'

    all_vecs_file_name = 'e:/dc/20ng_bydate/vecs/doc_vecs_0.bin'
    train_vecs_file_name = 'e:/dc/20ng_bydate/vecs/train_dedw_vecs.bin'
    test_vecs_file_name = 'e:/dc/20ng_bydate/vecs/test_dedw_vecs.bin'

    split_labels_file_name = 'e:/dc/20ng_bydate/doc_split_labels.bin'
    data_set_20ng.split_vecs(all_vecs_file_name, split_labels_file_name,
                             train_vecs_file_name, test_vecs_file_name)
    train_label_file = 'e:/dc/20ng_bydate/train_labels.bin'
    test_label_file = 'e:/dc/20ng_bydate/test_labels.bin'
    doc_classification.doc_classification(train_vecs_file_name, train_label_file, test_vecs_file_name,
                                          test_label_file, 0, -1)


def test():
    va = np.random.rand(1, 3)[0]
    print va
    va = va[1:]
    print va
    print type(va)


def main():
    # test()
    # make_dataset_info()
    # split_vectors()
    split_and_classify()
    # make_bydate_dataset_info()

if __name__ == '__main__':
    main()

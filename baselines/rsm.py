import numpy as np
import os

import dataarange
from textclassification import doc_classification_lr, doc_classification_svm, get_scores_label_file
from ioutils import text_vecs_to_bin


def __job_text_vecs_to_bin_file():
    # datadir = 'e:/data/emadr/nyt-all/business/rsm'
    # datadir = 'e:/data/emadr/20ng_bydate/rsm'
    datadir = 'e:/data/emadr/nyt-less-docs/arts/rsm'
    minoc = 70
    text_vecs_file = os.path.join(datadir, 'rsm-hidden-%d.txt' % minoc)
    binfile = os.path.join(datadir, 'rsm-vecs-%d.bin' % minoc)
    # text_vecs_file = os.path.join(datadir, 'drbm-vecs-100-30.txt')
    # binfile = os.path.join(datadir, 'drbm-vecs.bin')
    text_vecs_to_bin(text_vecs_file, binfile)


def __nyt_classification():
    # datadir = 'e:/data/emadr/20ng_bydate'
    # datadir = 'e:/data/emadr/nyt-all/world'
    datadir = 'e:/data/emadr/nyt-less-docs/arts'
    all_vecs_file_name = os.path.join(datadir, 'rsm/rsm-vecs-70.bin')
    # all_vecs_file_name = os.path.join(datadir, 'rsm/drbm-vecs.bin')
    # split_labels_file_name = os.path.join(datadir, 'bindata/data-split-labels.bin')
    split_labels_file_name = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_label_file = os.path.join(datadir, 'bindata/train-labels.bin')
    test_label_file = os.path.join(datadir, 'bindata/test-labels.bin')
    train_vecs_file_name = os.path.join(datadir, 'rsm/train-rsm-vecs.bin')
    test_vecs_file_name = os.path.join(datadir, 'rsm/test-rsm-vecs.bin')

    dataarange.split_vecs(all_vecs_file_name, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)
    doc_classification_lr(train_vecs_file_name, train_label_file, test_vecs_file_name,
                          test_label_file, 0, -1)


def __job_text_vecs_to_bin_classification():
    datadir = 'e:/data/emadr/nyt-less-docs/world'
    minoc = 100
    text_vecs_file = os.path.join(datadir, 'rsm/rsm-hidden-%d.txt' % minoc)
    binfile = os.path.join(datadir, 'rsm/rsm-vecs-%d.bin' % minoc)
    # text_vecs_file = os.path.join(datadir, 'drbm-vecs-100-30.txt')
    # binfile = os.path.join(datadir, 'drbm-vecs.bin')
    text_vecs_to_bin(text_vecs_file, binfile)

    # all_vecs_file_name = os.path.join(datadir, 'rsm/drbm-vecs.bin')
    # split_labels_file_name = os.path.join(datadir, 'bindata/data-split-labels.bin')
    split_labels_file_name = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_label_file = os.path.join(datadir, 'bindata/train-labels.bin')
    test_label_file = os.path.join(datadir, 'bindata/test-labels.bin')
    train_vecs_file_name = os.path.join(datadir, 'rsm/train-rsm-vecs.bin')
    test_vecs_file_name = os.path.join(datadir, 'rsm/test-rsm-vecs.bin')

    dataarange.split_vecs(binfile, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)
    # doc_classification_lr(train_vecs_file_name, train_label_file, test_vecs_file_name,
    #                       test_label_file, 0, -1)
    y_pred_test = doc_classification_svm(train_vecs_file_name, train_label_file, test_vecs_file_name, 0, -1)
    get_scores_label_file(test_label_file, y_pred_test)


if __name__ == '__main__':
    # __job_text_vecs_to_bin_file()
    # __nyt_classification()
    __job_text_vecs_to_bin_classification()

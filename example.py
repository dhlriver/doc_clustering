import os
from itertools import izip
import ioutils
from tngdataset import doc_classes


def __load_doc_paths(doc_paths_file):
    doc_paths = list()
    f = open(doc_paths_file, 'r')
    for line in f:
        doc_paths.append(line.strip())
    f.close()
    return doc_paths


def __get_test_doc_paths(doc_paths_file, dataset_split_labels_file):
    test_doc_paths = list()
    doc_paths = __load_doc_paths(doc_paths_file)
    split_labels = ioutils.load_labels_file(dataset_split_labels_file)
    assert len(doc_paths) == len(split_labels)
    for doc_path, sl in izip(doc_paths, split_labels):
        if sl == 2:
            test_doc_paths.append(doc_path)
    return test_doc_paths


def __get_doc_text(doc_path):
    f = open(doc_path)
    doc_text = f.read()
    f.close()
    return doc_text


def __emadr_vs_pvdbow():
    datadir = 'e:/data/emadr/20ng_bydate'
    doc_paths_file = os.path.join(datadir, 'all_doc_path_list.txt')
    dataset_split_labels_file = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    y_true_file = os.path.join(datadir, 'bindata/test-labels.bin')
    y_pred_pvdbow_file = os.path.join(datadir, 'bindata/ypred-pvdbow.bin')
    y_pred_emadr_file = os.path.join(datadir, 'bindata/ypred-emadr.bin')
    dst_file = os.path.join(datadir, 'example-candidates.txt')

    test_doc_paths = __get_test_doc_paths(doc_paths_file, dataset_split_labels_file)

    y_true = ioutils.load_labels_file(y_true_file)
    y_pvdbow = ioutils.load_labels_file(y_pred_pvdbow_file)
    y_emadr = ioutils.load_labels_file(y_pred_emadr_file)
    fout = open(dst_file, 'wb')
    cnt = 0
    for i, tup in enumerate(izip(y_true, y_pvdbow, y_emadr)):
        yt, y0, y1 = tup
        if yt != y0 and yt == y1:
            print test_doc_paths[i], doc_classes[yt], doc_classes[y0]
            doc_path = os.path.join(datadir, test_doc_paths[i][20:])
            fout.write('%s\t%s\t%s\n' % (doc_path, doc_classes[yt], doc_classes[y0]))
            doc_text = __get_doc_text(doc_path)
            fout.write('%s\n' % doc_text)
            cnt += 1
    print cnt, len(y_true), float(cnt) / len(y_true)
    fout.close()


if __name__ == '__main__':
    __emadr_vs_pvdbow()

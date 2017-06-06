import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from sklearn.linear_model import LogisticRegression

import dataarange


def load_features(file_name):
    fin = open(file_name, 'rb')
    num_vecs, vec_len = np.fromfile(fin, dtype=np.int32, count=2)
    print 'file:', file_name, 'num_vecs:', num_vecs, 'dim:', vec_len
    vec_list = list()
    for i in xrange(num_vecs):
        vec = np.fromfile(fin, np.float32, vec_len)
        # vec /= np.linalg.norm(vec)
        vec_list.append(vec)
    fin.close()
    return vec_list


def load_labels(file_name):
    fin = open(file_name, 'rb')
    num_labels = np.fromfile(fin, np.int32, 1)
    labels = np.fromfile(fin, np.int32, num_labels)
    fin.close()
    return labels


def __truncate_vecs(vec_list, vec_beg, vec_end):
    for idx, vec in enumerate(vec_list):
        if vec_end != -1:
            vec_list[idx] = vec[vec_beg:vec_end]
        else:
            vec_list[idx] = vec[vec_beg:]


def __doc_classification(classifier, train_vec_file, train_label_file, test_vec_file, test_label_file,
                         vec_beg=0, vec_end=-1):
    train_x = load_features(train_vec_file)
    train_y = load_labels(train_label_file)

    test_x = load_features(test_vec_file)
    test_y = load_labels(test_label_file)
    # print train_y[1000:1100]
    # print test_y[1000:1100]

    print train_x[0][50:60]

    if vec_beg != 0 or vec_end != -1:
        __truncate_vecs(train_x, vec_beg, vec_end)
        __truncate_vecs(test_x, vec_beg, vec_end)

    print 'training model ...'
    classifier.fit(train_x, train_y)
    print 'done.'

    y_pred = classifier.predict(test_x)
    acc = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred, average='macro')
    recall = recall_score(test_y, y_pred, average='macro')
    f1 = f1_score(test_y, y_pred, average='macro')
    print 'accuracy', acc
    print 'precision', prec
    print 'recall', recall
    print 'macro f1', f1
    print '%f\t%f\t%f\t%f' % (acc, prec, recall, f1)


def doc_classification_lr(train_vec_file, train_label_file, test_vec_file, test_label_file,
                          vec_beg=0, vec_end=-1):
    classifier = LogisticRegression(C=1000, multi_class='multinomial', solver='newton-cg', max_iter=100)
    __doc_classification(classifier, train_vec_file, train_label_file, test_vec_file,
                         test_label_file, vec_beg, vec_end)


# use validation set for hyperparameters
def doc_classification_svm(train_vec_file, train_label_file, test_vec_file, test_label_file,
                           vec_beg=0, vec_end=-1):
    train_x = load_features(train_vec_file)
    train_y = load_labels(train_label_file)

    test_x = load_features(test_vec_file)
    test_y = load_labels(test_label_file)
    # print train_y[1000:1100]
    # print test_y[1000:1100]

    print train_x[0][50:60]

    def trunc_vecs(vec_list):
        for idx, vec in enumerate(vec_list):
            if vec_end != -1:
                vec_list[idx] = vec[vec_beg:vec_end]
            else:
                vec_list[idx] = vec[vec_beg:]

    if vec_beg != 0 or vec_end != -1:
        trunc_vecs(train_x)
        trunc_vecs(test_x)

    print 'training svm ...'
    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear')
    # clf = svm.LinearSVC(dual=False)
    clf.fit(train_x, train_y)
    print 'done.'

    y_pred = clf.predict(test_x)
    acc = accuracy_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred, average='macro')
    recall = recall_score(test_y, y_pred, average='macro')
    f1 = f1_score(test_y, y_pred, average='macro')
    print 'accuracy', acc
    print 'precision', prec
    print 'recall', recall
    print 'macro f1', f1
    print '%f\t%f\t%f\t%f' % (acc, prec, recall, f1)

    return acc, prec, recall, f1


def __20ng_classification():
    all_vecs_file_name = 'e:/data/emadr/20ng_bydate/vecs/dew-vecs-0_8-50.bin'
    # all_vecs_file_name = 'e:/data/emadr/20ng_bydate/vecs/dedw-vecs.bin'
    split_labels_file_name = 'e:/data/emadr/20ng_bydate/bindata/dataset-split-labels.bin'
    train_label_file = 'e:/data/emadr/20ng_bydate/bindata/train-labels.bin'
    test_label_file = 'e:/data/emadr/20ng_bydate/bindata/test-labels.bin'
    train_vecs_file_name = 'e:/data/emadr/20ng_bydate/bindata/train-dedw-vecs.bin'
    test_vecs_file_name = 'e:/data/emadr/20ng_bydate/bindata/test-dedw-vecs.bin'

    dataarange.split_vecs(all_vecs_file_name, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)
    # doc_classification_lr(train_vecs_file_name, train_label_file, test_vecs_file_name,
    #                       test_label_file, 0, -1)
    doc_classification_svm(train_vecs_file_name, train_label_file, test_vecs_file_name,
                           test_label_file, 0, -1)


def __nyt_classification():
    # datadir = 'e:/data/emadr/nyt-world-full/processed/'
    datadir = 'f:/data/emadr/nyt-less-docs/world/'
    all_vecs_file_name = os.path.join(datadir, 'vecs/dew-vecs-0_9-40.bin')
    split_labels_file_name = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_label_file = os.path.join(datadir, 'bindata/train-labels.bin')
    test_label_file = os.path.join(datadir, 'bindata/test-labels.bin')
    train_vecs_file_name = os.path.join(datadir, 'vecs/train-dedw-vecs.bin')
    test_vecs_file_name = os.path.join(datadir, 'vecs/test-dedw-vecs.bin')

    dataarange.split_vecs(all_vecs_file_name, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)
    doc_classification_svm(train_vecs_file_name, train_label_file, test_vecs_file_name,
                           test_label_file, 0, -1)
    # doc_classification_lr(train_vecs_file_name, train_label_file, test_vecs_file_name,
    #                       test_label_file, 0, -1)


def __job_split_vecs():
    datadir = 'e:/data/emadr/nyt-less-docs/business/'
    all_vecs_file_name = os.path.join(datadir, 'vecs/dew-vecs-100-0_15-40.bin')
    split_labels_file_name = os.path.join(datadir, 'bindata/dataset-split-labels.bin')
    train_vecs_file_name = os.path.join(datadir, 'vecs/train-dedw-vecs.bin')
    test_vecs_file_name = os.path.join(datadir, 'vecs/test-dedw-vecs.bin')

    dataarange.split_vecs(all_vecs_file_name, split_labels_file_name,
                          train_vecs_file_name, test_vecs_file_name, train_label=0, test_label=2)


if __name__ == '__main__':
    # __20ng_classification()
    # __job_split_vecs()
    __nyt_classification()
    # doc_classification()

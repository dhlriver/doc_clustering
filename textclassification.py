import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_features(file_name):
    fin = open(file_name, 'rb')
    x = np.fromfile(fin, dtype=np.int32, count=2)
    num_vecs = x[0]
    vec_len = x[1]
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


def doc_classification(train_vec_file, train_label_file, test_vec_file, test_label_file,
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
    clf = svm.SVC(decision_function_shape='ovo')
    # clf = svm.LinearSVC()
    clf.fit(train_x, train_y)
    print 'done.'

    y_pred = clf.predict(test_x)
    print 'accuracy', accuracy_score(test_y, y_pred)
    print 'precision', precision_score(test_y, y_pred, average='macro')
    print 'recall', recall_score(test_y, y_pred, average='macro')
    print 'macro f1', f1_score(test_y, y_pred, average='macro')


def main():
    print 'classify'
    # doc_classification()

if __name__ == '__main__':
    main()

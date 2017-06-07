import os
import numpy as np
import re
from random import randint
from itertools import izip

import ioutils
import textclassification
import dataarange
import textutils


doc_classes = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
               "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
               "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
               "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
               "talk.politics.misc", "talk.religion.misc"]


def __get_doc_paths(dirpath):
    path_list = list()
    for f in os.listdir(dirpath):
        cur_path = os.path.join(dirpath, f)
        if not os.path.isdir(cur_path):
            continue
        print cur_path

        for f1 in os.listdir(cur_path):
            cur_path1 = os.path.join(cur_path, f1)
            if not os.path.isfile(cur_path1):
                continue

            path_list.append(cur_path1)
    return path_list


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

    ioutils.save_labels(split_labels, dst_split_labels_file)
    ioutils.save_labels(all_labels, dst_class_labels_file)
    ioutils.save_labels(train_labels, dst_train_class_labels_file)
    ioutils.save_labels(test_labels, dst_test_class_labels_file)


def __load_doc_paths(doc_list_file):
    doc_paths = list()
    fin = open(doc_list_file, 'rb')
    for line in fin:
        doc_paths.append(line[:-1])
    fin.close()
    return doc_paths


subject_head = 'Subject: '
num_lines_head = 'Lines: '


def __read_20ng_text(doc_file):
    text = ''
    start_text = False
    fin = open(doc_file, 'rb')
    for line in fin:
        if not start_text:
            if line.startswith(subject_head):
                text += line[len(subject_head):]
            if line.strip() == '':
                start_text = True
        else:
            text += line
            # if line[len(num_lines_head):-1] == 'dog':
            #     print doc_file
            # num_lines = int(line[len(num_lines_head):-1])
            # print num_lines
    fin.close()

    return text


# a list of docs to a file with one line per doc
def pack_docs_for_ner(doc_list_file, dst_file):
    doc_paths = __load_doc_paths(doc_list_file)
    fout = open(dst_file, 'wb')
    for i, doc_path in enumerate(doc_paths):
        doc_text = __read_20ng_text(doc_path)

        doc_text = re.sub('\n\n+', '  ', doc_text)
        doc_text = re.sub('\s', ' ', doc_text)

        doc_text = re.sub('<[^\s]*>', '', doc_text)
        doc_text = re.sub('(>+)|(}\|*)|(\|)', '', doc_text)

        # doc_text = re.sub('\n\n+', ' ', doc_text)
        doc_text = re.sub('\s\s+', '; ', doc_text)
        doc_text = re.sub('([\.,\?!:"]);', '\g<1>', doc_text)
        fout.write(doc_text.decode('mbcs').encode('utf-8') + '\n')
        # if i == 10:
        #     break
    fout.close()


def __gen_words_dict():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed'
    data_dir = 'e:/data/emadr/20ng_bydate'
    word_cnt_file_name = os.path.join(data_dir, 'words-dict-lc.txt')
    stop_words_file_name = 'e:/data/common-res/stopwords.txt'
    dst_file_name = os.path.join(data_dir, 'words-dict-proper.txt')
    textutils.gen_proper_words_dict_with_cnts(word_cnt_file_name, stop_words_file_name, 2, 20,
                                              dst_file_name)


def __gen_lowercase_token_file():
    data_dir = 'e:/data/emadr/20ng_bydate/'

    tokenized_line_docs_file_name = os.path.join(data_dir, 'docs-tokenized-lc.txt')
    # proper_word_cnts_dict_file = os.path.join(data_dir, 'words-dict-lc.txt')
    proper_word_cnts_dict_file = os.path.join(data_dir, 'words-dict-proper.txt')
    dataset_split_file = os.path.join(data_dir, 'bindata/dataset-split-labels.bin')
    max_word_len = 20
    min_occurrance = 2
    all_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-%d.txt' % min_occurrance)
    train_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-train-%d.txt' % min_occurrance)
    val_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-val-%d.txt' % min_occurrance)
    test_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-test-%d.txt' % min_occurrance)

    textutils.gen_lowercase_token_file(tokenized_line_docs_file_name, proper_word_cnts_dict_file,
                                       max_word_len, min_occurrance, all_doc_text_file)

    # textutils.split_docs_text_file_by_dataset_labels(all_doc_text_file, dataset_split_file, train_doc_text_file,
    #                                                  test_doc_text_file)

    textutils.split_docs_text_file_by_dataset_labels_tvt(all_doc_text_file, dataset_split_file, train_doc_text_file,
                                                         val_doc_text_file, test_doc_text_file)


def __index_dataset_docs():
    dirpath = 'e:/data/emadr/20news-bydate/'
    dst_doc_paths_file = 'e:/data/emadr/20ng_bydate/docpaths.txt'

    train_files_path = os.path.join(dirpath, '20news-bydate-train')
    test_files_path = os.path.join(dirpath, '20news-bydate-test')
    train_docs = __get_doc_paths(train_files_path)
    test_docs = __get_doc_paths(test_files_path)
    all_docs = train_docs + test_docs
    fout = open(dst_doc_paths_file, 'wb')
    for doc_path in all_docs:
        fout.write('%s\n' % doc_path)
    fout.close()


def __split_dataset():
    doc_paths_file = 'e:/data/emadr/20ng_bydate/docpaths.txt'
    dataset_labels_file = 'e:/data/emadr/20ng_bydate/bindata/dataset-split-labels.bin'

    fin = open(doc_paths_file, 'r')
    docpaths = list()
    for line in fin:
        docpaths.append(line.strip())
    fin.close()

    dataset_labels = list()
    for docpath in docpaths:
        if 'test' in docpath:
            dataset_labels.append(2)
        elif 'train' in docpath:
            rv = randint(0, 4)
            # rv = 1
            if rv == 0:
                dataset_labels.append(1)
            else:
                dataset_labels.append(0)

    fout = open(dataset_labels_file, 'wb')
    np.asarray([len(docpaths)], np.int32).tofile(fout)
    np.asarray(dataset_labels, np.int32).tofile(fout)
    fout.close()


def __gen_class_labels():
    doc_paths_file = 'e:/data/emadr/20ng_bydate/docpaths.txt'
    dataset_labels_file = 'e:/data/emadr/20ng_bydate/bindata/dataset-split-labels.bin'
    all_docs_class_labels_file = 'e:/data/emadr/20ng_bydate/bindata/labels.bin'
    training_class_labels_file = 'e:/data/emadr/20ng_bydate/bindata/train-labels.bin'
    validation_class_labels_file = 'e:/data/emadr/20ng_bydate/bindata/val-labels.bin'
    testing_class_labels_file = 'e:/data/emadr/20ng_bydate/bindata/test-labels.bin'

    fin = open(doc_paths_file, 'r')
    docpaths = list()
    for line in fin:
        docpaths.append(line.strip())
    fin.close()

    all_labels, train_labels, val_labels, test_labels = list(), list(), list(), list()
    dataset_split_labels = ioutils.load_labels_file(dataset_labels_file)
    for dataset_split_label, docpath in izip(dataset_split_labels, docpaths):
        class_label_idx = 0
        for lidx, cl in enumerate(doc_classes):
            if cl in docpath:
                class_label_idx = lidx
        print dataset_split_label, docpath, class_label_idx
        all_labels.append(class_label_idx)
        if dataset_split_label == 0:
            train_labels.append(class_label_idx)
        elif dataset_split_label == 1:
            val_labels.append(class_label_idx)
        else:
            test_labels.append(class_label_idx)

    ioutils.save_labels(all_labels, all_docs_class_labels_file)
    ioutils.save_labels(train_labels, training_class_labels_file)
    ioutils.save_labels(val_labels, validation_class_labels_file)
    ioutils.save_labels(test_labels, testing_class_labels_file)


def __get_doc_text(docpath):
    f = open(docpath, 'r')
    doc_text = ''
    for line in f:
        if line.startswith('Subject: '):
            doc_text += line[9:]
        if not line.strip():
            break

    for line in f:
        doc_text += line
    f.close()
    doc_text = re.sub('\n>+', '\n', doc_text)
    doc_text = re.sub('[\r\n]+', ' ', doc_text)
    return doc_text


def __gen_docs_text_file():
    doc_paths_file = 'e:/data/emadr/20ng_bydate/docpaths.txt'
    dst_file = 'e:/data/emadr/20ng_bydate/docs.txt'

    fin = open(doc_paths_file, 'r')
    fout = open(dst_file, 'wb')
    for i, line in enumerate(fin):
        doc_path = line.strip()
        doc_text = __get_doc_text(doc_path)
        # print doc_text
        fout.write('%s\n' % doc_text)
        if i % 1000 == 0:
            print i
        # break
    fout.close()
    fin.close()


def __gen_word_cnts_dict():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed'
    data_dir = 'e:/data/emadr/20ng_bydate'

    tokenized_line_docs_file = os.path.join(data_dir, 'docs-tokenized.txt')
    dst_file_name = os.path.join(data_dir, 'word-cnts-lc.txt')
    textutils.gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file, dst_file_name)
    dst_file_name = os.path.join(data_dir, 'word-cnts-with-case.txt')
    textutils.gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file, dst_file_name, tolower=False)


def __setup_entity_pairs_file():
    doc_list_file = 'e:/dc/20ng_bydate/all_doc_path_list.txt'
    docs_ner_file = 'e:/dc/20ng_bydate/docs-for-ner.txt'
    pack_docs_for_ner(doc_list_file, docs_ner_file)

    docs_ner_file = 'e:/dc/20ng_bydate/docs-for-ner.txt'
    ner_result_file = 'e:/dc/20ng_bydate/ner-result.txt'
    cooccur_mentions_file = 'e:/dc/20ng_bydate/cooccur-mentions.txt'
    # dataarange.gen_ee_pairs_with_ner_result(docs_ner_file, ner_result_file, cooccur_mentions_file)

    # gen entity name dict
    entity_name_dict_file = 'e:/dc/20ng_bydate/entity-names-ner.txt'
    # dataarange.gen_entity_name_dict(ner_result_file, entity_name_dict_file)

    ner_result_file = 'e:/dc/20ng_bydate/ner-result.txt'
    doc_all_mentions_file = 'e:/dc/20ng_bydate/doc-mentions-ner.txt'
    # dataarange.ner_result_to_tab_sep(ner_result_file, doc_all_mentions_file)

    name_dict_file = 'e:/dc/20ng_bydate/entity-names-ner.txt'
    doc_entity_file = 'e:/dc/20ng_bydate/bin/de-ner.bin'
    dataarange.gen_doc_entity_pairs(name_dict_file, doc_all_mentions_file, doc_entity_file)

    entity_candidate_cliques_file = 'e:/dc/20ng_bydate/cooccur-mentions.txt'
    ee_file = 'e:/dc/20ng_bydate/bin/ee-ner.bin'
    dataarange.gen_entity_entity_pairs(name_dict_file, entity_candidate_cliques_file, ee_file)

    cnts_file = 'e:/dc/20ng_bydate/bin/entity-cnts-ner.bin'
    dataarange.gen_cnts_file(doc_entity_file, cnts_file)


def gen_files_for_twe():
    stopwords_file = 'e:/common-res/stopwords.txt'
    text_file = 'e:/dc/20ng_bydate/doc_text_data.txt'
    words_dict_file = 'e:/dc/20ng_bydate/twe/words.txt'
    # textutils.gen_word_cnts_dict_with_line_docs(text_file, words_dict_file,
    #                                             tolower=False, stopwords_file=stopwords_file)

    lda_input_file = 'e:/dc/20ng_bydate/twe/docs.txt'
    textutils.filter_words_in_line_docs(text_file, words_dict_file, lda_input_file)


def __gen_dw():
    data_dir = 'e:/data/emadr/20ng_bydate/'
    min_occurrence = 30
    # proper_word_cnts_dict_file = os.path.join(data_dir, 'words-dict-lc.txt')
    proper_word_cnts_dict_file = os.path.join(data_dir, 'words-dict-proper.txt')

    line_docs_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-2.txt')
    dst_bow_docs_file = os.path.join(data_dir, 'bindata/dw-%d.bin' % min_occurrence)

    words_dict = textutils.load_words_to_idx_dict(proper_word_cnts_dict_file, min_occurrence)
    print 'vocab size:', len(words_dict)
    textutils.line_docs_to_bow(line_docs_file, words_dict, min_occurrence, dst_bow_docs_file)

    dst_word_cnts_file = os.path.join(data_dir, 'bindata/word-cnts-%d.bin' % min_occurrence)
    textutils.gen_word_cnts_file_from_bow_file(dst_bow_docs_file, dst_word_cnts_file)

    train_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-train-2.txt')
    val_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-val-2.txt')
    test_doc_text_file = os.path.join(data_dir, 'tokenizedlc/docs-tokenized-lc-test-2.txt')
    dst_train_dw_file = os.path.join(data_dir, 'bindata/dw-train-%d.bin' % min_occurrence)
    dst_val_dw_file = os.path.join(data_dir, 'bindata/dw-val-%d.bin' % min_occurrence)
    dst_test_dw_file = os.path.join(data_dir, 'bindata/dw-test-%d.bin' % min_occurrence)

    textutils.line_docs_to_bow(train_doc_text_file, words_dict, min_occurrence, dst_train_dw_file)
    textutils.line_docs_to_bow(val_doc_text_file, words_dict, min_occurrence, dst_val_dw_file)
    textutils.line_docs_to_bow(test_doc_text_file, words_dict, min_occurrence, dst_test_dw_file)


def __gen_idx_cnt_file():
    min_occurance = 30
    datadir = 'e:/data/emadr/20ng_bydate'
    dict_file = os.path.join(datadir, 'words-dict-lc.txt')
    docs_file = os.path.join(datadir, 'tokenizedlc/docs-tokenized-lc-%d.txt' % min_occurance)
    dst_file = os.path.join(datadir, 'rsm/docs-tokenized-idx-cnt-%d.txt' % min_occurance)
    textutils.line_docs_to_idx_cnt_no_dict(docs_file, dst_file)


def __test():
    print 'test'

if __name__ == '__main__':
    # __index_dataset_docs()
    # __split_dataset()
    # __gen_class_labels()
    # __gen_docs_text_file()

    # __gen_words_dict()
    # __gen_lowercase_token_file()
    # __gen_idx_cnt_file()
    __gen_dw()
    # __setup_entity_pairs_file()
    # gen_files_for_twe()
    # __test()
    pass

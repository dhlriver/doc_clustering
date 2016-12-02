import os
import numpy as np
import re

import ioutils
import textclassification
import dataarange
import textutils


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


def __make_bydate_dataset_info():
    dir_path = 'e:/dc/20news-bydate'
    dst_path_list_file = 'e:/dc/20ng_bydate/all_doc_path_list.txt'
    dst_class_labels_file = 'e:/dc/20ng_bydate/all_doc_class_labels.bin'
    dst_split_labels_file = 'e:/data/emadr/20ng_bydate/doc_split_labels.bin'
    dst_train_labels_file = 'e:/data/emadr/20ng_bydate/train_labels.bin'
    dst_test_labels_file = 'e:/data/emadr/20ng_bydate/test_labels.bin'
    # index_20ng_bydate(dir_path, dst_path_list_file, dst_class_labels_file,
    #                   dst_split_labels_file, dst_train_labels_file,
    #                   dst_test_labels_file)
    split_labels = ioutils.load_labels_file(dst_split_labels_file)
    train_labels = ioutils.load_labels_file(dst_train_labels_file)
    test_labels = ioutils.load_labels_file(dst_test_labels_file)
    print len(split_labels), len(train_labels), len(test_labels)


def setup_entity_pairs_file():
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


def __gen_lowercase_token_file():
    dataset_split_file = 'e:/data/emadr/20ng_bydate/data-split-labels.bin'
    max_word_len = 20
    min_occurrence = 2
    all_doc_text_file = 'e:/data/emadr/20ng_bydate/docs-tokenized-lc.txt'
    test_doc_text_file = 'e:/data/emadr/20ng_bydate/docs-tokenized-lc-test.txt'
    train_doc_text_file = 'e:/data/emadr/20ng_bydate/docs-tokenized-lc-train.txt'

    # textutils.gen_lowercase_token_file(tokenized_line_docs_file_name, proper_word_cnts_dict_file,
    #                                    max_word_len, min_occurrance, all_doc_text_file)

    textutils.split_docs_text_file_by_dataset_labels(all_doc_text_file, dataset_split_file, train_doc_text_file,
                                                     test_doc_text_file)


def __gen_dw():
    min_occurrence = 50
    proper_word_cnts_dict_file = 'e:/data/emadr/20ng_bydate/words-dict-lc.txt'

    line_docs_file_name = 'e:/data/emadr/20ng_bydate/docs-tokenized-lc.txt'
    dst_bow_docs_file_name = 'e:/data/emadr/20ng_bydate/bin/dw-%d.bin' % min_occurrence

    words_dict = textutils.load_words_to_idx_dict(proper_word_cnts_dict_file, min_occurrence)
    # textutils.line_docs_to_bow(line_docs_file_name, words_dict, min_occurrence, dst_bow_docs_file_name)

    dst_word_cnts_file = 'e:/data/emadr/20ng_bydate/bin/word-cnts-%d.bin' % min_occurrence
    # textutils.gen_word_cnts_file_from_bow_file(dst_bow_docs_file_name, dst_word_cnts_file)

    train_doc_text_file = 'e:/data/emadr/20ng_bydate/docs-tokenized-lc-train.txt'
    test_doc_text_file = 'e:/data/emadr/20ng_bydate/docs-tokenized-lc-test.txt'
    dst_train_dw_file = 'e:/data/emadr/20ng_bydate/bin/dw-train-%d.bin' % min_occurrence
    dst_test_dw_file = 'e:/data/emadr/20ng_bydate/bin/dw-test-%d.bin' % min_occurrence
    textutils.line_docs_to_bow(train_doc_text_file, words_dict, min_occurrence, dst_train_dw_file)
    textutils.line_docs_to_bow(test_doc_text_file, words_dict, min_occurrence, dst_test_dw_file)


def __test():
    print 'test'

if __name__ == '__main__':
    # __make_bydate_dataset_info()
    # __gen_lowercase_token_file()
    __gen_dw()
    # setup_entity_pairs_file()
    # gen_files_for_twe()
    # __test()

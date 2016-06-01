# -*- coding: utf-8 -*-
import os
import re
import numpy as np

import textutils
import dataarange
from random import shuffle


def __get_dirs(year_data_dirs):
    dir_list = list()
    for year_data_dir in year_data_dirs:
        for f in os.listdir(year_data_dir):
            month_dir = os.path.join(year_data_dir, f)
            for mf in os.listdir(month_dir):
                day_dir = os.path.join(month_dir, mf)
                dir_list.append(day_dir)

    return dir_list


head_str = 'link:http://www.nytimes.com/2014/01/01/'
head_len = len(head_str)


def __get_info(doc_file_path):
    info_file_path = doc_file_path[:-4] + '.info'
    fin = open(info_file_path, 'rb')
    fin.next()  # skip title line
    linkline = fin.next()
    fin.close()

    vals = linkline[head_len:].split('/')
    return vals[0], vals[1]


world_categories = ['africa', 'europe', 'americas', 'asia', 'middleeast']


def __get_category_to_label_dict():
    category_dict = dict()
    for idx, cat in enumerate(world_categories):
        category_dict[cat] = idx
    return category_dict


def __doc_info_file_to_label_file(doc_info_file, dst_label_file):
    category_dict = __get_category_to_label_dict()

    label_list = list()
    fin = open(doc_info_file, 'rb')
    for line in fin:
        vals = line.strip().split('\t')
        cur_label = category_dict[vals[1]]
        label_list.append(cur_label)
    fin.close()

    fout = open(dst_label_file, 'wb')
    np.asarray([len(label_list)], np.int32).tofile(fout)
    np.asarray(label_list, np.int32).tofile(fout)
    fout.close()


def nyt_dataset_merge():
    # year_data_dirs = ['e:/dc/nyt-world-full/2013', 'e:/dc/nyt-world-full/2014']
    year_data_dirs = ['e:/dc/nyt-world-full/2011', 'e:/dc/nyt-world-full/2012']

    # doc_list_file = 'e:/dc/nyt-world-full/processed/train/doclist.txt'
    # docs_file = 'e:/dc/nyt-world-full/processed/train/docs.txt'
    # doc_info_file = 'e:/dc/nyt-world-full/processed/train/docinfo.txt'
    # doc_labels_file = 'e:/dc/nyt-world-full/processed/train/doc-labels.bin'

    doc_list_file = 'e:/dc/nyt-world-full/processed/doclist-train.txt'
    docs_file = 'e:/dc/nyt-world-full/processed/docs-train.txt'
    doc_info_file = 'e:/dc/nyt-world-full/processed/docinfo-train.txt'
    doc_labels_file = 'e:/dc/nyt-world-full/processed/doc-labels-train.bin'

    dir_list = __get_dirs(year_data_dirs)

    fout_doclist = open(doc_list_file, 'wb')
    fout_docinfo = open(doc_info_file, 'wb')
    fout_doc = open(docs_file, 'wb')

    doc_cnt = 0
    for docdir in dir_list:
        for f in os.listdir(docdir.decode('utf-8')):
            if not f.endswith('.txt'):
                continue
            doc_file_path = os.path.join(docdir, f)

            cat0, cat1 = __get_info(doc_file_path)

            if cat1 not in world_categories:
                continue

            fout_doclist.write(doc_file_path.encode('utf-8') + '\n')

            fout_docinfo.write('%s\t%s\n' % (cat0, cat1))

            # write text
            fin = open(doc_file_path, 'rb')
            doc_text = fin.read()
            fin.close()
            doc_text = re.sub('\s', ' ', doc_text)
            doc_text = re.sub('\s\s+', ' ', doc_text)
            fout_doc.write(doc_text + '\n')

            doc_cnt += 1

            if doc_cnt % 10000 == 0:
                print doc_cnt

        # if doc_cnt == 2043:
        #     break
    print '%d documents processed.' % doc_cnt

    fout_doclist.close()
    fout_docinfo.close()
    fout_doc.close()

    __doc_info_file_to_label_file(doc_info_file, doc_labels_file)


def gen_word_cnts_dict():
    tokenized_line_docs_file_name = 'e:/dc/nyt-world-full/processed/docs-tokenized.txt'
    dst_file_name = 'e:/dc/nyt-world-full/processed/word_cnts_lc.txt'
    textutils.gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file_name, dst_file_name)
    dst_file_name = 'e:/dc/nyt-world-full/processed/word_cnts_with_case.txt'
    textutils.gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file_name, dst_file_name, tolower=False)


def gen_words_dict_nyt():
    word_cnt_file_name = 'e:/dc/nyt-world-full/processed/word_cnts_lc.txt'
    stop_words_file_name = 'e:/common_res/stopwords.txt'
    dst_file_name = 'e:/dc/nyt-world-full/processed/words_dict_proper.txt'
    textutils.gen_proper_words_dict_with_cnts(word_cnt_file_name, stop_words_file_name, 2, 20,
                                              dst_file_name)


def gen_lowercase_token_file_nyt():
    tokenized_line_docs_file_name = 'e:/dc/nyt-world-full/processed/docs-tokenized.txt'
    proper_word_cnts_dict_file = 'e:/dc/nyt-world-full/processed/words_dict_proper.txt'
    max_word_len = 20
    min_occurrance = 40
    dst_file_name = 'e:/dc/nyt-world-full/processed/docs-tokenized-lc-%d.txt' % min_occurrance
    textutils.gen_lowercase_token_file(tokenized_line_docs_file_name, proper_word_cnts_dict_file,
                                       max_word_len, min_occurrance, dst_file_name)


def gen_dw_nyt():
    min_occurrance = 40
    line_docs_file_name = 'e:/dc/nyt-world-full/processed/docs_tokenized_lc.txt'
    proper_word_cnts_dict_file = 'e:/dc/nyt-world-full/processed/words_dict_proper.txt'
    dst_bow_docs_file_name = 'e:/dc/nyt-world-full/processed/bin/dw-%d.bin' % min_occurrance
    textutils.line_docs_to_bow(line_docs_file_name, proper_word_cnts_dict_file, min_occurrance, dst_bow_docs_file_name)

    dst_word_cnts_file = 'e:/dc/nyt-world-full/processed/bin/word-cnts-%d.bin' % min_occurrance
    textutils.gen_word_cnts_file_from_bow_file(dst_bow_docs_file_name, dst_word_cnts_file)


def retrieve_mentions():
    line_docs_file_name = 'e:/dc/nyt-world-full/processed/docs-tokenized.txt'
    illegal_start_words_file = 'e:/dc/20ng_bydate/stopwords.txt'

    output_dir = 'e:/dc/nyt-world-full/processed/mentions/'
    dst_doc_entity_candidates_list_file = output_dir + 'doc_mention_candidates.txt'
    dst_entity_candidate_clique_file = output_dir + 'mention_candidate_cliques.txt'
    dst_doc_entity_indices_file = output_dir + 'doc_mention_candidate_indices.txt'
    # init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
    #                 dst_entity_candidate_clique_file, dst_doc_entity_indices_file)

    lc_word_cnts_file_name = 'e:/dc/el/wiki/wiki_word_cnts_lc.txt'
    wc_word_cnts_file_name = 'e:/dc/el/wiki/wiki_word_cnts_with_case.txt'
    proper_entity_dict_file = output_dir + 'entity_names.txt'
    # gen_entity_name_dict(dst_doc_entity_candidates_list_file, lc_word_cnts_file_name, wc_word_cnts_file_name,
    #                      proper_entity_dict_file)

    de_file_bin = 'e:/dc/nyt-world-full/processed/bin/de.bin'
    de_file_txt = 'e:/dc/nyt-world-full/processed/bin/de.txt'
    dataarange.gen_doc_entity_pairs(proper_entity_dict_file, dst_doc_entity_candidates_list_file, de_file_bin,
                                    de_file_txt)

    entity_candidate_cliques_file = dst_entity_candidate_clique_file
    ee_file = 'e:/dc/nyt-world-full/processed/bin/ee.bin'
    dataarange.gen_entity_entity_pairs(proper_entity_dict_file, entity_candidate_cliques_file, ee_file)


def get_cnts_file():
    adj_list_file = 'e:/dc/nyt-world-full/processed/bin/de.bin'
    cnts_file = 'e:/dc/nyt-world-full/processed/bin/entity-cnts.bin'
    dataarange.gen_cnts_file(adj_list_file, cnts_file)


def doc_info_to_labels():
    doc_info_file = 'e:/dc/nyt-world-full/processed/docinfo.txt'
    doc_labels_file = 'e:/dc/nyt-world-full/processed/doc-labels.bin'
    __doc_info_file_to_label_file(doc_info_file, doc_labels_file)


def setup_entity_pairs_file():
    # docs_ner_file = 'e:/dc/nyt-world-full/processed/docs.txt'
    # ner_result_file = 'e:/dc/nyt-world-full/processed/ner-result.txt'
    # cooccur_mentions_file = 'e:/dc/nyt-world-full/processed/mentions-ner/cooccur-mentions.txt'
    # entity_name_dict_file = 'e:/dc/nyt-world-full/processed/mentions-ner/entity-names-nloc.txt'
    # ner_result_file = 'e:/dc/nyt-world-full/processed/ner-result.txt'
    # doc_all_mentions_file = 'e:/dc/nyt-world-full/processed/mentions-ner/doc-mentions.txt'
    # doc_entity_file = 'e:/dc/nyt-world-full/processed/bin/de-ner.bin'
    # ee_file = 'e:/dc/nyt-world-full/processed/bin/ee-ner.bin'
    # cnts_file = 'e:/dc/nyt-world-full/processed/bin/entity-cnts-ner.bin'

    docs_ner_file = 'e:/dc/nyt-world-full/processed/train/docs.txt'
    ner_result_file = 'e:/dc/nyt-world-full/processed/train/ner-result.txt'
    cooccur_mentions_file = 'e:/dc/nyt-world-full/processed/train/cooccur-mentions.txt'
    entity_name_dict_file = 'e:/dc/nyt-world-full/processed/train/entity-names-nloc.txt'
    ee_file = 'e:/dc/nyt-world-full/processed/bin/ee-ner-train.bin'

    # dataarange.gen_ee_pairs_with_ner_result(docs_ner_file, ner_result_file, cooccur_mentions_file)

    # gen entity name dict
    dataarange.gen_entity_name_dict(ner_result_file, entity_name_dict_file, True)
    # dataarange.ner_result_to_tab_sep(ner_result_file, doc_all_mentions_file)

    # dataarange.gen_doc_entity_pairs(entity_name_dict_file, doc_all_mentions_file, doc_entity_file)

    dataarange.gen_entity_entity_pairs(entity_name_dict_file, cooccur_mentions_file, ee_file)

    # dataarange.gen_cnts_file(doc_entity_file, cnts_file)


def gen_bow_file():
    min_occurrance = 40
    tokenized_data_file = 'e:/dc/nyt-world-full/processed/docs_tokenized_lc.txt'
    words_dict_file = 'e:/dc/nyt-world-full/processed/words_dict_proper.txt'
    bow_file = 'e:/dc/nyt-world-full/processed/nyt-docs-bow-rsm-%d.txt' % min_occurrance
    textutils.tokenized_text_to_bow(tokenized_data_file, words_dict_file, bow_file, min_occurrance)


if __name__ == '__main__':
    nyt_dataset_merge()
    # dataset_statistics()

    # gen_word_cnts_dict()
    # gen_words_dict_nyt()
    # gen_lowercase_token_file_nyt()
    # gen_dw_nyt()

    # setup_entity_pairs_file()
    # gen_bow_file()

    # retrieve_mentions()
    # get_cnts_file()
    # doc_info_to_labels()

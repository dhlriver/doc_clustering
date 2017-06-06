# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from itertools import izip
from collections import Counter
from random import shuffle

from ioutils import load_labels_file, save_labels
import textutils
import dataarange


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


def __get_sub_classes(categories_file):
    f = open(categories_file, 'r')
    line = f.next()
    f.close()

    return line.strip().split(' ')


def __parse_url(url_line):
    url_ptn_str = '/(\d\d\d\d)/\d\d/\d\d/(.*?)/(.*?)/'
    m = re.search(url_ptn_str, url_line)
    if not m:
        return None

    syear = m.group(1)
    main_class = m.group(2)
    sub_class = m.group(3)
    return int(syear), main_class, sub_class


def __gen_class_info_file():
    datadir = 'e:/data/emadr/nyt-all/'
    wanted_main_class = 'sports'
    beg_year = 2010
    dst_file = os.path.join(datadir, '%s/labels-info-2010.txt' % wanted_main_class)
    sub_class_cnts = dict()
    nyt_all_file = os.path.join(datadir, 'nyt_single.txt')
    f = open(nyt_all_file, 'r')
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print i

        url_line = f.next().strip()
        # print url_line

        for text_line in f:
            if not text_line.strip():
                break

        pr = __parse_url(url_line)
        if not pr:
            continue

        doc_year, main_class, sub_class = pr

        if main_class != wanted_main_class:
            continue

        if doc_year < beg_year:
            continue

        cur_sub_class_cnt = sub_class_cnts.get(sub_class, 0)
        sub_class_cnts[sub_class] = cur_sub_class_cnt + 1
    f.close()

    fout = open(dst_file, 'wb')
    for sub_class, cnt in sub_class_cnts.iteritems():
        fout.write('%s\t%d\n' % (sub_class, cnt))
    fout.close()


def __show_sub_classes_info(sub_classes_dict):
    for mc, sclist in sub_classes_dict.iteritems():
        print mc, len(sclist)


def __split_by_main_classes():
    nyt_all_file = 'e:/data/emadr/nyt-all/nyt_single.txt'
    # dst_data_dir = 'e:/data/emadr/nyt-all/'
    dst_data_dir = 'e:/data/emadr/nyt-less-docs/'
    wanted_classes = ['arts', 'business', 'sports']
    beg_year = 2010
    # wanted_classes = ['world']
    all_sub_classes = list()
    for wanted_class in wanted_classes:
        all_sub_classes.append(__get_sub_classes('e:/data/emadr/nyt-less-docs/%s/subcategories.txt'
                                                 % wanted_class))
    # all_sub_classes.append(__get_sub_classes('e:/data/emadr/nyt-less-docs/world/subcategories.txt'))

    fouts_text, fouts_label = list(), list()
    for main_class in wanted_classes:
        fout_text = open(os.path.join(dst_data_dir, main_class, 'docs.txt'), 'wb')
        fouts_text.append(fout_text)
        fout_label = open(os.path.join(dst_data_dir, main_class, 'labels.txt'), 'wb')
        fouts_label.append(fout_label)

    sub_classes_dict = dict()
    for wc in wanted_classes:
        sub_classes_dict[wc] = list()

    url_ptn_str = '/(\d\d\d\d)/\d\d/\d\d/(.*?)/(.*?)/'
    f = open(nyt_all_file, 'r')
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print i

        url_line = f.next().strip()
        # print url_line
        # else:
        #     print url_line

        doc_text = ''
        for text_line in f:
            if text_line.strip():
                doc_text += text_line
            else:
                break

        pr = __parse_url(url_line)
        if not pr:
            continue

        doc_year, main_class, sub_class = pr

        if doc_year < beg_year:
            continue

        if main_class not in wanted_classes:
            continue

        class_idx = wanted_classes.index(main_class)

        if sub_class not in all_sub_classes[class_idx]:
            continue

        sub_classes_dict[main_class].append(sub_class)

        cur_fout_text = fouts_text[class_idx]
        doc_text = doc_text.strip().replace('\n', ' ')
        cur_fout_text.write('%s\n' % doc_text)

        cur_fout_label = fouts_label[class_idx]
        cur_fout_label.write('%s\t%s\n' % (main_class, sub_class))
        # if i >= 500:
        #     break
    f.close()

    for fout in fouts_text:
        fout.close()
    for fout in fouts_label:
        fout.close()


def __gen_bin_label_file():
    # labels_file = 'e:/data/emadr/nyt-all/arts/labels.txt'
    # dst_file = 'e:/data/emadr/nyt-all/arts/labels.bin'
    # labels_file = 'e:/data/emadr/nyt-all/business/labels.txt'
    # dst_file = 'e:/data/emadr/nyt-all/business/labels.bin'
    # labels_file = 'e:/data/emadr/nyt-all/sports/labels.txt'
    # dst_file = 'e:/data/emadr/nyt-all/sports/labels.bin'
    # labels_file = 'e:/data/emadr/nyt-all/world/labels.txt'
    # dst_file = 'e:/data/emadr/nyt-all/world/labels.bin'
    labels_file = 'e:/data/emadr/nyt-less-docs/business/labels.txt'
    dst_file = 'e:/data/emadr/nyt-less-docs/business/bindata/labels.bin'

    f = open(labels_file, 'r')
    sub_label_cnts = dict()
    sub_labels = list()
    for line in f:
        main_label, sub_label = line.strip().split('\t')
        sub_labels.append(sub_label)
        cnt = sub_label_cnts.get(sub_label, 0)
        sub_label_cnts[sub_label] = cnt + 1
    f.close()

    label_indices = dict()
    for i, (l, cnt) in enumerate(sub_label_cnts.iteritems()):
        label_indices[l] = i + 1
        print '%d\t%s\t%d' % (i + 1, l, cnt)

    labels_arr = np.zeros(len(sub_labels), np.int32)
    for i, sub_label in enumerate(sub_labels):
        labels_arr[i] = label_indices[sub_label]

    fout = open(dst_file, 'wb')
    np.asarray([len(labels_arr)], np.int32).tofile(fout)
    labels_arr.tofile(fout)
    fout.close()


def __nyt_dataset_merge():
    # year_data_dirs = ['e:/dc/nyt-world-full/2013', 'e:/dc/nyt-world-full/2014']
    # year_data_dirs = ['e:/dc/nyt-world-full/2011', 'e:/dc/nyt-world-full/2012']
    # year_data_dirs = ['e:/dc/nyt-world-full/2013', 'e:/dc/nyt-world-full/2014']
    year_data_dirs = ['e:/data/emadr/nyt-world-full/2011', 'e:/data/emadr/nyt-world-full/2012',
                      'e:/data/emadr/nyt-world-full/2013', 'e:/data/emadr/nyt-world-full/2014']

    # doc_list_file = 'e:/dc/nyt-world-full/processed/train/doclist.txt'
    # docs_file = 'e:/dc/nyt-world-full/processed/train/docs.txt'
    # doc_info_file = 'e:/dc/nyt-world-full/processed/train/docinfo.txt'
    # doc_labels_file = 'e:/dc/nyt-world-full/processed/train/doc-labels.bin'

    # doc_list_file = 'e:/dc/nyt-world-full/processed/doclist-train.txt'
    # docs_file = 'e:/dc/nyt-world-full/processed/docs-train.txt'
    # doc_info_file = 'e:/dc/nyt-world-full/processed/docinfo-train.txt'
    # doc_labels_file = 'e:/dc/nyt-world-full/processed/doc-labels-train.bin'

    doc_list_file = 'e:/data/emadr/nyt-world-full/processed/doclist.txt'
    docs_file = 'e:/data/emadr/nyt-world-full/processed/docs.txt'
    doc_info_file = 'e:/data/emadr/nyt-world-full/processed/doc-labels.txt'
    doc_labels_file = 'e:/data/emadr/nyt-world-full/processed/doc-labels.bin'

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


def __gen_word_cnts_dict():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed'
    data_dir = 'e:/data/emadr/nyt-less-docs/business'

    tokenized_line_docs_file = os.path.join(data_dir, 'docs-tokenized.txt')
    dst_file_name = os.path.join(data_dir, 'word-cnts-lc.txt')
    textutils.gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file, dst_file_name)
    dst_file_name = os.path.join(data_dir, 'word-cnts-with-case.txt')
    textutils.gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file, dst_file_name, tolower=False)


def __gen_words_dict_nyt():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed'
    data_dir = 'e:/data/emadr/nyt-less-docs/business'
    word_cnt_file_name = os.path.join(data_dir, 'word-cnts-lc.txt')
    stop_words_file_name = 'e:/data/common-res/stopwords.txt'
    dst_file_name = os.path.join(data_dir, 'words-dict-proper.txt')
    textutils.gen_proper_words_dict_with_cnts(word_cnt_file_name, stop_words_file_name, 2, 20,
                                              dst_file_name)


def __gen_lowercase_token_file_nyt():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed/'
    data_dir = 'e:/data/emadr/nyt-less-docs/world'

    tokenized_line_docs_file_name = os.path.join(data_dir, 'docs-tokenized.txt')
    proper_word_cnts_dict_file = os.path.join(data_dir, 'words-dict-proper.txt')
    dataset_split_file = os.path.join(data_dir, 'bindata/dataset-split-labels.bin')
    max_word_len = 20
    min_occurrance = 100
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


def __gen_dw_nyt():
    # data_dir = 'e:/data/emadr/nyt-world-full/processed/'
    data_dir = 'e:/data/emadr/nyt-less-docs/business'
    min_occurrence = 10
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


def doc_info_to_labels():
    doc_info_file = 'e:/dc/nyt-world-full/processed/docinfo.txt'
    doc_labels_file = 'e:/dc/nyt-world-full/processed/doc-labels.bin'
    __doc_info_file_to_label_file(doc_info_file, doc_labels_file)


def __setup_entity_pairs_file():
    # docs_ner_file = 'e:/dc/nyt-world-full/processed/docs.txt'
    # ner_result_file = 'e:/dc/nyt-world-full/processed/ner-result.txt'
    # cooccur_mentions_file = 'e:/dc/nyt-world-full/processed/mentions-ner/cooccur-mentions.txt'
    # entity_name_dict_file = 'e:/dc/nyt-world-full/processed/mentions-ner/entity-names-nloc.txt'
    # ner_result_file = 'e:/dc/nyt-world-full/processed/ner-result.txt'

    # datadir = 'e:/data/emadr/nyt-world-full/processed/'
    datadir = 'e:/data/emadr/nyt-less-docs/sports'
    filter_loc = False

    docs_ner_file = os.path.join(datadir, 'docs.txt')
    ner_result_file = os.path.join(datadir, 'ner-result.txt')
    cooccur_mentions_file = os.path.join(datadir, 'cooccur-mentions.txt')
    entity_name_dict_file = os.path.join(datadir, 'entity-names.txt')
    doc_all_mentions_file = os.path.join(datadir, 'doc-mentions.txt')
    ee_file = os.path.join(datadir, 'bindata/ee.bin')
    de_file = os.path.join(datadir, 'bindata/de.bin')
    cnts_file = os.path.join(datadir, 'bindata/entity-cnts.bin')

    dataarange.gen_ee_pairs_with_ner_result(docs_ner_file, ner_result_file, cooccur_mentions_file)

    # gen entity name dict
    dataarange.gen_entity_name_dict(ner_result_file, entity_name_dict_file, filter_loc)

    dataarange.ner_result_to_tab_sep(ner_result_file, doc_all_mentions_file)
    dataarange.gen_doc_entity_pairs(entity_name_dict_file, doc_all_mentions_file, de_file)

    dataarange.gen_entity_entity_pairs(entity_name_dict_file, cooccur_mentions_file, ee_file)

    dataarange.gen_cnts_file(de_file, cnts_file)


def gen_bow_file():
    min_occurrance = 40
    tokenized_data_file = 'e:/dc/nyt-world-full/processed/docs_tokenized_lc.txt'
    words_dict_file = 'e:/dc/nyt-world-full/processed/words_dict_proper.txt'
    bow_file = 'e:/dc/nyt-world-full/processed/nyt-docs-bow-rsm-%d.txt' % min_occurrance
    textutils.tokenized_text_to_bow(tokenized_data_file, words_dict_file, bow_file, min_occurrance)


def __gen_data_split_labels_file():
    doc_labels_file = 'e:/data/emadr/nyt-world-full/processed/doc-labels.bin'
    data_split_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/data-split-labels.bin'
    train_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/train-labels.bin'
    test_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/test-labels.bin'
    f = open(doc_labels_file, 'rb')
    nums = np.fromfile(f, np.int32, 1)
    all_labels = np.fromfile(f, np.int32, nums)
    f.close()

    split_labels = np.random.randint(0, 2, nums)
    fout_data_split = open(data_split_labels_file, 'wb')
    nums.tofile(fout_data_split)
    split_labels.tofile(fout_data_split)
    fout_data_split.close()

    train_labels = [cl for cl, sl in izip(all_labels, split_labels) if sl == 0]
    test_labels = [cl for cl, sl in izip(all_labels, split_labels) if sl == 1]
    save_labels(train_labels, train_labels_file)
    save_labels(test_labels, test_labels_file)


def __gen_idx_cnt_file():
    min_occurance = 100
    # for mc in ['arts', 'business', 'sports', 'world']:
    for mc in ['world']:
        datadir = 'e:/data/emadr/nyt-less-docs/%s' % mc
        dict_file = os.path.join(datadir, 'words-dict-proper.txt')
        # docs_file = os.path.join(datadir, 'tokenizedlc/docs-tokenized-lc-%d-part.txt' % min_occurance)
        # dst_file = os.path.join(datadir, 'rsm/docs-tokenized-idx-cnt-%d-part.txt' % min_occurance)
        docs_file = os.path.join(datadir, 'tokenizedlc/docs-tokenized-lc-%d.txt' % min_occurance)
        dst_file = os.path.join(datadir, 'rsm/docs-tokenized-idx-cnt-%d.txt' % min_occurance)
        textutils.line_docs_to_idx_cnt_no_dict(docs_file, dst_file)

    # textutils.line_docs_to_idx_cnt(docs_file, dict_file, dst_file, min_occurance)


def __gen_data_split_labels_tvt():
    # doc_labels_file = 'e:/data/emadr/nyt-world-full/processed/doc-labels.bin'
    # data_split_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/data-split-labels.bin'
    # train_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/train-labels.bin'
    # val_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/val-labels.bin'
    # test_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/test-labels.bin'

    main_class = 'business'
    doc_labels_file = 'e:/data/emadr/nyt-less-docs/%s/bindata/labels.bin' % main_class
    data_split_labels_file = 'e:/data/emadr/nyt-less-docs/%s/bindata/dataset-split-labels.bin' % main_class
    train_labels_file = 'e:/data/emadr/nyt-less-docs/%s/bindata/train-labels.bin' % main_class
    val_labels_file = 'e:/data/emadr/nyt-less-docs/%s/bindata/val-labels.bin' % main_class
    test_labels_file = 'e:/data/emadr/nyt-less-docs/%s/bindata/test-labels.bin' % main_class

    all_labels = load_labels_file(doc_labels_file)
    num_labels = len(all_labels)

    split_labels = np.random.randint(0, 3, num_labels)
    fout_data_split = open(data_split_labels_file, 'wb')
    np.asarray([num_labels], np.int32).tofile(fout_data_split)
    split_labels.tofile(fout_data_split)
    fout_data_split.close()

    def write_labels(cur_labels, filename):
        fout = open(filename, 'wb')
        np.asarray([len(cur_labels)], dtype=np.int32).tofile(fout)
        np.asarray(cur_labels, dtype=np.int32).tofile(fout)
        fout.close()

    train_labels = [cl for cl, sl in izip(all_labels, split_labels) if sl == 0]
    val_labels = [cl for cl, sl in izip(all_labels, split_labels) if sl == 1]
    test_labels = [cl for cl, sl in izip(all_labels, split_labels) if sl == 2]
    write_labels(train_labels, train_labels_file)
    write_labels(val_labels, val_labels_file)
    write_labels(test_labels, test_labels_file)


def __classification():
    print 'ok'


def __test():
    doc_labels_file = 'e:/data/emadr/nyt-world-full/processed/doc-labels.bin'
    data_split_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/data-split-labels.bin'
    train_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/train-labels.bin'
    test_labels_file = 'e:/data/emadr/nyt-world-full/processed/bin/test-labels.bin'

    doc_labels = load_labels_file(doc_labels_file)
    print len(doc_labels), doc_labels[:20]
    split_labels = load_labels_file(data_split_labels_file)
    print len(split_labels), split_labels[:20]
    train_labels = load_labels_file(train_labels_file)
    print len(train_labels), train_labels[:20]
    test_labels = load_labels_file(test_labels_file)
    print len(test_labels), test_labels[:20]


if __name__ == '__main__':
    # __test()
    __nyt_dataset_merge()

    # __gen_class_info_file()
    __split_by_main_classes()
    __gen_bin_label_file()

    # __gen_data_split_labels_tvt()

    __gen_word_cnts_dict()
    __gen_words_dict_nyt()
    __gen_lowercase_token_file_nyt()
    # __gen_dw_nyt()
    # __gen_idx_cnt_file()

    __setup_entity_pairs_file()
    # __gen_data_split_labels_file()

    # __classification()

    # gen_bow_file()

    # dataset_statistics()
    # retrieve_mentions()
    # doc_info_to_labels()
    pass

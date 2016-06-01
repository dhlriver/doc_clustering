import re
import os
import numpy as np
from time import time
from array import array
import ioutils
import textutils


def gen_legal_wid_list_file():
    # mid_alias_cnt_file_name = 'd:/data/el/merged_fb_mid_alias_cnt.txt'
    filter_mid_file_name = 'e:/el/res/freebase/filter_mids_10_8.bin'
    mid_wid_file_name = 'd:/data/el/mid_to_wid_full_ord_wid.txt'
    dst_wid_file_name = 'e:/dc/el/legal_wid_list.bin'
    print 'loading filter mids ...'
    fin = open(filter_mid_file_name, 'rb')
    num_mids = np.fromfile(fin, '>i4', 1)
    filter_mid_set = set()
    for i in xrange(num_mids):
        mid = ioutils.read_str_with_fixed_len(fin, 8)
        filter_mid_set.add(mid)
    fin.close()
    print 'done.'

    print 'loading wids ...'
    fin = open(mid_wid_file_name, 'rb')
    wid_list = list()
    for line in fin:
        vals = line.strip().split('\t')
        mid = vals[0]
        wid = int(vals[1])
        if mid not in filter_mid_set:
            wid_list.append(wid)
    fin.close()
    print 'done.'

    print len(wid_list), 'wids'
    fout = open(dst_wid_file_name, 'wb')
    np.asarray([len(wid_list)], np.int32).tofile(fout)
    np.asarray(wid_list, np.int32).tofile(fout)
    fout.close()
    print 'done.'


def gen_line_docs_file_wiki():
    wiki_texts_file_name = 'e:/el/tmpres/wiki/enwiki-20150403-text-with-links-no-title-main_cleaned.txt'
    legal_wid_file_name = 'e:/dc/el/legal_wid_list.bin'
    dst_line_docs_file_name = 'e:/dc/el/wiki_lines.txt'
    page_ids_file_name = 'e:/dc/el/wiki_page_ids.bin'

    fin = open(legal_wid_file_name, 'rb')
    num_wids = np.fromfile(fin, np.int32, 1)
    legal_wids = np.fromfile(fin, np.int32, num_wids)
    fin.close()

    fin = open(wiki_texts_file_name, 'rb')
    fout0 = open(dst_line_docs_file_name, 'wb')
    fout1 = open(page_ids_file_name, 'wb')
    np.ones(1, np.int32).tofile(fout1)

    page_cnt = 0
    written_page_cnt = 0
    id_list = list()
    title = fin.readline()
    legal_wid_idx = 0
    while title:
        if legal_wid_idx == len(legal_wids):
            break

        page_cnt += 1
        wid = int(fin.readline().strip())
        num_line = int(fin.readline().strip())
        text = ''
        for i in xrange(num_line):
            line = fin.readline()
            if i != 0:
                text += ' <s> '
            text += line.strip()

        while legal_wid_idx < len(legal_wids) and legal_wids[legal_wid_idx] < wid:
            legal_wid_idx += 1
        if legal_wid_idx == len(legal_wids):
            break
        if legal_wids[legal_wid_idx] != wid:
            title = fin.readline()
            continue

        id_list.append(wid)
        if len(id_list) == 1000000:
            np.asarray(id_list, np.int32).tofile(fout1)
            id_list = list()

        text = re.sub('\[\[(.*?)\|(.*?)\]\]', '\g<2>', text)
        text = re.sub('\[\[(.*?)\]\]', '\g<1>', text)
        fout0.write(text + '\n')

        written_page_cnt += 1
        if written_page_cnt % 10000 == 0:
            print written_page_cnt
        # if written_page_cnt == 100:
        #     break
        title = fin.readline()

    print page_cnt, 'pages,', written_page_cnt, 'written'
    np.asarray(id_list, np.int32).tofile(fout1)
    fout1.seek(0)
    np.asarray([written_page_cnt], np.int32).tofile(fout1)
    fin.close()
    fout0.close()
    fout1.close()


def gen_word_cnts_dict():
    tokenized_line_docs_file_name = 'e:/dc/el/wiki_lines_tokenized.txt'
    dst_file_name = 'e:/dc/el/wiki/wiki_word_cnts_lc.txt'
    textutils.gen_word_cnts_dict_for_line_docs(tokenized_line_docs_file_name, dst_file_name)
    dst_file_name = 'e:/dc/el/wiki/wiki_word_cnts_with_case.txt'
    textutils.gen_word_cnts_dict_for_line_docs(tokenized_line_docs_file_name, dst_file_name, to_lower=False)


def gen_words_dict_wiki():
    word_cnt_file_name = 'e:/dc/el/wiki/wiki_word_cnts_lc.txt'
    stop_words_file_name = 'e:/common_res/stopwords.txt'
    dst_file_name = 'e:/dc/el/wiki/words_dict_proper.txt'
    textutils.gen_proper_words_dict_with_cnts(word_cnt_file_name, stop_words_file_name, 4, 20,
                                              dst_file_name)


def gen_lowercase_token_file_wiki():
    tokenized_line_docs_file_name = 'e:/dc/el/wiki/wiki_lines_tokenized.txt'
    proper_word_cnts_dict_file = 'e:/dc/el/wiki/words_dict_proper.txt'
    max_word_len = 20
    dst_file_name = 'e:/dc/el/wiki/wiki_lines_tokenized_lc.txt'
    textutils.gen_lowercase_token_file(tokenized_line_docs_file_name, proper_word_cnts_dict_file,
                                       max_word_len, dst_file_name)


def gen_bow_wiki():
    line_docs_file_name = 'e:/dc/el/wiki/wiki_lines_tokenized_lc.txt'
    proper_word_cnts_dict_file = 'e:/dc/el/wiki/words_dict_proper.txt'
    dst_bow_docs_file_name = 'e:/dc/el/wiki/wiki_bow.bin'
    # text_process_common.line_docs_to_bow(line_docs_file_name, proper_word_cnts_dict_file, dst_bow_docs_file_name)

    dst_word_cnts_file = 'e:/dc/el/wiki/word_cnts.bin'
    textutils.gen_word_cnts_file_from_bow_file(dst_bow_docs_file_name, dst_word_cnts_file)


def tac_el_job_14train():
    docs_dir = r'D:\data\el\LDC2015E20_EDL_2014\data\training\source_documents'
    line_docs_file = 'e:/dc/el/tac/tac_2014_train_docs_text.txt'
    docs_list_file = 'e:/dc/el/tac/tac_2014_train_docs_list.txt'
    # gen_line_docs_file_tac(docs_dir, line_docs_file, docs_list_file)

    tokenized_line_docs_file = 'e:/dc/el/tac/tac_2014_train_docs_text_tokenized.txt'
    proper_word_cnts_dict_file = 'e:/dc/el/wiki/words_dict_proper.txt'
    max_word_len = 20
    tokenized_line_docs_lc_file = 'e:/dc/el/tac/tac_2014_train_docs_text_tokenized_lc.txt'
    textutils.gen_lowercase_token_file(tokenized_line_docs_file, proper_word_cnts_dict_file,
                                       max_word_len, tokenized_line_docs_lc_file)

    bow_docs_file = 'e:/dc/el/tac/tac_2014_train_docs_bow.bin'
    # text_process_common.line_docs_to_bow(tokenized_line_docs_lc_file, proper_word_cnts_dict_file, bow_docs_file)


def tac_el_job_14eval():
    docs_dir = r'D:\data\el\LDC2015E20_EDL_2014\data\eval\source_documents'
    line_docs_file = 'e:/dc/el/tac/tac_2014_eval_docs_text.txt'
    docs_list_file = 'e:/dc/el/tac/tac_2014_eval_docs_list.txt'
    # gen_line_docs_file_tac(docs_dir, line_docs_file, docs_list_file)

    tokenized_line_docs_file = 'e:/dc/el/tac/tac_2014_eval_docs_text_tokenized.txt'
    proper_word_cnts_dict_file = 'e:/dc/el/wiki/words_dict_proper.txt'
    max_word_len = 20
    tokenized_line_docs_lc_file = 'e:/dc/el/tac/tac_2014_eval_docs_text_tokenized_lc.txt'
    # text_process_common.gen_lowercase_token_file(tokenized_line_docs_file, proper_word_cnts_dict_file,
    #                                              max_word_len, tokenized_line_docs_lc_file)

    bow_docs_file = 'e:/dc/el/tac/tac_2014_eval_docs_bow.bin'
    textutils.line_docs_to_bow(tokenized_line_docs_lc_file, proper_word_cnts_dict_file, bow_docs_file)


def clean_line_wiki_docs_file():
    line_docs_file = 'e:/dc/el/wiki/wiki_lines.txt'
    dst_file = 'e:/dc/el/wiki/wiki-docs.txt'
    fin = open(line_docs_file, 'rb')
    fout = open(dst_file, 'wb')
    for i, line in enumerate(fin):
        # line = line.replace(' <s> ', ' ')
        line = re.sub(' <s> | \(\)', ' ', line)
        fout.write(line)
        # if i == 1000:
        #     break
        if (i + 1) % 10000 == 0:
            print i + 1
    fin.close()
    fout.close()


def split_line_docs():
    lines_docs_file = 'e:/dc/el/wiki/wiki-docs.txt'
    # lines_docs_file = 'e:/dc/el/wiki/tmp.txt'
    num_files = 20
    dst_files = list()
    for i in xrange(num_files):
        dst_files.append('e:/dc/el/wiki/wiki-docs-split/wiki-docs-%d.txt' % i)
        # dst_files.append('e:/dc/el/wiki/wiki-docs-split/tmp-%d.txt' % i)
    textutils.split_line_docs_file(lines_docs_file, dst_files)


def test():
    wiki_texts_file_name = 'e:/el/tmpres/wiki/enwiki-20150403-text-with-links-no-title-main_cleaned.txt'
    dst_line_docs_file_name = 'e:/dc/el/wiki_lines.txt'
    page_ids_file_name = 'e:/dc/el/wiki_page_ids.bin'
    legal_wid_file_name = 'e:/dc/el/legal_wid_list.bin'

    fin = open(legal_wid_file_name, 'rb')
    num_wids = np.fromfile(fin, np.int32, 1)
    print num_wids
    legal_wids = np.fromfile(fin, np.int32, num_wids)
    fin.close()

    fin = open(wiki_texts_file_name, 'rb')
    page_cnt = 0
    written_page_cnt = 0
    title = fin.readline()
    legal_wid_idx = 0
    while title:
        page_cnt += 1
        if page_cnt % 100000 == 100000 - 1:
            print page_cnt + 1

        wid = int(fin.readline().strip())
        num_line = int(fin.readline().strip())
        for i in xrange(num_line):
            fin.readline()

        while legal_wid_idx < len(legal_wids) and legal_wids[legal_wid_idx] < wid:
            legal_wid_idx += 1
        if legal_wid_idx == len(legal_wids) or legal_wids[legal_wid_idx] != wid:
            title = fin.readline()
            continue

        if wid == 23235:
            print 'hit', wid

        written_page_cnt += 1
        title = fin.readline()

    print page_cnt, 'pages,', written_page_cnt, 'written'
    fin.close()


if __name__ == '__main__':
    start_time = time()

    # gen_legal_wid_list_file()
    # gen_line_docs_file_wiki()
    # gen_word_cnts_dict()
    # gen_words_dict_wiki()
    # gen_lowercase_token_file_wiki()
    # gen_bow_wiki()

    # tac_el_job_14train()
    # tac_el_job_10eval()
    # tac_el_job()

    # clean_line_wiki_docs_file()
    split_line_docs()

    # test()
    print 'Elapsed time:', time() - start_time

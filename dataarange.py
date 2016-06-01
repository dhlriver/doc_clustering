import numpy as np
from time import time
from array import array
from itertools import izip

import text_process_common
from nltk import word_tokenize


con_words = ['of', 'and', 'at']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']


def first_letter_uppercase(word):
    if word == 'I':
        return False

    if not word[0].isupper():
        return False

    for ch in word:
        if (not ch.isalpha()) and ch != '-' and ch != '.':
            return False

    return not word.isupper()


def entity_candidate_cliques_in_words(words, illegal_start_words):
    idx = 0
    cliques = list()
    indices_list = list()
    cur_clique_entites = list()
    while idx < len(words):
        cur_word = words[idx]
        if (cur_word.lower() in illegal_start_words) or (len(cur_word) == 0 or len(cur_word) > 20):
            idx += 1
            continue

        if text_process_common.is_sentence_end(cur_word):
            if cur_clique_entites:
                cliques.append(cur_clique_entites)
                cur_clique_entites = list()

        if text_process_common.all_uppercase_word(cur_word):
            cur_clique_entites.append(cur_word)
            indices_list.append([idx])
        elif first_letter_uppercase(cur_word):
            cur_name = cur_word
            beg_idx = idx + 1
            while idx + 1 < len(words) and (first_letter_uppercase(words[idx + 1]) or words[idx + 1] in con_words):
                idx += 1
            while words[idx].lower() in con_words and idx > beg_idx - 1:
                idx -= 1

            if idx + 1 <= beg_idx and cur_name in months:
                idx += 1
                continue

            cur_indices = [beg_idx - 1]
            for i in xrange(beg_idx, idx + 1):
                cur_name += ' ' + words[i]
                cur_indices.append(i)
            if 2 < len(cur_name) < 50:
                cur_clique_entites.append(cur_name)
                indices_list.append(cur_indices)
        idx += 1

    if cur_clique_entites:
        cliques.append(cur_clique_entites)

    return cliques, indices_list


def init_entity_net(tokenized_line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                    dst_entity_candidate_clique_file, dst_doc_entity_indices_file=None):
    illegal_start_words = text_process_common.load_word_set(illegal_start_words_file, has_num_words=True)

    line_cnt = 0
    fin = open(tokenized_line_docs_file_name, 'rb')
    fout0 = open(dst_doc_entity_candidates_list_file, 'wb')
    fout1 = open(dst_entity_candidate_clique_file, 'wb')
    fout2 = open(dst_doc_entity_indices_file, 'wb') if dst_doc_entity_indices_file else None
    for line_cnt, line in enumerate(fin):
        words = line.strip().split(' ')

        cliques, indices_list = entity_candidate_cliques_in_words(words, illegal_start_words)
        # print indices_list
        if fout2:
            for i, indices in enumerate(indices_list):
                for j, idx in enumerate(indices):
                    if i > 0 and j == 0:
                        fout2.write('\t')
                    if j > 0:
                        fout2.write(' ')
                    fout2.write(str(idx))
            fout2.write('\n')

        for i, clique in enumerate(cliques):
            for j, entity_name in enumerate(clique):
                if i > 0 or j > 0:
                    fout0.write('\t')
                fout0.write(entity_name)
            if len(clique) > 1:
                for j, entity_name in enumerate(clique):
                    if j != 0:
                        fout1.write('\t')
                    fout1.write(entity_name)
                fout1.write('\n')
        fout0.write('\n')

        if line_cnt % 10000 == 10000 - 1:
            print line_cnt + 1
        # if line_cnt == 2:
        #     break
    fin.close()
    fout0.close()
    fout1.close()
    if fout2:
        fout2.close()
    print line_cnt + 1, 'lines total'


def gen_entity_name_dict_from_candidates(doc_entity_candidates_file, lc_word_cnts_file, wc_word_cnts_file,
                                         dst_file_name, min_occurance=2):
    name_cnts = dict()
    fin = open(doc_entity_candidates_file, 'rb')
    line_cnt = 0
    for line_cnt, line in enumerate(fin):
        entity_names = line.strip().split('\t')
        doc_entity_name_set = set()
        for entity_name in entity_names:
            if len(entity_name) > 0:
                doc_entity_name_set.add(entity_name)
        for entity_name in doc_entity_name_set:
            cnt = name_cnts.get(entity_name, 0)
            name_cnts[entity_name] = cnt + 1

        if line_cnt % 100000 == 100000 - 1:
            print line_cnt + 1
    fin.close()
    print line_cnt, 'lines total'

    lc_word_cnts = text_process_common.load_word_cnts(lc_word_cnts_file)
    wc_word_cnts = text_process_common.load_word_cnts(wc_word_cnts_file)

    print 'filtering', len(name_cnts), 'names.'
    fout = open(dst_file_name, 'wb')
    for entity_name, cnt in name_cnts.items():
        if cnt < min_occurance:
            continue
        if ' ' in entity_name:
            fout.write('%s\t%d\n' % (entity_name, cnt))
            continue
        lc_word_cnt = lc_word_cnts.get(entity_name.lower(), 0)
        wc_word_cnt = wc_word_cnts.get(entity_name, 0)
        if wc_word_cnt > lc_word_cnt - wc_word_cnt:
            fout.write('%s\t%d\n' % (entity_name, cnt))
    fout.close()


def __read_ner_results(fin):
    try:
        mentions = list()
        num_mentions = int(fin.next()[:-1])
        for i in xrange(num_mentions):
            vals = fin.next()[:-1].split('\t')
            mentions.append((vals[0], int(vals[1]), int(vals[2])))
        return True, mentions
    except StopIteration:
        return False, []
    # except:
    #     print vals


def load_entity_dict(dict_file_name):
    entity_dict = dict()
    fin = open(dict_file_name, 'rb')
    for idx, line in enumerate(fin):
        vals = line.split('\t')
        entity_dict[vals[0]] = idx
    fin.close()
    return entity_dict


def ner_result_to_tab_sep(ner_result_file, dst_file):
    fin = open(ner_result_file, 'rb')
    fout = open(dst_file, 'wb')
    cnt = 0
    valid, mentions = __read_ner_results(fin)
    while valid:
        for i, mention in enumerate(mentions):
            if i > 0:
                fout.write('\t')
            fout.write(mention[0])
        fout.write('\n')

        valid, mentions = __read_ner_results(fin)
        cnt += 1
    fin.close()
    fout.close()
    print cnt


def gen_doc_entity_pairs(entity_dict_file_name, raw_doc_entity_file_name, dst_file_bin, dst_file_text=None):
    print 'gen', dst_file_bin
    entity_dict = load_entity_dict(entity_dict_file_name)

    fin = open(raw_doc_entity_file_name, 'rb')
    fout0 = open(dst_file_bin, 'wb')
    fout1 = open(dst_file_text, 'wb') if dst_file_text else None
    np.zeros(2, np.int32).tofile(fout0)
    doc_cnt = 0
    for doc_cnt, line in enumerate(fin):
        vals = line.split('\t')
        tmp_doc_entity_dict = dict()
        tmp_doc_entity_list = list()
        for val in vals:
            idx = entity_dict.get(val, -1)
            if idx != -1:
                cnt = tmp_doc_entity_dict.get(idx, 0)
                if cnt > 65535:
                    print 'entity cnt larger than 65535'
                    cnt = 65535
                if cnt == 0:
                    tmp_doc_entity_list.append(val)
                tmp_doc_entity_dict[idx] = cnt + 1
                # tmp_entity_list.append(idx)
        np.asarray([len(tmp_doc_entity_dict)], np.int32).tofile(fout0)
        np.asarray(tmp_doc_entity_dict.keys(), np.int32).tofile(fout0)
        np.asarray(tmp_doc_entity_dict.values(), np.uint16).tofile(fout0)
        # break
        if fout1:
            if len(tmp_doc_entity_list) > 0:
                fout1.write(tmp_doc_entity_list[0])
            for entity_name in tmp_doc_entity_list[1:]:
                fout1.write('\t' + entity_name)
            fout1.write('\n')

        if doc_cnt % 10000 == 10000 - 1:
            print doc_cnt + 1
    fin.close()

    if fout1:
        fout1.close()

    doc_cnt += 1
    print doc_cnt, 'docs'

    fout0.seek(0)
    np.asarray([doc_cnt, len(entity_dict)], np.int32).tofile(fout0)
    fout0.close()


def get_edge_list_from_clique(entity_list):
    edge_list = list()
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            va = entity_list[i]
            vb = entity_list[j]
            if va > vb:
                va = entity_list[j]
                vb = entity_list[i]
            edge_list.append(array('i', [va, vb]))
    return edge_list


###############################################################
# get entity edges from entity cliques

def get_entity_edges(entity_dict, raw_entity_clique_file_name):
    entity_edge_list = list()
    fin = open(raw_entity_clique_file_name, 'rb')
    for line in fin:
        vals = line.strip().split('\t')
        entity_list = list()
        for entity_name in vals:
            entity_idx = entity_dict.get(entity_name, -1)
            # if entity_idx != -1:
            #     print(entity_name + '\t' + str(entity_idx))
            if (entity_idx != -1) and (entity_idx not in entity_list):
                entity_list.append(entity_idx)

        tmp_edge_list = get_edge_list_from_clique(entity_list)
        entity_edge_list += tmp_edge_list
    fin.close()

    return entity_edge_list


def to_weighted_edges(edge_list):
    edge_list.sort()
    pre_edge = None
    cur_weight_edge = None
    weight_edge_list = list()
    for edge in edge_list:
        if pre_edge and edge == pre_edge:
            cur_weight_edge[2] += 1
        else:
            cur_weight_edge = array('i', [edge[0], edge[1], 1])
            weight_edge_list.append(cur_weight_edge)
        pre_edge = edge
    return weight_edge_list


def gen_entity_edge_list_from_cliques(entity_dict_file_name, raw_entity_cliques_file_name,
                                      dst_weighted_edge_list_file_name):
    entity_dict = load_entity_dict(entity_dict_file_name)
    entity_edges = get_entity_edges(entity_dict, raw_entity_cliques_file_name)
    weighted_entity_edges = to_weighted_edges(entity_edges)

    fout = open(dst_weighted_edge_list_file_name, 'wb')
    fout.write('%d\t%d\t%d\n' % (len(entity_dict), len(entity_dict), len(weighted_entity_edges)))
    for edge in weighted_entity_edges:
        fout.write('%d\t%d\t%d\n' % (edge[0], edge[1], edge[2]))
    fout.close()


def gen_entity_entity_pairs(entity_dict_file_name, entity_candidate_cliques_file,
                            dst_file_name):
    print 'gen', dst_file_name
    entity_dict = load_entity_dict(entity_dict_file_name)
    entity_adj_lists = [list() for _ in xrange(len(entity_dict))]
    fin = open(entity_candidate_cliques_file, 'rb')
    for line_idx, line in enumerate(fin):
        entity_names = line.strip().split('\t')
        idx_list = list()
        for entity_name in entity_names:
            entity_name = entity_name.strip()
            if not entity_name:
                continue
            entity_idx = entity_dict.get(entity_name, -1)
            if entity_idx == -1:
                continue

            if entity_idx not in idx_list:
                idx_list.append(entity_idx)

        for i in xrange(len(idx_list)):
            for j in xrange(i + 1, len(idx_list)):
                entity_adj_lists[idx_list[i]].append(idx_list[j])
                entity_adj_lists[idx_list[j]].append(idx_list[i])

        if line_idx % 1000000 == 1000000 - 1:
            print line_idx + 1
        # if line_idx == 1000:
        #     break
    fin.close()

    print len(entity_dict), 'entities to write'
    fout = open(dst_file_name, 'wb')
    np.asarray([len(entity_dict), len(entity_dict)], np.int32).tofile(fout)
    for i, adj_list in enumerate(entity_adj_lists):
        indices_list = list()
        cnts_list = list()
        pre_idx = None
        cur_cnt = 0
        adj_list.sort()
        # print adj_list
        for idx in adj_list:
            if idx == pre_idx:
                cur_cnt += 1
            else:
                if pre_idx:
                    indices_list.append(pre_idx)
                    if cur_cnt > 65535:
                        print 'too large', cur_cnt
                        cur_cnt = 65535
                    cnts_list.append(cur_cnt)
                cur_cnt = 1
            pre_idx = idx

        if pre_idx:
            indices_list.append(pre_idx)
            cnts_list.append(cur_cnt)
        np.asarray([len(indices_list)], np.int32).tofile(fout)
        np.asarray(indices_list, np.int32).tofile(fout)
        np.asarray(cnts_list, np.uint16).tofile(fout)

        if i % 10000 == 10000 - 1:
            print i + 1
    fout.close()


def gen_cnts_file(adj_list_file, dst_cnts_file):
    fin = open(adj_list_file, 'rb')
    num_left, num_right = np.fromfile(fin, np.int32, 2)
    cnts = np.zeros(num_right, np.int32)
    for i in xrange(num_left):
        num = np.fromfile(fin, np.int32, 1)
        indices = np.fromfile(fin, np.int32, num)
        cur_cnts = np.fromfile(fin, np.uint16, num)
        for idx, cnt in izip(indices, cur_cnts):
            cnts[idx] += cnt
    fin.close()

    fout = open(dst_cnts_file, 'wb')
    np.asarray([num_right], np.int32).tofile(fout)
    cnts.tofile(fout)
    fout.close()

sentence_end_tokens = '.?!;'


def __has_sentence_end(words):
    for word in words:
        if word in sentence_end_tokens:
            return True
    return False


def __write_mention_clique(mention_clique, fout):
    for i, m in enumerate(mention_clique):
        if i > 0:
            fout.write('\t')
        fout.write(m)


def gen_ee_pairs_with_ner_result(docs_file, ner_result_file, mention_cliques_file):
    fin0 = open(docs_file, 'rb')
    fin1 = open(ner_result_file, 'rb')
    fout = open(mention_cliques_file, 'wb')
    sentence_cnt = 0
    for i, line in enumerate(fin0):
        # words = word_tokenize(line)
        # print words
        line = line.decode('utf-8')
        valid, mentions = __read_ner_results(fin1)
        cur_mention_clique = set()
        prev_pos = 0
        for mention in mentions:
            if line[mention[1]:mention[2]] != mention[0].decode('utf-8'):
                print mention
            words = word_tokenize(line[prev_pos:mention[1]])
            if __has_sentence_end(words) and cur_mention_clique:
                if len(cur_mention_clique) > 1:
                    __write_mention_clique(cur_mention_clique, fout)
                    fout.write('\n')
                cur_mention_clique = set()
                cur_mention_clique.add(mention[0])
            else:
                cur_mention_clique.add(mention[0])
            prev_pos = mention[2]
        # break
        if (i + 1) % 1000 == 0:
            print i + 1
    fout.close()
    fin0.close()
    fin1.close()


def gen_entity_name_dict(ner_result_file, dst_file, filter_loc=False):
    name_dict = dict()
    fin = open(ner_result_file, 'rb')
    for line in fin:
        num_mentions = int(line[:-1])
        names_in_doc = set()
        for i in xrange(num_mentions):
            line = fin.next()
            vals = line[:-1].split('\t')
            if filter_loc and vals[3] == 'LOCATION':
                continue
            names_in_doc.add(vals[0])
        for name in names_in_doc:
            cnt = name_dict.get(name, 0)
            name_dict[name] = cnt + 1
    fin.close()

    fout = open(dst_file, 'wb')
    for name, cnt in name_dict.iteritems():
        if cnt > 1 and len(name) < 50:
            fout.write('%s\t%d\n' % (name, cnt))
    fout.close()


###################################################################
# the jobs

def job_gen_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities.bin'
    gen_doc_entity_pairs(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def job_gen_entity_edge_list_from_cliques():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_entity_clique_file_name = 'e:/dc/20ng_bydate/entity_cliques_raw.txt'
    dst_weighted_edge_list_file_name = 'e:/dc/20ng_bydate/weighted_entity_edge_list.txt'
    gen_entity_edge_list_from_cliques(entity_dict_file_name, raw_entity_clique_file_name,
                                      dst_weighted_edge_list_file_name)


def job_init_entity_net_wiki():
    line_docs_file_name = 'e:/dc/el/wiki/wiki_lines_tokenized.txt'
    illegal_start_words_file = 'e:/dc/20ng_bydate/stopwords.txt'
    dst_doc_entity_candidates_list_file = 'e:/dc/el/wiki/doc_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/dc/el/wiki/entity_candidate_cliques.txt'
    init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                    dst_entity_candidate_clique_file)

    lc_word_cnts_file_name = 'e:/dc/el/wiki//wiki_word_cnts_lc.txt'
    wc_word_cnts_file_name = 'e:/dc/el/wiki/wiki_word_cnts_with_case.txt'
    dst_entity_name_list_file = 'e:/dc/el/wiki/entity_names.txt'
    gen_entity_name_dict_from_candidates(dst_doc_entity_candidates_list_file, lc_word_cnts_file_name,
                                         wc_word_cnts_file_name, dst_entity_name_list_file)


def gen_entity_net_20ng():
    proper_entity_dict_file = 'e:/dc/20ng_bydate/entity_names.txt'
    doc_entity_candidates_file = 'e:/dc/20ng_bydate/doc_entity_candidates.txt'
    dst_doc_entity_list_file = 'e:/dc/20ng_bydate/doc_entities_short.bin'
    # gen_doc_entity_list(proper_entity_dict_file, doc_entity_candidates_file, dst_doc_entity_list_file)

    dst_entity_cnts_file = 'e:/dc/20ng_bydate/entity_cnts.bin'
    text_process_common.gen_word_cnts_file_from_bow_file(dst_doc_entity_list_file, dst_entity_cnts_file)

    entity_candidate_cliques_file = 'e:/dc/20ng_bydate/entity_candidate_cliques.txt'
    dst_entity_net_adj_list_file = 'e:/dc/20ng_bydate/entity_net_adj_list.bin'
    # gen_entity_net_adj_list(proper_entity_dict_file, entity_candidate_cliques_file, dst_entity_net_adj_list_file)


def gen_entity_net_wiki():
    proper_entity_dict_file = 'e:/dc/el/wiki/entity_names.txt'
    doc_entity_candidates_file = 'e:/dc/el/wiki/doc_entity_candidates.txt'
    dst_doc_entity_list_file = 'e:/dc/el/wiki/wiki_entities.bin'

    gen_doc_entity_pairs(proper_entity_dict_file, doc_entity_candidates_file, dst_doc_entity_list_file)
    dst_entity_cnts_file = 'e:/dc/el/wiki/entity_cnts.bin'
    text_process_common.gen_word_cnts_file_from_bow_file(dst_doc_entity_list_file, dst_entity_cnts_file)

    entity_candidate_cliques_file = 'e:/dc/el/wiki/entity_candidate_cliques.txt'
    dst_entity_net_adj_list_file = 'e:/dc/el/wiki/entity_net_adj_list.bin'
    gen_entity_entity_pairs(proper_entity_dict_file, entity_candidate_cliques_file, dst_entity_net_adj_list_file)


def gen_entity_net_tac14():
    line_docs_file_name = 'e:/dc/el/tac/tac_2014_train_docs_text_tokenized.txt'
    dst_doc_entity_candidates_list_file = 'e:/dc/el/tac/tac14_train_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/dc/el/tac/tac14_train_entity_candidate_cliques.txt'
    dst_doc_entity_list_file = 'e:/dc/el/tac/tac14_train_entities.bin'

    line_docs_file_name = 'e:/dc/el/tac/tac_2014_eval_docs_text_tokenized.txt'
    dst_doc_entity_candidates_list_file = 'e:/dc/el/tac/tac14_eval_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/dc/el/tac/tac14_eval_entity_candidate_cliques.txt'
    dst_doc_entity_list_file = 'e:/dc/el/tac/tac14_eval_entities.bin'

    illegal_start_words_file = 'e:/dc/20ng_bydate/stopwords.txt'
    init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                    dst_entity_candidate_clique_file)

    proper_entity_dict_file = 'e:/dc/el/wiki/entity_names.txt'
    gen_doc_entity_pairs(proper_entity_dict_file, dst_doc_entity_candidates_list_file, dst_doc_entity_list_file)


def gen_entity_net_tac():
    year = '2009'
    part = 'eval'
    file_tag = year + '_' + part
    line_docs_file_name = 'e:/dc/el/tac/tac_' + file_tag + '_docs_text_tokenized.txt'
    dst_doc_entity_candidates_list_file = 'e:/dc/el/tac/tac_' + file_tag + '_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/dc/el/tac/tac_' + file_tag + '_entity_candidate_cliques.txt'
    dst_doc_entity_list_file_bin = 'e:/dc/el/tac/tac_' + file_tag + '_entities.bin'
    dst_doc_entity_list_file_text = 'e:/dc/el/tac/tac_' + file_tag + '_entities.txt'

    illegal_start_words_file = 'e:/dc/20ng_bydate/stopwords.txt'
    init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                    dst_entity_candidate_clique_file)

    proper_entity_dict_file = 'e:/dc/el/wiki/entity_names.txt'
    gen_doc_entity_pairs(proper_entity_dict_file, dst_doc_entity_candidates_list_file, dst_doc_entity_list_file_bin,
                         dst_doc_entity_list_file_text)


def main():
    # job_gen_20ng_doc_entity_list()
    # job_gen_entity_edge_list_from_cliques()
    # gen_entity_net_20ng()

    # job_init_entity_net_wiki()
    gen_entity_net_wiki()

    # gen_entity_net_tac14()
    # gen_entity_net_tac()


def test():
    print 'test'


if __name__ == '__main__':
    start_time = time()
    # test()
    main()
    print 'Elapsed time:', time() - start_time

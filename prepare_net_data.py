import numpy
from time import time
from array import array

import text_process_common


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

            for i in xrange(beg_idx, idx + 1):
                cur_name += ' ' + words[i]
            if 2 < len(cur_name) < 50:
                cur_clique_entites.append(cur_name)
        idx += 1

    return cliques


def init_entity_net(tokenized_line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                    dst_entity_candidate_clique_file):
    illegal_start_words = text_process_common.load_word_set(illegal_start_words_file)

    line_cnt = 0
    fin = open(tokenized_line_docs_file_name, 'rb')
    fout0 = open(dst_doc_entity_candidates_list_file, 'wb')
    fout1 = open(dst_entity_candidate_clique_file, 'wb')
    for line_cnt, line in enumerate(fin):
        words = line.strip().split(' ')

        cliques = entity_candidate_cliques_in_words(words, illegal_start_words)

        is_first_doc_entity = True
        for clique in cliques:
            for i, entity_name in enumerate(clique):
                if is_first_doc_entity:
                    is_first_doc_entity = False
                else:
                    fout0.write('\t')
                fout0.write(entity_name)
            if len(clique) > 1:
                for i, entity_name in enumerate(clique):
                    if i != 0:
                        fout1.write('\t')
                    fout1.write(entity_name)
                fout1.write('\n')
        fout0.write('\n')

        if line_cnt % 10000 == 10000 - 1:
            print line_cnt + 1
        # if line_cnt == 100:
        #     break
    fin.close()
    fout0.close()
    fout1.close()
    print line_cnt + 1, 'lines total'


def gen_entity_name_dict(doc_entity_candidates_file, lc_word_cnts_file, wc_word_cnts_file, dst_file_name):
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
        if cnt < 4:
            continue
        if ' ' in entity_name:
            fout.write('%s\t%d\n' % (entity_name, cnt))
            continue
        lc_word_cnt = lc_word_cnts.get(entity_name.lower(), 0)
        wc_word_cnt = wc_word_cnts.get(entity_name, 0)
        if wc_word_cnt > lc_word_cnt - wc_word_cnt:
            fout.write('%s\t%d\n' % (entity_name, cnt))
    fout.close()


def load_entity_dict(dict_file_name):
    entity_dict = dict()
    fin = open(dict_file_name, 'rb')
    for idx, line in enumerate(fin):
        vals = line.split('\t')
        entity_dict[vals[0]] = idx
    fin.close()
    return entity_dict


def gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name,
                             dst_file_name):
    entity_dict = load_entity_dict(entity_dict_file_name)
    entity_list_docs = list()
    entity_cnt_list_docs = list()
    fin = open(raw_doc_entity_file_name, 'rb')
    for line in fin:
        vals = line.split('\t')
        # tmp_entity_list = list()
        tmp_doc_entity_dict = dict()
        for val in vals:
            idx = entity_dict.get(val, -1)
            if idx != -1:
                cnt = tmp_doc_entity_dict.get(idx, 0)
                tmp_doc_entity_dict[idx] = cnt + 1
                # tmp_entity_list.append(idx)
        entity_list_docs.append(numpy.array(tmp_doc_entity_dict.keys(), dtype=numpy.int32))
        entity_cnt_list_docs.append(numpy.array(tmp_doc_entity_dict.values(), dtype=numpy.int32))
        # break
    fin.close()

    fout = open(dst_file_name, 'wb')
    numpy.array([len(entity_list_docs)], dtype=numpy.int32).tofile(fout)
    numpy.array([len(entity_dict)], dtype=numpy.int32).tofile(fout)
    for entity_list, entity_cnt_list in zip(entity_list_docs, entity_cnt_list_docs):
        numpy.array([len(entity_list)], dtype=numpy.int32).tofile(fout)
        entity_list.tofile(fout)
        entity_cnt_list.tofile(fout)
        # numpy.ones(len(entity_list), dtype=numpy.int32).tofile(fout)
    fout.close()


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


###################################################################
# the jobs

def job_gen_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def job_gen_entity_edge_list_from_cliques():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_entity_clique_file_name = 'e:/dc/20ng_bydate/entity_cliques_raw.txt'
    dst_weighted_edge_list_file_name = 'e:/dc/20ng_bydate/weighted_entity_edge_list.txt'
    gen_entity_edge_list_from_cliques(entity_dict_file_name, raw_entity_clique_file_name,
                                      dst_weighted_edge_list_file_name)


def job_init_entity_net_wiki():
    line_docs_file_name = 'e:/dc/el/wiki_lines_tokenized.txt'
    illegal_start_words_file = 'e:/dc/20ng_bydate/stopwords.txt'
    dst_doc_entity_candidates_list_file = 'e:/dc/el/doc_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/dc/el/entity_candidate_cliques.txt'
    # init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
    #                 dst_entity_candidate_clique_file)

    lc_word_cnts_file_name = 'e:/dc/el/wiki_word_cnts_lc.txt'
    wc_word_cnts_file_name = 'e:/dc/el/wiki_word_cnts_with_case.txt'
    dst_entity_name_list_file = 'e:/dc/el/entity_names.txt'
    gen_entity_name_dict(dst_doc_entity_candidates_list_file, lc_word_cnts_file_name, wc_word_cnts_file_name,
                         dst_entity_name_list_file)


def main():
    # job_gen_20ng_doc_entity_list()
    # job_gen_train_20ng_doc_entity_list()
    # job_gen_test_20ng_doc_entity_list()
    # job_gen_entity_edge_list_from_cliques()
    job_init_entity_net_wiki()


def test():
    f0 = open('e:/dc/20ng_bydate/weighted_entity_edge_list.txt', 'rb')
    f1 = open('e:/dc/20ng_bydate/weighted_entity_edge_list_tmp.txt', 'rb')
    for idx, (line0, line1) in enumerate(zip(f0, f1)):
        if line0 != line1:
            print idx, 'not equal'
    f0.close()
    f1.close()

if __name__ == '__main__':
    start_time = time()
    # test()
    main()
    print 'Elapsed time:', time() - start_time

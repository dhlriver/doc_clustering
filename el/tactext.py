import re
import os

import textutils
import dataarange


def __is_full_name(acronym, full_name):
    fpos = 0
    for ch in acronym:
        if not ch.isalpha():
            return False
            # continue

        flg = False
        while fpos < len(full_name) and (not flg):
            if full_name[fpos].isupper():
                if full_name[fpos] == ch:
                    flg = True
                else:
                    break
            fpos += 1
        if not flg:
            return False

    while fpos < len(full_name):
        if full_name[fpos].isupper():
            return False
        fpos += 1

    return True


def __get_doc_id_from_path(docpath):
    beg_pos = docpath.rfind('\\')
    tmp = docpath.rfind('/')
    beg_pos = beg_pos if beg_pos > tmp else tmp
    if docpath.endswith('.nw.xml') or docpath.endswith('.df.xml'):
        return docpath[beg_pos + 1:-7]
    return docpath[beg_pos + 1:-4]


def load_doc_entity_names(doc_list_file, doc_entity_name_file):
    doc_list = list()
    fin = open(doc_list_file, 'rb')
    for line in fin:
        line = line.strip()
        doc_list.append(__get_doc_id_from_path(line))
    fin.close()

    entity_names_dict = dict()
    fin = open(doc_entity_name_file, 'rb')
    for idx, line in enumerate(fin):
        line = line.strip()
        entity_names = line.split('\t')
        entity_names_dict[doc_list[idx]] = entity_names
    fin.close()

    return entity_names_dict


def __acronym_expansion(query_file, doc_list_file, doc_entity_name_file, dst_query_file):
    entity_names_dict = load_doc_entity_names(doc_list_file, doc_entity_name_file)

    fin = open(query_file, 'rb')
    queries_text = fin.read()
    fin.close()

    fout = open(dst_query_file, 'wb')

    exp_cnt = 0
    ps = r'<query id="(.*?)">\s*<name>(.*?)</name>\s*<docid>(.*?)</docid>'
    miter = re.finditer(ps, queries_text)
    for m in miter:
        query_id = m.group(1)
        query_name = m.group(2)
        doc_id = m.group(3)
        if query_name.isupper():
            print query_id, doc_id, query_name
            candidates = entity_names_dict[doc_id]
            for entity_name in candidates:
                if __is_full_name(query_name, entity_name):
                    if len(entity_name) > len(query_name):
                        print '%s\t%s\t%s#%s*' % (query_id, doc_id, query_name, entity_name),
                        query_name = entity_name
                        exp_cnt += 1
                        print '*###',
                        print

        fout.write('  <query id="%s">\n    <name>%s</name>\n    <docid>%s</docid>\n  </query>\n'
                   % (query_id, query_name, doc_id))
    fout.close()

    print exp_cnt


def __load_ner_result(filename):
    doc_entities = dict()
    f = open(filename, 'r')
    for line in f:
        vals = line.strip().split('\t')
        docid = vals[0]
        num_entities = int(vals[1])
        entities = list()
        for i in xrange(num_entities):
            line = f.next()
            vals = line.strip().split('\t')
            entities.append((vals[0], vals[1]))
        doc_entities[docid] = entities
    f.close()
    return doc_entities


def __expand_acronym(query_name, entities_in_doc):
    new_name = query_name
    for (name, entity_type) in entities_in_doc:
        print name, entity_type
        if __is_full_name(query_name, name) and len(entity_name) > len(query_name):
    # if len(entity_name) > len(query_name):
    #                 print '%s\t%s\t%s#%s*' % (query_id, doc_id, query_name, entity_name),
    #                 query_name = entity_name
    #                 exp_cnt += 1
    #                 print '*###',
    #                 print
    return new_name


def __name_expansion(query_file, doc_list_file, doc_ner_file, dst_query_file):
    doc_entities = __load_ner_result(doc_ner_file)
    # print doc_entities['APW_ENG_20080121.1013.LDC2009T13']
    fin = open(query_file, 'rb')
    queries_text = fin.read()
    fin.close()

    fout = open(dst_query_file, 'wb')

    exp_cnt = 0
    ps = r'<query id="(.*?)">\s*<name>(.*?)</name>\s*<docid>(.*?)</docid>'
    miter = re.finditer(ps, queries_text)
    for m in miter:
        query_id = m.group(1)
        query_name = m.group(2)
        doc_id = m.group(3)
        print '%s\t%s\t%s' % (query_id, query_name, doc_id)
        cur_doc_entities = doc_entities[doc_id]
        if query_name.isupper():
            print query_name
            query_name = __expand_acronym(query_name, cur_doc_entities)
            break
        #     print query_id, doc_id, query_name
        #     candidates = entity_names_dict[doc_id]
        #     for entity_name in candidates:
        #         if __is_full_name(query_name, entity_name):
        #             if len(entity_name) > len(query_name):
        #                 print '%s\t%s\t%s#%s*' % (query_id, doc_id, query_name, entity_name),
        #                 query_name = entity_name
        #                 exp_cnt += 1
        #                 print '*###',
        #                 print
        #
        # fout.write('  <query id="%s">\n    <name>%s</name>\n    <docid>%s</docid>\n  </query>\n'
        #            % (query_id, query_name, doc_id))
    fout.close()

    print exp_cnt


def __gen_line_docs_file_tac(doc_list_file, dst_line_docs_file):
    f = open(doc_list_file, 'r')
    fout = open(dst_line_docs_file, 'wb')
    for line in f:
        line = line.strip()
        print line
        line_text = textutils.doc_to_line(line)
        fout.write(line_text)
        fout.write('\n')
    f.close()
    fout.close()
    # fout0 = open(dst_line_docs_file, 'wb')
    # fout1 = open(dst_doc_list_file, 'wb')
    # for f in os.listdir(docs_dir):
    #     file_path = os.path.join(docs_dir, f)
    #     if not os.path.isfile(file_path):
    #         continue
    #
    #     fout1.write(f)
    #     fout1.write('\n')
    #
    #     line_text = textutils.doc_to_line(file_path)
    #     fout0.write(line_text)
    #     fout0.write('\n')
    # fout0.close()
    # fout1.close()


def gen_doc_mention_names():
    # query_file = r'D:\data\el\LDC2015E19\data\2010\eval\tac_kbp_2010' \
    #              r'_english_entity_linking_evaluation_queries.xml'
    # docs_list_file = 'e:/dc/el/tac/tac_2010_eval_docs_list.txt'
    # doc_entity_names_file = 'e:/dc/el/tac/tac_2010_eval_entities.txt'
    # dst_file = 'e:/dc/el/tac/tac_2010_eval_doc_queries.txt'

    query_file = r'D:\data\el\LDC2015E19\data\2009\eval\tac_kbp_2009' \
                 r'_english_entity_linking_evaluation_queries.xml'
    docs_list_file = 'e:/dc/el/tac/tac_2009_eval_docs_list.txt'
    doc_entity_names_file = 'e:/dc/el/tac/tac_2009_eval_entities.txt'
    dst_file = 'e:/dc/el/tac/tac_2009_eval_doc_queries.txt'

    doc_entity_names = list()
    fin = open(doc_entity_names_file, 'rb')
    for line in fin:
        names = line.strip().split('\t')
        doc_entity_names.append(names)
    fin.close()

    fin = open(query_file, 'rb')
    queries_text = fin.read()
    fin.close()

    ps = r'<query id="(.*?)">\s*<name>(.*?)</name>\s*<docid>(.*?)</docid>'
    m_iters = re.finditer(ps, queries_text)
    cnt = 0
    doc_queries = dict()
    for m in m_iters:
        cnt += 1
        cur_doc_queries = doc_queries.get(m.group(3), list())
        cur_doc_queries.append((m.group(1), m.group(2)))
        doc_queries[m.group(3)] = cur_doc_queries

    fout = open(dst_file, 'wb')
    fin = open(docs_list_file, 'rb')
    for doc_idx, line in enumerate(fin):
        doc_name = line.strip()
        if doc_name.endswith('.xml'):
            doc_name = doc_name[:-4]
        cur_doc_queries = doc_queries[doc_name]
        cur_doc_entity_names = doc_entity_names[doc_idx]
        hit = False
        for qid, name in cur_doc_queries:
            candidates = list()
            for entity_name in cur_doc_entity_names:
                if (' ' not in name) and len(entity_name) > len(name) and entity_name.find(name) > 0:
                    candidates.append(entity_name)
            if len(candidates) > 0:
                hit = True
                fout.write('%s\t%s' % (qid, name))
                for candidate in candidates:
                    fout.write('\t' + candidate)
        if hit:
            fout.write('\n')
        # print '\n'
    fin.close()
    fout.close()


def process_docs_for_ner():
    # cur_docs_file = 'e:/dc/el/tac/2010/train/docs.txt'
    # dst_docs_file = 'e:/dc/el/tac/2010/train/docs-ner.txt'
    # cur_docs_file = 'e:/dc/el/tac/2010/eval/docs.txt'
    # dst_docs_file = 'e:/dc/el/tac/2010/eval/docs-ner.txt'
    cur_docs_file = 'e:/dc/el/tac/2009/eval/docs.txt'
    dst_docs_file = 'e:/dc/el/tac/2009/eval/docs-ner.txt'

    sub0 = '[^\s]*-[^\s]*-[^\s]*'
    sub1 = '[^\s]*[<>][^\s]*'
    sub2 = '&lt;[^\s]*&gt;'
    sub3 = 'https?:[^\s]*'
    sub4 = '[^\s]*@[^\s]*\.[^\s]*'
    sub5 = '&[^\s]*;'
    sub6 = '==+|__+|\*\*+'
    sub7 = '[^\s]+\.[^\s]+\.[^\s]+'

    fin = open(cur_docs_file, 'rb')
    fout = open(dst_docs_file, 'wb')
    for line in fin:
        line = line.replace('USENET TEXT', '')
        line = line.replace('NEWS STORY', '')
        line = re.sub('%s|%s|%s|%s|%s|%s|%s|%s' % (sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7), '', line)
        line = re.sub('\s+', ' ', line)
        fout.write(line)
        fout.write('\n')
    fin.close()
    fout.close()


def __job_init_entity_net():
    line_docs_file_name = 'e:/data/emadr/el/tac/2011/eval/docs-tokenized.txt'
    illegal_start_words_file = 'e:/data/emadr/20ng_bydate/stopwords.txt'
    dst_doc_entity_candidates_list_file = 'e:/data/emadr/el/tac/2011/eval/doc_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/data/emadr/el/tac/2011/eval/entity_candidate_cliques.txt'
    dataarange.init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                               dst_entity_candidate_clique_file)


def __setup_doc_entities_file():
    line_docs_file_name = 'e:/data/emadr/el/tac/2011/eval/docs-tokenized.txt'
    illegal_start_words_file = 'e:/data/emadr/20ng_bydate/stopwords.txt'
    doc_entity_candidates_list_file = 'e:/data/emadr/el/tac/2011/eval/doc_entity_candidates.txt'
    entity_candidate_clique_file = 'e:/data/emadr/el/tac/2011/eval/entity_candidate_cliques.txt'
    # dataarange.init_entity_net(line_docs_file_name, illegal_start_words_file, doc_entity_candidates_list_file,
    #                            entity_candidate_clique_file)

    proper_entity_dict_file = 'e:/data/emadr/el/wiki/entity_names.txt'
    # doc_entity_candidates_file = 'e:/data/emadr/el/wiki/doc_entity_candidates.txt'
    dst_doc_entity_list_file = 'e:/data/emadr/el/tac/2011/eval/de.bin'

    dataarange.gen_doc_entity_pairs(proper_entity_dict_file, doc_entity_candidates_list_file, dst_doc_entity_list_file)
    dst_entity_cnts_file = 'e:/data/emadr/el/tac/2011/eval/entity_cnts.bin'
    textutils.gen_word_cnts_file_from_bow_file(dst_doc_entity_list_file, dst_entity_cnts_file)


def __gen_tac_docs():
    # docs_dir = r'D:\data\el\LDC2015E19\data\2010\training\source_documents'
    docs_dir = r'D:\data\el\LDC2015E19\data\2010\eval\source_documents'
    # docs_dir = r'D:\data\el\LDC2015E19\data\2009\eval\source_documents'
    year = 2010
    part = 'eval'

    doc_list_file = 'e:/data/el/LDC2015E19/data/%d/%s/data/eng-docs-list-win.txt' % (year, part)
    line_docs_file = 'e:/data/emadr/el/tac/%d/%s/docs.txt' % (year, part)
    # __gen_line_docs_file_tac(doc_list_file, line_docs_file)

    tokenized_line_docs_file = 'e:/data/emadr/el/tac/%d/%s/docs-tokenized.txt' % (year, part)
    proper_word_cnts_dict_file = 'e:/data/emadr/el/wiki/words_dict_proper.txt'
    max_word_len = 20
    tokenized_line_docs_lc_file = 'e:/data/emadr/el/tac/%d/%s/docs-tokenized-lc.txt' % (year, part)
    bow_docs_file = 'e:/data/emadr/el/tac/%d/%s/dw.bin' % (year, part)

    textutils.gen_lowercase_token_file(tokenized_line_docs_file, proper_word_cnts_dict_file,
                                       max_word_len, 1, tokenized_line_docs_lc_file)
    min_occurrence = 2
    words_dict = textutils.load_words_to_idx_dict(proper_word_cnts_dict_file, min_occurrence)
    textutils.line_docs_to_bow(tokenized_line_docs_lc_file, words_dict, min_occurrence, bow_docs_file)

    # dst_word_cnts_file = 'e:/data/emadr/el/tac/%d/%s/word_cnts.bin' % (year, part)
    # textutils.gen_word_cnts_file_from_bow_file(bow_docs_file, dst_word_cnts_file)


def __job_acronym_expansion():
    # query_file = r'e:\data\el\LDC2015E19\data\2009\eval\tac_kbp_2009' \
    #              r'_english_entity_linking_evaluation_queries.xml'
    # doc_list_file = 'e:/data/el/LDC2015E19/data/2009/eval/data/eng-docs-list-win.txt'
    # doc_entity_name_file = 'e:/data/emadr/el/tac/2009/tac_2009_eval_entities.txt'
    # dst_query_file = r'e:/data/el/LDC2015E19/data/2009/eval/data/queries_expanded.xml'

    # query_file = r'e:/dc/el/tac/2010/eval/queries.xml'
    # doc_list_file = 'e:/dc/el/tac/2010/eval/docs_list.txt'
    # doc_entity_name_file = 'e:/dc/el/tac/2010/eval/tac_2010_eval_entities.txt'
    # dst_query_file = r'e:/dc/el/tac/2010/eval/queries-expanded.xml'

    query_file = r'e:\data\el\LDC2015E19\data\2011\eval\tac' \
                 r'_kbp_2011_english_entity_linking_evaluation_queries.xml'
    doc_list_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/eng-docs-list-win.txt'
    doc_entity_name_file = 'e:/data/emadr/el/tac/2011/eval/doc_entity_candidates.txt'
    dst_query_file = r'e:/data/el/LDC2015E19/data/2011/eval/data/queries_expanded.xml'

    __acronym_expansion(query_file, doc_list_file, doc_entity_name_file, dst_query_file)


def __job_name_expansion():
    query_file = r'e:\data\el\LDC2015E19\data\2011\eval\tac' \
                 r'_kbp_2011_english_entity_linking_evaluation_queries.xml'
    doc_list_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/eng-docs-list-win.txt'
    doc_ner_file = 'e:/data/el/LDC2015E19/data/2011/eval/data/doc-entities-ner.txt'
    dst_query_file = r'e:/data/el/LDC2015E19/data/2011/eval/data/queries-name-expansion.xml'
    __name_expansion(query_file, doc_list_file, doc_ner_file, dst_query_file)

if __name__ == '__main__':
    # __gen_tac_docs()
    # __setup_doc_entities_file()
    # gen_doc_mention_names()
    # __job_acronym_expansion()
    __job_name_expansion()
    # process_docs_for_ner()

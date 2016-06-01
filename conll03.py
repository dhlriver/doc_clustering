from itertools import izip
from prepare_net_data import init_entity_net, gen_entity_name_dict


def __load_entity_names(entity_name_file):
    fin = open(entity_name_file, 'rb')
    names_set = set()
    for line in fin:
        vals = line.strip().split('\t')
        name = vals[0]
        names_set.add(name)
    fin.close()
    return names_set


def raw_to_tokenized_lines(raw_conll_file, dst_file):
    fin = open(raw_conll_file, 'rb')
    fout = open(dst_file, 'wb')
    cur_words = list()
    cnt = 0
    for line in fin:
        line = line.strip()
        if len(line) == 0:
            if cur_words:
                for i, word, in enumerate(cur_words):
                    if i > 0:
                        fout.write(' ')
                    fout.write(word)
                fout.write('\n')

                cur_words = list()
                cnt += 1
                # if cnt == 10:
                #     break
            continue
        else:
            vals = line.split(' ')
            cur_words.append(vals[0])
    fin.close()
    fout.close()


def gen_ner_results(doc_entity_candidates_file, doc_entity_candidate_indices_file,
                    entity_names_file, dst_indices_file):
    name_set = __load_entity_names(entity_names_file)
    fin0 = open(doc_entity_candidates_file, 'rb')
    fin1 = open(doc_entity_candidate_indices_file, 'rb')
    fout = open(dst_indices_file, 'wb')
    for line0, line1 in izip(fin0, fin1):
        name_vals = line0.strip().split('\t')
        idx_vals = line1.strip().split('\t')
        for name, indices in izip(name_vals, idx_vals):
            if name in name_set:
                fout.write(indices + '\t')
        fout.write('\n')
    fin0.close()
    fin1.close()
    fout.close()


def raw_gold_to_mention_indices(gold_file, gold_mention_indices_file):
    fout = open(gold_mention_indices_file, 'wb')
    fin_gold = open(gold_file, 'rb')
    cur_gold_mentions = list()
    cur_gold_mention = list()
    gold_mention_cnt = 0
    cur_word_idx = 0
    for line in fin_gold:
        line = line.strip()
        if len(line) == 0:
            if cur_gold_mention:
                cur_gold_mentions.append(cur_gold_mention)
                cur_gold_mention = list()
                gold_mention_cnt += 1
            for i, mention_indices_list in enumerate(cur_gold_mentions):
                for j, idx in enumerate(mention_indices_list):
                    if i > 0 and j == 0:
                        fout.write('\t')
                    if j > 0:
                        fout.write(' ')
                    fout.write(str(idx))
            fout.write('\n')
            cur_gold_mentions = list()
            cur_word_idx = 0
            continue

        vals = line.split(' ')
        if vals[3].startswith('I'):
            cur_gold_mention.append(cur_word_idx)
        else:
            if cur_gold_mention:
                cur_gold_mentions.append(cur_gold_mention)
                gold_mention_cnt += 1
                cur_gold_mention = list()
        cur_word_idx += 1
    fin_gold.close()
    fout.close()


def __load_mention_indices(mention_indices_file):
    fin = open(mention_indices_file, 'rb')
    mention_indices_list = list()
    for line in fin:
        cur_sent_mentions = list()
        line = line.strip()
        if len(line) > 0:
            vals = line.split('\t')
            for val in vals:
                mention = list()
                idx_vals = val.split(' ')
                for idx_val in idx_vals:
                    mention.append(int(idx_val))
                cur_sent_mentions.append(mention)
        mention_indices_list.append(cur_sent_mentions)
    fin.close()
    return mention_indices_list


def evaluate_ner_result(sys_mention_indices_file, gold_mention_indices_file):
    gold_mentions = __load_mention_indices(gold_mention_indices_file)
    sys_mentions = __load_mention_indices(sys_mention_indices_file)
    cnt_sys = 0
    cnt_gold = 0
    hit_cnt = 0
    for sent_mentions_gold, sent_mentions_sys in izip(gold_mentions, sys_mentions):
        cnt_sys += len(sent_mentions_sys)
        cnt_gold += len(sent_mentions_gold)
        for mention_sys in sent_mentions_sys:
            for mention_gold in sent_mentions_gold:
                if len(mention_sys) != len(mention_gold):
                    continue
                hit = True
                for idx_sys, idx_gold in izip(mention_sys, mention_gold):
                    if idx_gold != idx_sys:
                        hit = False
                        break
                if hit:
                    hit_cnt += 1
    print hit_cnt, cnt_sys, cnt_gold
    hit_cnt = float(hit_cnt)
    p, r = hit_cnt / cnt_sys, hit_cnt / cnt_gold
    print p, r, 2 * p * r / (p + r)


def transform_dataset():
    raw_conll_file = 'e:/exp/emadr/eng.testb.txt'
    dst_file = 'e:/exp/emadr/testb-lines.txt'
    raw_to_tokenized_lines(raw_conll_file, dst_file)


def retrieve_mentions():
    line_docs_file_name = 'e:/exp/emadr/testb-lines.txt'
    illegal_start_words_file = 'e:/dc/20ng_bydate/stopwords.txt'
    dst_doc_entity_candidates_list_file = 'e:/exp/emadr/doc_entity_candidates.txt'
    dst_entity_candidate_clique_file = 'e:/exp/emadr/entity_candidate_cliques.txt'
    dst_doc_entity_indices_file = 'e:/exp/emadr/doc_entity_candidate_indices.txt'
    init_entity_net(line_docs_file_name, illegal_start_words_file, dst_doc_entity_candidates_list_file,
                    dst_entity_candidate_clique_file, dst_doc_entity_indices_file)

    lc_word_cnts_file_name = 'e:/dc/el/wiki/wiki_word_cnts_lc.txt'
    wc_word_cnts_file_name = 'e:/dc/el/wiki/wiki_word_cnts_with_case.txt'
    dst_entity_name_list_file = 'e:/exp/emadr/entity_names.txt'
    gen_entity_name_dict(dst_doc_entity_candidates_list_file, lc_word_cnts_file_name, wc_word_cnts_file_name,
                         dst_entity_name_list_file)


def job_gen_ner_results():
    doc_entity_candidates_file = 'e:/exp/emadr/doc_entity_candidates.txt'
    doc_entity_candidate_indices_file = 'e:/exp/emadr/doc_entity_candidate_indices.txt'
    entity_names_file = 'e:/exp/emadr/entity_names.txt'
    dst_indices_file = 'e:/exp/emadr/metion_indices.txt'
    gen_ner_results(doc_entity_candidates_file, doc_entity_candidate_indices_file,
                    entity_names_file, dst_indices_file)


def job_raw_gold_to_mention_indices():
    gold_file = 'e:/exp/emadr/eng.testb.txt'
    gold_mention_indices_file = 'e:/exp/emadr/gold_metion_indices.txt'
    raw_gold_to_mention_indices(gold_file, gold_mention_indices_file)


def job_evaluate_ner_result():
    sys_mention_indices_file = 'e:/exp/emadr/mention_indices.txt'
    gold_mention_indices_file = 'e:/exp/emadr/gold_metion_indices.txt'
    evaluate_ner_result(sys_mention_indices_file, gold_mention_indices_file)


def main():
    # transform_dataset()
    # retrieve_mentions()
    # job_gen_ner_results()
    # job_raw_gold_to_mention_indices()
    job_evaluate_ner_result()

if __name__ == '__main__':
    main()

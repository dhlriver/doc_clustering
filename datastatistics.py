import numpy as np
import dataarange
import textutils


def __count_mentions(ner_result_file, filter_type):
    cnt = 0
    doc_cnt = 0
    fin = open(ner_result_file, 'rb')
    for line in fin:
        doc_cnt += 1
        num_mentions = int(line[:-1])
        for i in xrange(num_mentions):
            line = fin.next()
            vals = line[:-1].split('\t')
            if vals[3] != filter_type:
                cnt += 1
            # cnt += 1
    fin.close()
    return cnt, doc_cnt


def __count_sentences(tokenized_docs_file):
    fin = open(tokenized_docs_file, 'rb')
    sentence_cnt = 0
    for line in fin:
        words = line.strip().split(' ')
        for word in words:
            if textutils.is_sentence_end(word):
                sentence_cnt += 1
    fin.close()
    return sentence_cnt


def __count_sentences_with_mentions_pairs(cooccur_mention_file, entity_name_dict_file):
    entity_name_dict = dataarange.load_entity_dict(entity_name_dict_file)
    fin = open(cooccur_mention_file, 'rb')
    cnt = 0
    for line in fin:
        mentions = line.strip().split('\t')
        tmp_cnt = 0
        for mention in mentions:
            if mention in entity_name_dict:
                tmp_cnt += 1
        if tmp_cnt > 1:
            cnt += 1
    fin.close()
    return cnt


def __count_pairs_in_pair_file(file_name):
    fin = open(file_name, 'rb')
    num_left, num_right = np.fromfile(fin, np.int32, 2)
    print num_left, num_right
    pairs_cnt = 0
    for i in xrange(num_left):
        num_val = np.fromfile(fin, np.int32, 1)
        np.fromfile(fin, np.int32, num_val)
        cnts = np.fromfile(fin, np.uint16, num_val)
        pairs_cnt += sum(cnts)
    fin.close()
    return pairs_cnt


def dataset_statistics_nyt():
    ner_result_file_test = 'e:/dc/nyt-world-full/processed/test/ner-result.txt'
    ner_result_file_train = 'e:/dc/nyt-world-full/processed/train/ner-result.txt'
    cnt_test, doc_cnt_test = __count_mentions(ner_result_file_test, 'LOCATION')
    # cnt_train, doc_cnt_train = __count_mentions(ner_result_file_train, 'LOCATION')
    # print cnt, doc_cnt
    cnt_train = doc_cnt_train = 0
    cnt = cnt_test + cnt_train
    doc_cnt = doc_cnt_test + doc_cnt_train
    print cnt, doc_cnt
    print float(cnt) / doc_cnt

    tokenized_docs_file_test = 'e:/dc/nyt-world-full/processed/test/docs-tokenized.txt'
    tokenized_docs_file_train = 'e:/dc/nyt-world-full/processed/train/docs-tokenized.txt'
    sentence_cnt_test = __count_sentences(tokenized_docs_file_test)
    # sentence_cnt_train = __count_sentences(tokenized_docs_file_train)
    sentence_cnt_train = 0
    sentence_cnt = sentence_cnt_test + sentence_cnt_train
    print sentence_cnt_test, sentence_cnt_train, sentence_cnt
    print float(cnt) / sentence_cnt

    entity_name_dict_file_test = 'e:/dc/nyt-world-full/processed/test/entity-names-nloc.txt'
    entity_name_dict_file_train = 'e:/dc/nyt-world-full/processed/train/entity-names-nloc.txt'
    cooccur_mention_file_test = 'e:/dc/nyt-world-full/processed/test/cooccur-mentions.txt'
    cooccur_mention_file_train = 'e:/dc/nyt-world-full/processed/train/cooccur-mentions.txt'
    valid_cnt_test = __count_sentences_with_mentions_pairs(cooccur_mention_file_test, entity_name_dict_file_test)
    # valid_cnt_train = __count_sentences_with_mentions_pairs(cooccur_mention_file_train, entity_name_dict_file_train)
    valid_cnt_train = 0
    print valid_cnt_test, valid_cnt_train, valid_cnt_test + valid_cnt_train

    ee_pair_file_train = 'e:/dc/nyt-world-full/processed/bin/ee-ner-train.bin'
    ee_pair_file_test = 'e:/dc/nyt-world-full/processed/bin/ee-ner.bin'
    pairs_cnt_test = __count_pairs_in_pair_file(ee_pair_file_test)
    # pairs_cnt_train = __count_pairs_in_pair_file(ee_pair_file_train)
    pairs_cnt_train = 0
    print pairs_cnt_test, pairs_cnt_train, pairs_cnt_test + pairs_cnt_train


def dataset_statistics_tac():
    ner_result_file = 'e:/dc/el/tac/2010/train/ner-result.txt'
    mention_cnt, doc_cnt = __count_mentions(ner_result_file, None)
    print mention_cnt, doc_cnt
    print float(mention_cnt) / doc_cnt

    tokenized_docs_file = 'e:/dc/el/tac/2010/train/docs-ner-tokenized.txt'
    sentence_cnt = __count_sentences(tokenized_docs_file)
    print sentence_cnt
    print float(mention_cnt) / sentence_cnt


if __name__ == '__main__':
    # dataset_statistics_nyt()
    dataset_statistics_tac()

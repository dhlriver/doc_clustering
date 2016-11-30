import numpy as np
from itertools import izip
import ioutils
from ioutils import load_labels_file

sentence_end_words = ['.', '?', '<s>', '!']


def split_docs_text_file_by_dataset_labels(doc_text_file, dataset_split_file,
                                           dst_train_doc_text_file, dst_test_doc_text_file):
    data_split_labels = load_labels_file(dataset_split_file)
    print data_split_labels[:10]
    print len(data_split_labels)
    fin = open(doc_text_file, 'r')
    ftrain = open(dst_train_doc_text_file, 'wb')
    ftest = open(dst_test_doc_text_file, 'wb')
    for l, line in izip(data_split_labels, fin):
        if l == 0:
            ftrain.write(line)
        else:
            ftest.write(line)
    fin.close()
    ftrain.close()
    ftest.close()


def split_line_docs_file(file_name, dst_files):
    file_len = ioutils.get_file_len(file_name)
    cur_file_idx = 0
    next_pos = file_len / (len(dst_files) - cur_file_idx)
    fin = open(file_name, 'rb')
    fout = open(dst_files[cur_file_idx], 'wb')
    print 'writing #%d' % cur_file_idx
    for i, line in enumerate(fin):
        cur_pos = fin.tell()
        if cur_pos > next_pos:
            cur_file_idx += 1
            fout.close()
            next_pos = cur_pos + (file_len - cur_pos) / (len(dst_files) - cur_file_idx)
            fout = open(dst_files[cur_file_idx], 'wb')
            print 'writing #%d' % cur_file_idx
        fout.write(line)

    fin.close()
    fout.close()


def stem_tokenized_text(tokenized_docs_file, dst_file):
    fin = open(tokenized_docs_file, 'rb')
    fout = open(dst_file, 'wb')
    for line in fin:
        words = line.strip().split(' ')

    fin.close()
    fout.close()


def tokenized_text_to_bow(tokenized_text_file, word_dict_file, dst_bow_file, min_occurance=5):
    word_dict = load_words_to_idx_dict(word_dict_file, min_occurance)
    print 'num words in dict: %d' % len(word_dict)

    fin = open(tokenized_text_file, 'rb')
    fout = open(dst_bow_file, 'wb')
    for line in fin:
        doc_bow = dict()
        words = line.strip().split(' ')
        for word in words:
            idx = word_dict.get(word, -1)
            if idx != -1:
                cnt = doc_bow.get(idx, 0)
                doc_bow[idx] = cnt + 1

        for i, (idx, cnt) in enumerate(doc_bow.iteritems()):
            if i > 0:
                fout.write(' ')
            fout.write('%d:%d' % (idx, cnt))
        fout.write('\n')

    fin.close()
    fout.close()


def get_num_lines_in_file(filename):
    fin = open(filename, 'rb')
    cnt = 0
    for _ in fin:
        cnt += 1
    fin.close()
    return cnt


def doc_to_line(doc_file):
    fin = open(doc_file, 'rb')
    text = ''
    for idx, line in enumerate(fin):
        if idx != 0:
            text += ' <s> '
        text += line.strip()
    fin.close()

    return text


def all_uppercase_word(word):
    if (len(word) < 3) or (not word.isupper()):
        return False

    for ch in word:
        if (not ch.isalpha()) and ch != '-' and ch != '.':
            return False

    return True


def is_sentence_end(word):
    return word in sentence_end_words


def load_words_to_set(file_name, with_cnts=False, has_num_words=False, min_occurance=0):
    words = set()

    fin = open(file_name, 'rb')
    if has_num_words:
        fin.next()

    if with_cnts:
        for line in fin:
            vals = line[:-1].split('\t')
            if int(vals[1]) < min_occurance:
                continue
            words.add(vals[0])
    else:
        for line in fin:
            words.add(line[:-1])
    fin.close()
    return words


def load_words_to_idx_dict(dict_file, min_occurance=2):
    fin = open(dict_file, 'rb')
    word_dict = dict()
    for idx, line in enumerate(fin):
        vals = line.strip().split('\t')
        if int(vals[1]) < min_occurance:
            continue
        word_dict[vals[0]] = len(word_dict)
    fin.close()
    return word_dict


def __get_word_cnts_dict(tokenized_line_docs_file, max_word_len=20, to_lower=True):
    word_cnts = dict()
    fin = open(tokenized_line_docs_file, 'rb')
    line_cnt = 0
    for line_cnt, line in enumerate(fin):
        if to_lower:
            line = line.lower()
        words = line.strip().split(' ')
        doc_words = set()
        for word in words:
            word = word.strip()
            if len(word) > max_word_len or len(word) < 2:
                continue
            doc_words.add(word)

        for word in doc_words:
            cnt = word_cnts.get(word, 0)
            word_cnts[word] = cnt + 1

        if line_cnt % 10000 == 10000 - 1:
            print line_cnt + 1
        # if line_cnt == 5:
        #     break
    fin.close()

    print '%d lines in %s' % (line_cnt + 1, tokenized_line_docs_file)
    return word_cnts


def gen_word_cnts_dict_with_line_docs(tokenized_line_docs_file, dst_file_name, min_occurance=3,
                                      max_word_len=20, tolower=True, stopwords_file=None):
    word_cnts = __get_word_cnts_dict(tokenized_line_docs_file, max_word_len, tolower)
    stopwords = load_words_to_set(stopwords_file) if stopwords_file else None

    fout = open(dst_file_name, 'wb')
    for word, cnt, in word_cnts.items():
        if cnt < min_occurance:
            continue
        if stopwords and word in stopwords:
            continue
        fout.write('%s\t%d\n' % (word, cnt))
    fout.close()


def filter_words_in_line_docs(line_docs_file, word_dict_file, dst_file, with_num_docs_head=True):
    num_docs = 0
    if with_num_docs_head:
        num_docs = get_num_lines_in_file(line_docs_file)
        print num_docs, ' documents/lines'

    proper_words = load_words_to_set(word_dict_file, True)
    fin = open(line_docs_file, 'rb')
    fout = open(dst_file, 'wb')
    if with_num_docs_head:
        fout.write('%d\n' % num_docs)
    for line in fin:
        words = line.strip().split(' ')
        tmp_word_list = list()
        for word in words:
            if word in proper_words:
                tmp_word_list.append(word)
        for i, word in enumerate(tmp_word_list):
            if i > 0:
                fout.write(' ')
            fout.write(word)
        fout.write('\n')
        # break
    fin.close()
    fout.close()


# one word per line, no counts
def load_word_list(file_name):
    words = list()
    fin = open(file_name, 'rb')
    for line in fin:
        words.append(line.strip())
    fin.close()
    return words


def load_word_cnts(file_name):
    print 'loading', file_name, '...'
    word_cnts = dict()
    fin = open(file_name, 'rb')
    for line in fin:
        vals = line.strip().split('\t')
        word_cnts[vals[0]] = int(vals[1])
    print 'done.'
    return word_cnts


def legal_word(word, max_word_len):
    word_len = len(word)
    if word_len < 2 or word_len > max_word_len:
        return False

    if word[0] == '<':
        return False

    for ch in word:
        if ch.isalpha():
            return True
    return False


def gen_proper_words_dict_with_cnts(word_cnt_file_name, stop_words_file_name, min_word_cnt,
                                    max_word_len, dst_file_name):
    stop_words = load_words_to_set(stop_words_file_name)
    word_cnts = load_word_cnts(word_cnt_file_name)
    fout = open(dst_file_name, 'wb')
    for word, cnt in word_cnts.items():
        if cnt >= min_word_cnt and word not in stop_words and legal_word(word, max_word_len):
            fout.write('%s\t%d\n' % (word, cnt))
    fout.close()


# will remove words that aren't in proper_word_cnts_dict_file
def gen_lowercase_token_file(tokenized_line_docs_file_name, proper_word_cnts_dict_file,
                             max_word_len, min_word_occurrance, dst_file_name):
    words_dict = load_words_to_set(proper_word_cnts_dict_file, True, min_occurance=min_word_occurrance)
    print '%d words in dict' % len(words_dict)

    fin = open(tokenized_line_docs_file_name, 'rb')
    fout = open(dst_file_name, 'wb')
    doc_cnt = 0
    for doc_cnt, line in enumerate(fin):
        words = line.strip().lower().split(' ')
        is_first = True
        for word in words:
            if word not in words_dict:
                continue

            if is_first:
                is_first = False
            else:
                fout.write(' ')
            fout.write(word)
        fout.write('\n')

        if doc_cnt % 10000 == 10000 - 1:
            print doc_cnt + 1
        # if doc_cnt == 100:
        #     break
    print doc_cnt + 1, 'lines'
    fin.close()
    fout.close()


def line_docs_to_bow(line_docs_file_name, words_dict, min_occurance, dst_bow_docs_file_name):
    # fin = open(proper_word_cnts_dict_file, 'rb')
    # words_dict = dict()
    # for idx, line in enumerate(fin):
    #     vals = line.strip().split('\t')
    #     words_dict[vals[0]] = idx
    # fin.close()
    # words_dict = load_words_to_idx_dict(proper_word_cnts_dict_file, min_occurance)

    line_cnt = 0
    word_cnt = 0
    fin = open(line_docs_file_name, 'rb')
    fout = open(dst_bow_docs_file_name, 'wb')
    np.zeros(2, np.int32).tofile(fout)  # reserve space
    for line_cnt, line in enumerate(fin):
        words = line.strip().split(' ')
        doc_word_cnts = dict()
        for word in words:
            if len(word) == 0:
                continue
            idx = words_dict.get(word, -1)
            if idx < 0:
                continue
            cnt = doc_word_cnts.get(idx, 0)
            doc_word_cnts[idx] = cnt + 1

        word_indices = np.zeros(len(doc_word_cnts), np.int32)
        word_cnts_arr = np.zeros(len(doc_word_cnts), np.uint16)
        for i, (idx, cnt) in enumerate(doc_word_cnts.iteritems()):
            word_indices[i] = idx
            word_cnts_arr[i] = cnt
        # for idx, item in enumerate(doc_word_cnts.items()):
        #     word_indices[idx] = words_dict[item[0]]
        #     if item[1] > 65535:
        #         print 'word cnt larger than 65535!', item[0]
        #         word_cnts_arr[idx] = 65535
        #     else:
        #         word_cnts_arr[idx] = item[1]
            # fout.write('%d %d ' % (words_dict[word][0], cnt))
        # fout.write('\n')
        np.asarray([len(doc_word_cnts)], np.int32).tofile(fout)
        word_indices.tofile(fout)
        word_cnts_arr.tofile(fout)

        if line_cnt % 10000 == 10000 - 1:
            print line_cnt + 1
        # if line_cnt == 100:
        #     break
    fin.close()

    fout.seek(0)
    np.asarray([line_cnt + 1, len(words_dict)], np.int32).tofile(fout)
    fout.close()
    print line_cnt + 1, 'lines total'


def gen_word_cnts_file_from_bow_file(bow_file, dst_word_cnts_file):
    fin = open(bow_file, 'rb')
    num_docs, num_words = np.fromfile(fin, np.int32, 2)
    word_cnts = np.zeros(num_words, np.int32)
    for i in xrange(num_docs):
        num_doc_words = np.fromfile(fin, np.int32, 1)
        indices = np.fromfile(fin, np.int32, num_doc_words)
        cnts = np.fromfile(fin, np.uint16, num_doc_words)
        for idx, cnt in zip(indices, cnts):
            word_cnts[idx] += cnt

        if i % 100000 == 100000 - 1:
            print i + 1
    fin.close()

    fout = open(dst_word_cnts_file, 'wb')
    np.asarray([num_words], np.int32).tofile(fout)
    word_cnts.tofile(fout)
    fout.close()


def first_letter_uppercase(word):
    if len(word) == 0:
        return False

    if word == 'I':
        return False

    if not word[0].isupper():
        return False

    for ch in word:
        if (not ch.isalpha()) and ch != '-' and ch != '.':
            return False

    return not word.isupper()

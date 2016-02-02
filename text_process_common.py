import numpy as np

sentence_end_words = ['.', '?', '<s>', '!']


def all_uppercase_word(word):
    if (len(word) < 3) or (not word.isupper()):
        return False

    for ch in word:
        if (not ch.isalpha()) and ch != '-' and ch != '.':
            return False


def is_sentence_end(word):
    return word in sentence_end_words


def gen_word_cnts_dict_for_line_docs(tokenized_line_docs_file_name, dst_file_name, max_word_len=20, to_lower=True):
    word_cnts = dict()
    fin = open(tokenized_line_docs_file_name, 'rb')
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

    print line_cnt + 1, 'lines total'
    fout = open(dst_file_name, 'wb')
    for word, cnt, in word_cnts.items():
        fout.write('%s\t%d\n' % (word, cnt))
    fout.close()


def load_word_list(file_name):
    words = list()
    fin = open(file_name, 'rb')
    for line in fin:
        words.append(line.strip())
    fin.close()
    return words


def load_word_set(file_name, with_cnts=False, has_num_words=False):
    words = set()

    fin = open(file_name, 'rb')
    if has_num_words:
        fin.readline()

    if with_cnts:
        for line in fin:
            words.add(line.strip().split('\t')[0])
    else:
        for line in fin:
            words.add(line.strip())
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
    stop_words = load_word_set(stop_words_file_name)
    word_cnts = load_word_cnts(word_cnt_file_name)
    fout = open(dst_file_name, 'wb')
    for word, cnt in word_cnts.items():
        if cnt >= min_word_cnt and word not in stop_words and legal_word(word, max_word_len):
            fout.write('%s\t%d\n' % (word, cnt))
    fout.close()


def gen_lowercase_token_file(tokenized_line_docs_file_name, proper_word_cnts_dict_file,
                             max_word_len, dst_file_name):
    words_dict = load_word_set(proper_word_cnts_dict_file, True)

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


def line_docs_to_bow(line_docs_file_name, proper_word_cnts_dict_file, dst_bow_docs_file_name):
    fin = open(proper_word_cnts_dict_file, 'rb')
    words_dict = dict()
    for idx, line in enumerate(fin):
        vals = line.strip().split('\t')
        words_dict[vals[0]] = idx
    fin.close()

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
            cnt = doc_word_cnts.get(word, 0)
            doc_word_cnts[word] = cnt + 1

        word_indices = np.zeros(len(doc_word_cnts), np.int32)
        word_cnts_arr = np.zeros(len(doc_word_cnts), np.uint16)
        for idx, item in enumerate(doc_word_cnts.items()):
            word_indices[idx] = words_dict[item[0]]
            if item[1] > 65535:
                print 'word cnt larger than 65535!', item[0]
                word_cnts_arr[idx] = 65535
            else:
                word_cnts_arr[idx] = item[1]
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

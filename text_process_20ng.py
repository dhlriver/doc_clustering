import numpy as np
import ioutils
import textutils
# TODO use text_process_common


def get_num_lines(file_name):
    fin = open(file_name, 'rb')
    cnt = 0
    for cnt, line in enumerate(fin):
        pass
    fin.close()
    return cnt + 1


# generate a word dict from documents
# filter those that only occur once
def gen_word_dict(words_doc_file, dst_file_name, to_lowercase=False):
    fin = open(words_doc_file, 'rb')
    words_dict = dict()
    for line_idx, line in enumerate(fin):
        line = line.strip()
        if len(line) == 0:
            print line_idx, 'Empty line!'
            continue

        if to_lowercase:
            line = line.lower()
        words = line.split(' ')
        for word in words:
            cnt = words_dict.get(word, 0)
            words_dict[word] = cnt + 1
    fin.close()

    words_list = list()
    for key, value in words_dict.items():
        if value > 1:
            words_list.append((key, value))

    fout = open(dst_file_name, 'wb')
    fout.write(str(len(words_list)) + '\n')
    for key, value in words_list:
        fout.write(key + '\t' + str(value) + '\n')
    fout.close()


def load_word_dict(word_dict_file_name):
    word_dict = dict()
    fin = open(word_dict_file_name, 'rb')
    num_words = int(fin.readline())
    print 'word dict has', num_words, 'words'
    for i in xrange(num_words):
        line = fin.readline()
        vals = line.strip().split('\t')
        word_dict[vals[0]] = i
    fin.close()
    return word_dict


def save_as_word_indices(doc_line, word_dict, bin_index_cnts_fout, indices_fout=None):
    if doc_line == '':
        np.array([0], np.int32).tofile(bin_index_cnts_fout)
        if indices_fout:
            indices_fout.write('\n')
        return

    doc_word_dict = dict()
    word_indices = list()

    words = doc_line.split(' ')
    for word in words:
        word_idx = word_dict.get(word, -1)
        if word_idx == -1:
            continue
        cnt = doc_word_dict.get(word_idx, 0)
        doc_word_dict[word_idx] = cnt + 1
        word_indices.append(word_idx)

    np.array([len(doc_word_dict)], np.int32).tofile(bin_index_cnts_fout)
    np.array(doc_word_dict.keys(), np.int32).tofile(bin_index_cnts_fout)
    # np.array(doc_word_dict.values(), np.int32).tofile(bin_index_cnts_fout)
    np.array(doc_word_dict.values(), np.uint16).tofile(bin_index_cnts_fout)

    if indices_fout:
        if len(word_indices) > 0:
            indices_fout.write(str(word_indices[0]))
        for word_index in word_indices[1:]:
            indices_fout.write(' ' + str(word_index))
        indices_fout.write('\n')


def line_docs_to_net(line_docs_file_name, word_dict_file_name, dst_bin_file_name, dst_word_indices_doc_file_name):
    word_dict = load_word_dict(word_dict_file_name)
    num_docs = get_num_lines(line_docs_file_name)

    fout0 = open(dst_bin_file_name, 'wb')
    fout1 = None
    if dst_word_indices_doc_file_name:
        fout1 = open(dst_word_indices_doc_file_name, 'wb')

    np.array([num_docs, len(word_dict)], np.int32).tofile(fout0)

    fin = open(line_docs_file_name, 'rb')
    for line in fin:
        line = line.strip()
        save_as_word_indices(line, word_dict, fout0, fout1)

    fin.close()

    fout0.close()
    if fout1:
        fout1.close()


def to_processed_docs_final(docs_file_name, dst_file_name):
    fin = open(docs_file_name, 'rb')
    fout = open(dst_file_name, 'wb')
    for line in fin:
        fout.write(line.lower())
    fin.close()
    fout.close()


def do_gen_word_dict():
    words_doc_file = 'e:/dc/20ng_bydate/doc_text_data.txt'
    dst_dict_file_name = 'e:/dc/20ng_bydate/words_dict.txt'
    gen_word_dict(words_doc_file, dst_dict_file_name)


def process_docs_final():
    docs_file_name = 'e:/dc/20ng_bydate/doc_text.txt'
    dst_file_name = 'e:dc/20ng_bydate/doc_text_data.txt'
    to_processed_docs_final(docs_file_name, dst_file_name)


def test():
    words = ioutils.load_words_dict_to_list('e:/dc/20ng_bydate/words_dict.txt')
    fin = open('e:/dc/20ng_bydate/word_cnts.bin', 'rb')
    num_words = np.fromfile(fin, np.int32, 1)
    print num_words, 'words'
    cnts = np.fromfile(fin, np.int32, num_words)
    for idx, cnt in enumerate(cnts[:100]):
        print words[idx], cnt
    fin.close()


if __name__ == '__main__':
    test()
    # do_gen_word_dict()
    # process_docs_final()
    # all_line_docs_to_net()
    # do_line_docs_to_net_split()
    # train_lines_doc_to_net()
    # test_lines_doc_to_net()

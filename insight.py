import numpy as np
import io_utils
import heapq


def close_words():
    word_dict_file_name = 'e:/dc/20ng_bydate/words_dict.txt'
    words = io_utils.load_words_dict_to_list(word_dict_file_name)

    bow_docs_file_name = 'e:/dc/20ng_bydate/all_docs_dw_net.bin'
    word_indices_list, word_cnts_list, num_words = io_utils.load_bow_file(bow_docs_file_name)
    print num_words, 'words'

    doc_vec_file_name = 'e:/dc/20ng_bydate/vecs/doc_vec_cpp_100.bin'
    word_vec_file_name = 'e:/dc/20ng_bydate/vecs/word_vecs_cpp_100.bin'
    doc_vecs = io_utils.load_vec_list_file(doc_vec_file_name)
    word_vecs = io_utils.load_vec_list_file(word_vec_file_name)

    def show_close_words(doc_vec, word_indices):
        dist_list = list()
        for word_idx in word_indices:
            dist_list.append((np.dot(doc_vec, word_vecs[word_idx]), word_idx))
        dist_list.sort(key=lambda tup: tup[0])
        # closest_words = heapq.nlargest(10, dist_list, key=lambda tup: tup[0])
        for dist, idx in dist_list:
            print dist, words[idx]

    show_close_words(doc_vecs[0], word_indices_list[0])


def main():
    close_words()


if __name__ == '__main__':
    main()

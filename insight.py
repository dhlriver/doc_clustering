import numpy as np
import io_utils
import heapq
import scipy.spatial


def close_words_of_entities():
    word_dict_file_name = 'e:/dc/20ng_bydate/words_dict.txt'
    word_vec_file_name = 'e:/dc/20ng_bydate/vecs/word_vecs_joint_oml_100.bin'
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    entity_vecs_file_name = 'e:/dc/20ng_bydate/vecs/entity_vecs_joint_oml_100.bin'

    word_vecs = io_utils.load_vec_list_file(word_vec_file_name)
    words = io_utils.load_words_dict_to_list(word_dict_file_name)
    entity_vecs = io_utils.load_vec_list_file(entity_vecs_file_name)
    entities = io_utils.load_entity_dict(entity_dict_file_name)
    print len(entity_vecs)
    print len(entities)

    def show_close_words(entity_idx):
        print 'entity: ', entities[entity_idx]
        entity_vec = entity_vecs[entity_idx]
        dist_list = list()
        for word_idx in xrange(len(word_vecs)):
            # dist = np.dot(entity_vec, word_vecs[word_idx])
            dist = scipy.spatial.distance.cosine(entity_vec, word_vecs[word_idx])
            dist_list.append((dist, word_idx))
        # dist_list.sort(key=lambda tup: tup[0])
        closest_words = heapq.nsmallest(100, dist_list, key=lambda tup: tup[0])
        for dist, idx in closest_words:
            print dist, words[idx], idx

    show_close_words(25304)


def close_words_of_docs():
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
    # close_words_of_docs()
    close_words_of_entities()


if __name__ == '__main__':
    main()

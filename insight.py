import numpy as np
import ioutils
import heapq
import scipy.spatial
import math


def close_vecs(vecs, dst_vec, k=10):
    num_vecs = len(vecs)
    top_list = [(-1, -1) for _ in xrange(k)]
    for i in xrange(num_vecs):
        dist = scipy.spatial.distance.cosine(dst_vec, vecs[i])
        if math.isnan(dist):
            print vecs[i]
        insert_to_top(top_list, (i, dist))

        if i % 100000 == 100000 - 1:
            print i + 1
        if i == 2000000:
            break
    return top_list


def insert_to_top(top_list, val_pair):
    k = len(top_list)
    pos = 0
    for pos in xrange(k - 1, -1, -1):
        if top_list[pos][0] == -1:
            continue
        if top_list[pos][1] <= val_pair[1]:
            break

    if pos == k - 1:
        return

    if pos == 0 and (top_list[pos][1] > val_pair[1] or top_list[pos][0] == -1):
        pos = -1

    for i in xrange(k - 1, pos + 1, -1):
        top_list[i] = top_list[i - 1]

    top_list[pos + 1] = val_pair


def close_words():
    words_dict_file = 'e:/dc/el/words_dict_proper.txt'
    word_vecs_file = 'e:/dc/el/vecs/word_vecs.bin'

    words = ioutils.load_words_dict_to_list(words_dict_file, False)
    print len(words)
    idx = 774318
    print words[idx]

    word_vecs = ioutils.load_vec_list_file(word_vecs_file)
    print len(word_vecs)
    close_list = close_vecs(word_vecs, word_vecs[idx])
    for idx, dist in close_list:
        print words[idx], dist


def close_words_of_entities():
    word_dict_file_name = 'e:/dc/20ng_bydate/words_dict.txt'
    word_vec_file_name = 'e:/dc/20ng_bydate/vecs/word_vecs_joint_oml_100.bin'
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    entity_vecs_file_name = 'e:/dc/20ng_bydate/vecs/entity_vecs_joint_oml_100.bin'

    word_vecs = ioutils.load_vec_list_file(word_vec_file_name)
    words = ioutils.load_words_dict_to_list(word_dict_file_name)
    entity_vecs = ioutils.load_vec_list_file(entity_vecs_file_name)
    entities = ioutils.load_entity_dict(entity_dict_file_name)
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
    words = ioutils.load_words_dict_to_list(word_dict_file_name)

    bow_docs_file_name = 'e:/dc/20ng_bydate/all_docs_dw_net.bin'
    word_indices_list, word_cnts_list, num_words = ioutils.load_bow_file(bow_docs_file_name)
    print num_words, 'words'

    doc_vec_file_name = 'e:/dc/20ng_bydate/vecs/doc_vec_cpp_100.bin'
    word_vec_file_name = 'e:/dc/20ng_bydate/vecs/word_vecs_cpp_100.bin'
    doc_vecs = ioutils.load_vec_list_file(doc_vec_file_name)
    word_vecs = ioutils.load_vec_list_file(word_vec_file_name)

    def show_close_words(doc_vec, word_indices):
        dist_list = list()
        for word_idx in word_indices:
            dist_list.append((np.dot(doc_vec, word_vecs[word_idx]), word_idx))
        dist_list.sort(key=lambda tup: tup[0])
        # closest_words = heapq.nlargest(10, dist_list, key=lambda tup: tup[0])
        for dist, idx in dist_list:
            print dist, words[idx]

    show_close_words(doc_vecs[0], word_indices_list[0])


def close_docs(doc_vec_file, dst_idx=0, k=10):
    # doc_vec_file = 'e:/dc/20ng_bydate/vecs/dedw_vecs_0.bin'

    print 'loading', doc_vec_file
    doc_vecs = ioutils.load_vec_list_file(doc_vec_file)
    print 'done'

    dst_vec = doc_vecs[dst_idx]
    return close_vecs(doc_vecs, dst_vec)


def close_20ng_docs():
    # doc_vec_file = 'e:/dc/20ng_bydate/vecs/dedw_vecs_0.bin'
    doc_vec_file = 'e:/dc/20ng_bydate/vecs/dbow_doc_vecs.bin'
    # av_doc_vec_file = 'e:/dc/20ng_bydate/vecs/doc_vecs_av.bin'
    print 'loading', doc_vec_file
    doc_vecs = ioutils.load_vec_list_file(doc_vec_file)
    print len(doc_vecs)
    # dw_vecs = np.zeros((len(doc_vecs), 50), np.float32)
    # for i in xrange(len(doc_vecs)):
    #     dw_vecs[i] = doc_vecs[i][50:]
    # print 'done'

    # print 'loading', av_doc_vec_file
    # av_doc_vecs = ioutils.load_vec_list_file(av_doc_vec_file)
    # print 'done'

    dst_vec = doc_vecs[0]
    close_list = close_vecs(doc_vecs, dst_vec)
    for idx, dist in close_list:
        print idx, dist

    # print ''
    # dst_vec = av_doc_vecs[0]
    # close_list = close_vecs(dw_vecs, dst_vec, 20)
    # for idx, dist in close_list:
    #     print idx, dist


def close_wiki_pages():
    wiki_page_id_file = 'e:/dc/el/wiki/wiki_page_ids.bin'
    fin = open(wiki_page_id_file, 'rb')
    num_pages = np.fromfile(fin, np.int32, 1)
    page_ids = np.fromfile(fin, np.int32, num_pages)
    fin.close()

    wiki_vec_file = 'e:/dc/el/vecs/wiki_de_vecs.bin'
    # fin = open(wiki_vec_file, 'rb')
    # num_vecs, dim = np.fromfile(fin, np.int32, 2)
    # wiki_vecs = np.zeros((num_vecs, 50), np.float32)
    # for i in xrange(num_vecs):
    #     vec = np.fromfile(fin, np.float32, dim)
    #     wiki_vecs[i][:] = vec[50:]
    # fin.close()
    wiki_vecs = ioutils.load_vec_list_file(wiki_vec_file)

    top_list = close_vecs(wiki_vecs, wiki_vecs[0])
    for idx, dist in top_list:
        print page_ids[idx], dist


def close_wiki_pages_el_doc():
    wiki_page_id_file = 'e:/dc/el/wiki/wiki_page_ids.bin'
    fin = open(wiki_page_id_file, 'rb')
    num_pages = np.fromfile(fin, np.int32, 1)
    page_ids = np.fromfile(fin, np.int32, num_pages)
    fin.close()

    wiki_vec_file = 'e:/dc/el/vecs/wiki_dedw_vecs.bin'
    fin = open(wiki_vec_file, 'rb')
    num_vecs, dim = np.fromfile(fin, np.int32, 2)
    # wiki_vecs = np.zeros((num_vecs, 50), np.float32)
    wiki_vecs = np.zeros((num_vecs, 100), np.float32)
    for i in xrange(num_vecs):
        vec = np.fromfile(fin, np.float32, dim)
        # wiki_vecs[i][:] = vec[50:]
        wiki_vecs[i][:] = vec
    fin.close()

    # el_doc_vec_file = 'e:/dc/el/vecs/tac_2014_train_doc_vecs_av.bin'
    # el_doc_vec_file = 'e:/dc/el/vecs/wiki_vecs_av.bin'
    el_doc_vec_file = 'e:/dc/el/vecs/tac_2014_eval_vecs.bin'
    # el_doc_vec_file = 'e:/dc/el/vecs/tac_2014_train_dw_vecs.bin'
    el_doc_vecs = ioutils.load_vec_list_file(el_doc_vec_file)
    top_list = close_vecs(wiki_vecs, el_doc_vecs[0])
    for idx, dist in top_list:
        print page_ids[idx], dist

    # el_doc_vec_file = 'e:/dc/el/vecs/tac_2014_train_doc_vecs_av.bin'
    # el_doc_vecs = ioutils.load_vec_list_file(el_doc_vec_file)
    # print el_doc_vecs[0]
    # top_list = close_vecs(el_doc_vecs, el_doc_vecs[0])
    # for idx, dist in top_list:
    #     print idx, dist


def load_entity_name_set(dict_file_name):
    names = set()
    fin = open(dict_file_name, 'rb')
    for idx, line in enumerate(fin):
        vals = line.split('\t')
        names.add(vals[0])
    fin.close()
    return names


def test():
    entity_candidate_cliques_file = 'e:/dc/el/wiki/entity_candidate_cliques.txt'
    proper_entity_dict_file = 'e:/dc/el/wiki/entity_names.txt'
    names_set = load_entity_name_set(proper_entity_dict_file)

    fin = open(entity_candidate_cliques_file, 'rb')
    sent_cnt = 0
    for line_idx, line in enumerate(fin):
        entity_names = line.strip().split('\t')
        cnt = 0
        for entity_name in entity_names:
            entity_name = entity_name.strip()
            if not entity_name:
                continue
            if entity_name in names_set:
                cnt += 1
            if cnt >= 2:
                sent_cnt += 1
                break

        if line_idx % 1000000 == 1000000 - 1:
            print line_idx + 1
        # if line_idx == 1000:
        #     break
    fin.close()
    print sent_cnt


def _count_words(words):
    words_dict = dict()
    for word in words:
        cnt = words_dict.get(word, 0)
        words_dict[word] = cnt + 1
    return words_dict


def words_in_common():
    docs_file = 'e:/dc/20ng_bydate/doc_text_data.txt'
    idx0, idx1 = 0, 1
    len0, len1 = 0, 0
    words_dict0, words_dict1 = None, None
    fin = open(docs_file, 'rb')
    for i, line in enumerate(fin):
        if i == idx0:
            doc_words = line.strip().split(' ')
            len0 = len(doc_words)
            words_dict0 = _count_words(doc_words)
        elif i == idx1:
            doc_words = line.strip().split(' ')
            len1 = len(doc_words)
            words_dict1 = _count_words(doc_words)
    fin.close()

    same_cnt = 0
    for word, cnt in words_dict0.iteritems():
        cnt1 = words_dict1.get(word, -1)
        if cnt1 == -1:
            continue
        # print word, cnt, cnt1, cnt / float(len0), cnt1 / float(len1)
        print '%s\t%d\t%d\t%f\t%f' % (word, cnt, cnt1, cnt / float(len0), cnt1 / float(len1))
        same_cnt += 1
    print same_cnt


def main():
    # close_words()
    # close_words_of_docs()
    # close_words_of_entities()
    close_wiki_pages()
    # close_wiki_pages_el_doc()

    # close_20ng_docs()
    # words_in_common()


if __name__ == '__main__':
    main()
    # test()

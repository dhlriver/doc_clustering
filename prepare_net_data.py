import numpy
from array import array


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

def do_gen_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_bydate/doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def do_gen_train_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_data/split/train_entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_data/split/train_doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_data/split/train_doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def do_gen_test_20ng_doc_entity_list():
    entity_dict_file_name = 'e:/dc/20ng_data/split/train_entity_names.txt'
    raw_doc_entity_file_name = 'e:/dc/20ng_data/split/test_doc_entities_raw.txt'
    dst_doc_entity_file_name = 'e:/dc/20ng_data/split/test_doc_entities.bin'
    gen_20ng_doc_entity_list(entity_dict_file_name, raw_doc_entity_file_name, dst_doc_entity_file_name)


def job_gen_entity_edge_list_from_cliques():
    entity_dict_file_name = 'e:/dc/20ng_bydate/entity_names.txt'
    raw_entity_clique_file_name = 'e:/dc/20ng_bydate/entity_cliques_raw.txt'
    dst_weighted_edge_list_file_name = 'e:/dc/20ng_bydate/weighted_entity_edge_list.txt'
    gen_entity_edge_list_from_cliques(entity_dict_file_name, raw_entity_clique_file_name,
                                      dst_weighted_edge_list_file_name)


def main():
    # do_gen_20ng_path_label_file()
    # do_gen_20ng_doc_entity_list()
    # do_gen_train_20ng_doc_entity_list()
    # do_gen_test_20ng_doc_entity_list()
    job_gen_entity_edge_list_from_cliques()


def test():
    f0 = open('e:/dc/20ng_bydate/weighted_entity_edge_list.txt', 'rb')
    f1 = open('e:/dc/20ng_bydate/weighted_entity_edge_list_tmp.txt', 'rb')
    for idx, (line0, line1) in enumerate(zip(f0, f1)):
        if line0 != line1:
            print idx, 'not equal'
    f0.close()
    f1.close()

if __name__ == '__main__':
    # test()
    main()

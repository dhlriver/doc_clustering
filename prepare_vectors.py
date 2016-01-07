import numpy

label_list = ['africa', 'americas', 'asia', 'europe', 'middleeast']


def get_label_id(label):
    if label == 'null':
        return 0
    idx = 1
    for l in label_list:
        if label == l:
            return idx
        idx += 1
    return -1


def filter_vecs(full_vec_list_file_name, label_list_file_name, dst_file_name):
    fin0 = open(full_vec_list_file_name, 'rb')
    fin1 = open(label_list_file_name, 'rb')

    x = numpy.fromfile(fin0, '<i4', count=2)
    print x
    vec_len = x[1]
    vec_list = list()
    doc_cnt = 0
    for i in xrange(x[0]):
        vec = numpy.fromfile(fin0, '<f4', vec_len)
        line = fin1.readline().strip()
        vals = line.split('\t')
        label_id = get_label_id(vals[1])

        if label_id == -1:
            print 'label:', vals[1]
            continue
        if label_id == 0:
            continue

        vec_list.append(vec)
        doc_cnt += 1

    fout = open(dst_file_name, 'wb')
    numpy.array([len(vec_list), vec_len], numpy.int32).tofile(fout)
    for vec in vec_list:
        vec.tofile(fout)
    fout.close()


def filter_doc_entity_file(entity_list_file_name, full_doc_entity_file_name, label_list_file_name, dst_text_file_name,
                           dst_bin_file_name):
    fin = open(entity_list_file_name, 'rb')
    line = fin.readline()
    num_entities = int(line.strip())
    print num_entities, 'entities'
    fin.close()

    fin0 = open(full_doc_entity_file_name, 'rb')
    fin1 = open(label_list_file_name, 'rb')

    adj_list = list()
    weights_list = list()
    flg = False
    for line_idx, line0 in enumerate(fin0):
        if (line_idx + 1) % 5000 == 0:
            print line_idx + 1

        line1 = fin1.readline()
        if not line1:
            print 'file1 end earlier.'
            break

        vals0 = line0.strip().split(' ')
        vals1 = line1.strip().split('\t')
        if vals0[0] != vals1[0]:
            print 'doc not equal!'
            break

        if vals0[0].startswith('2012'):
            flg = True

        if not flg:
            continue

        label_id = get_label_id(vals1[1])
        if label_id <= 0:
            continue

        num_vertices = int(vals0[1])
        adj_vertices = numpy.zeros(num_vertices, dtype=numpy.int32)
        weights = numpy.zeros(num_vertices, dtype=numpy.int32)
        for i in xrange(num_vertices):
            adj_vertices[i] = int(vals0[i * 2 + 2])
            weights[i] = int(vals0[i * 2 + 3])
        adj_list.append(adj_vertices)
        weights_list.append(weights)
        # fout.write(line0)
    fin0.close()
    fin1.close()

    fout = open(dst_text_file_name, 'wb')
    fout.write(str(len(adj_list)) + ' ' + str(num_entities) + '\n')
    for i in xrange(len(adj_list)):
        fout.write(str(len(adj_list[i])))
        for j in xrange(len(adj_list[i])):
            fout.write(' ' + str(adj_list[i][j]) + ' ' + str(weights_list[i][j]))
        fout.write('\n')
    fout.close()

    fout = open(dst_bin_file_name, 'wb')
    numpy.array([len(adj_list), num_entities], dtype=numpy.int32).tofile(fout)
    for i in xrange(len(adj_list)):
        numpy.array([len(adj_list[i])], dtype=numpy.int32).tofile(fout)
        adj_list[i].tofile(fout)
        weights_list[i].tofile(fout)
    fout.close()


def do_filter_vecs():
    vec_list_file_name = 'e:/dc/nyt/vecs/es_doc_vec_64.bin'
    label_list_file_name = 'e:/dc/nyt/doc_label_list.txt'
    dst_file_name = 'e:/dc/nyt/vecs/es_doc_vec_64_lo.bin'
    filter_vecs(vec_list_file_name, label_list_file_name, dst_file_name)


def do_filter_doc_entity_file():
    entity_list_file_name = 'e:/dc/nyt/entity_name_id_list.txt'
    doc_entity_file_name = 'e:/dc/nyt/doc_entities.txt'
    label_list_file_name = 'e:/dc/nyt/doc_label_list.txt'
    dst_file_name = 'e:/dc/nyt/doc_entities_lo_f2012_tmp.txt'
    dst_bin_file_name = 'e:/dc/nyt/doc_entities_lo_f2012.bin'
    filter_doc_entity_file(entity_list_file_name, doc_entity_file_name, label_list_file_name,
                           dst_file_name, dst_bin_file_name)


def main():
    # do_filter_vecs()
    do_filter_doc_entity_file()


if __name__ == '__main__':
    main()

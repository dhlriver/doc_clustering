import text_process_common


def entity_names_in_words(words):
    idx = 1
    entity_names_list = list()
    while idx < len(words):
        cur_word = words[idx]
        if len(cur_word) == 0 or len(cur_word) > 20 :
            idx += 1
            continue

        if text_process_common.first_letter_uppercase(cur_word):
            cur_name = cur_word
            beg_idx = idx + 1
            while idx + 1 < len(words) and text_process_common.first_letter_uppercase(words[idx + 1]):
                idx += 1

            for i in xrange(beg_idx, idx + 1):
                cur_name += ' ' + words[i]
            if 2 < len(cur_name) < 50:
                entity_names_list.append(cur_name)
        idx += 1

    return entity_names_list


def find_possible_entity_names(line_docs_file, dst_file):
    fin = open(line_docs_file, 'rb')
    for line in fin:
        entity_names = entity_names_in_words(line.strip().split(' '))
        print entity_names
        break
    fin.close()


def main():
    line_docs_file = 'e:/dc/el/tac/tac_2009_eval_docs_text.txt'
    dst_file = 'e:/dc/el/tac/tac_2009_eval_names.txt'
    find_possible_entity_names(line_docs_file, dst_file)

if __name__ == '__main__':
    print main()

class Mention:
    # arrange mentions to a dict, { docid -> mentions-in-doc }
    @staticmethod
    def group_mentions_by_docid(mention_list):
        doc_mentions_dict = dict()
        for m in mention_list:
            # if m.mention_type != 'NOM':
            #     continue
            mlist = doc_mentions_dict.get(m.docid, list())
            if mlist:
                mlist.append(m)
            else:
                mlist.append(m)
                doc_mentions_dict[m.docid] = mlist
        return doc_mentions_dict

    # TODO remove
    @staticmethod
    def write_mentions(mentions, dst_file):
        fout = open(dst_file, 'wb')
        for m in mentions:
            fout.write('%s\t%d\t%d\t%s\t%s\n' % (m.docid, m.beg_pos, m.end_pos, m.name.encode('utf-8'),
                                                 m.entity_type))
        fout.close()

    @staticmethod
    def save_as_edl_file(mentions, dst_file, runid='ZJU'):
        fout = open(dst_file, 'wb')
        for m in mentions:
            m.to_edl_file(fout, runid)
        fout.close()

    @staticmethod
    def group_mentions_by_kbid(mentions):
        kbid_mentions_dict = dict()
        for m in mentions:
            cur_mentions = kbid_mentions_dict.get(m.kbid, list())
            if not cur_mentions:
                kbid_mentions_dict[m.kbid] = cur_mentions
            cur_mentions.append(m)
        return kbid_mentions_dict

    @staticmethod
    def group_mentions_by_qid(mentions):
        qid_mention_dict = dict()
        for m in mentions:
            qid_mention_dict[m.mention_id] = m
        return qid_mention_dict

    @staticmethod
    def load_edl_file(filename, arrange_by_docid=False):
        mentions = list()
        f = open(filename, 'r')
        for line in f:
            # print line,
            try:
                vals = line[:-1].split('\t')
                pos_vals = vals[3].split(':')
                pos_in_doc_vals = pos_vals[1].split('-')
                m = Mention(name=vals[2].decode('utf-8'), beg_pos=int(pos_in_doc_vals[0]), end_pos=int(pos_in_doc_vals[1]),
                            docid=pos_vals[0], mention_type=vals[6], entity_type=vals[5], kbid=vals[4], mention_id=vals[1])
                # if m.name.startswith('the '):
                #     m.name = m.name[4:]
                #     m.beg_pos += 4
                mentions.append(m)
            except:
                print 'error reading mentions'
                print line,
                exit()
        f.close()

        if arrange_by_docid:
            return Mention.group_mentions_by_docid(mentions)
        return mentions

    def to_edl_file(self, fout, runid='ZJU'):
        fout.write('%s\t%s\t%s\t%s:%d-%d\t%s\t%s\t%s\t1.0\n' % (
            runid, self.mention_id, self.name.encode('utf-8'), self.docid, self.beg_pos,
            self.end_pos, self.kbid, self.entity_type, self.mention_type))

    def __init__(self, name=u'', beg_pos=-1, end_pos=-1, docid='', mention_type='NAM',
                 entity_type='', kbid='NIL', mention_id='TEDL_000001'):
        self.mention_id = mention_id
        self.name = name
        self.beg_pos = beg_pos
        self.end_pos = end_pos
        self.docid = docid
        self.mention_type = mention_type
        self.entity_type = entity_type
        self.kbid = kbid

        # debug
        self.tags = None

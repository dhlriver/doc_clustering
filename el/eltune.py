import numpy as np
import theano
import theano.tensor as T

import neunet


class ELTune(object):
    def __init__(self, rng, n_in, n_middle, batch_size):
        self.hidden_layer = neunet.HiddenLayer(
            rng=rng,
            n_in=n_in,
            n_out=n_middle,
            activation=T.tanh
        )

        self.ones_vec = theano.shared(np.ones(batch_size, dtype=theano.config.floatX), borrow=True)
        self.zeros_vec = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX), borrow=True)
        self.l2_sqr = (self.hidden_layer.W ** 2).sum()
        self.params = [self.hidden_layer.W]

    def y_pred(self, mention_vecs, candidate_vecs, commonness, mask_matrix):
        mention_output = self.hidden_layer.get_output(mention_vecs)
        candidate_output = self.hidden_layer.get_output(candidate_vecs)
        # mention_output = mention_vecs
        # candidate_output = candidate_vecs

        c_norm = candidate_output.norm(2, axis=2).dimshuffle(0, 1, 'x')
        c = candidate_output / c_norm
        # c = candidate_output
        # m_norm = mention_output.norm(2, axis=1).dimshuffle(0, 'x')
        # m = (mention_output / m_norm).dimshuffle(0, 'x', 1)

        m = ELTune.unit_vec(mention_output).dimshuffle(0, 'x', 1)
        # m = mention_output.dimshuffle(0, 'x', 1)

        # cos similarities
        sims = (c * m).sum(axis=2) + mask_matrix
        # scores = 0.1 * commonness + 0.9 * sims
        scores = 0.45 * commonness + 0.55 * sims
        # scores = 0.6 * commonness + 0.4 * sims
        # scores = 0.45 * commonness
        # scores = sims

        # scores = 0.3 * T.log(commonness) + 0.7 * T.log(sims)
        # scores = sims
        y_pred = T.argmax(scores, 1)
        return y_pred, sims
        # return T.sum(T.eq(sys_y, y))

    def loss(self, mention_vecs, gold_vecs, crpt_vecs, cmns_gold, cmns_crpt, l2_reg):
        mention_output = self.hidden_layer.get_output(mention_vecs)
        gold_output = self.hidden_layer.get_output(gold_vecs)
        crpt_output = self.hidden_layer.get_output(crpt_vecs)

        mention_output = ELTune.unit_vec(mention_output)
        gold_output = ELTune.unit_vec(gold_output)
        crpt_output = ELTune.unit_vec(crpt_output)

        sims_gold = (mention_output * gold_output).sum(axis=1)
        sims_crpt = (mention_output * crpt_output).sum(axis=1)

        # scores_gold = 0.55 * cmns_gold + 0.45 * sims_gold
        # scores_crpt = 0.55 * cmns_crpt + 0.45 * sims_crpt
        scores_gold = sims_gold
        scores_crpt = sims_crpt

        return T.mean(T.maximum(self.zeros_vec, self.ones_vec - scores_gold + scores_crpt)) + l2_reg * self.l2_sqr

    def sims_gold(self, mention_vecs, gold_vecs):
        mention_output = self.hidden_layer.get_output(mention_vecs)
        gold_output = self.hidden_layer.get_output(gold_vecs)

        mention_output = ELTune.unit_vec(mention_output)
        gold_output = ELTune.unit_vec(gold_output)

        sims_gold = (mention_output * gold_output).sum(axis=1)
        return sims_gold

    @staticmethod
    def unit_vec(vecs):
        norms = vecs.norm(2, axis=1).dimshuffle(0, 'x')
        return vecs / norms

    @staticmethod
    def get_mask_matrix(candidates_nums, max_num_candidates):
        num_mentions = len(candidates_nums)
        mask_matrix = np.ones((num_mentions, max_num_candidates), theano.config.floatX)
        for i in xrange(num_mentions):
            mask_matrix[i][candidates_nums[i]:] = -1000
        return theano.shared(mask_matrix, borrow=True)

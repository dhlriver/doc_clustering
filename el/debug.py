import gzip
import os
import numpy as np
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import neunet
import timeit
import re


def load_data(dataset):
    print '... loading data'
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def test_mlp(dataset, learning_rate=0.01, l1_reg=0.00, l2_reg=0.0001, n_epochs=1000,
             batch_size=20, n_hidden=500):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print '... building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    classifier = neunet.MLP(
        rng=rng,
        input_vecs=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    cost = classifier.negative_log_likelihood(y) + l1_reg * classifier.L1 + l2_reg * classifier.L2_sqr

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        mode=theano.compile.mode.Mode(optimizer=None)
    )

    print('... training the model')
    patience = 10000  # look as this many examples regardless
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            cur_iter = (epoch - 1) * n_train_batches + minibatch_index

            if (cur_iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, cur_iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = cur_iter

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= cur_iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def filter_errors():
    fin = open('e:/el/error_list_09.txt', 'rb')
    err_values = list()
    for line in fin:
        vals = line.strip().split('\t')
        ch = vals[0][-1]
        if not ch.isdigit():
            continue
        if vals[1].startswith('NIL'):
            continue
        err_values.append(vals)
    fin.close()

    err_values.sort(key=lambda cur_val: cur_val[1])
    fout = open('e:/el/error_list_09_filtered_0.txt', 'wb')
    for err_val in err_values:
        fout.write('%s\t%s\t%s\t%s\n' % (err_val[0], err_val[1], err_val[2], err_val[3]))
    fout.close()


def show_errors():
    def load_queries(file_name):
        fin = open(file_name, 'rb')
        text = fin.read()
        fin.close()
        ps = r'<query id="(.*?)">\s*<name>(.*?)</name>\s*<docid>(.*?)</docid>'
        miter = re.finditer(ps, text)
        qdict = dict()
        for m in miter:
            qdict[m.group(1)] = (m.group(2), m.group(3))
        return qdict

    def load_file(file_name):
        fin = open(file_name, 'rb')
        tmp_err_list = list()
        for line in fin:
            vals = line.strip().split('\t')
            tmp_err_list.append(vals)
        fin.close()
        return tmp_err_list

    err_list = load_file('e:/el/error_list_09_filtered_0.txt')
    query_dict = load_queries(r'D:\data\el\LDC2015E19\data\2009\eval\tac_kbp_2'
                              r'009_english_entity_linking_evaluation_queries.xml')
    for err_val in err_list:
        name = query_dict[err_val[0]]
        print err_val[0], err_val[1], err_val[2], name[0], name[1]


def main():
    # test_mlp('e:/test_res/mnist.pkl.gz')
    # filter_errors()
    show_errors()

if __name__ == '__main__':
    main()

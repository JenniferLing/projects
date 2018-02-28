import codecs
import os, re
import numpy as np
import itertools
import pickle
from gensim.models import word2vec
from glove import Corpus, Glove
import gc

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname):
            line = line.strip()
            yield line.split()


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class WordEmbeddings:
    def __init__(self, window, epochs, dim, lrate, no_threads, verbose, directory=None):
        self.window = window
        self.epochs = epochs
        self.dim = dim
        self.lrate = lrate
        self.no_threads = no_threads
        self.verbose = verbose
        self.directory = directory

        self.param_str = 'dim_{0}-epochs_{1}-lrate_{2}-window_{3}'.format(dim, epochs, lrate, window)

    def create_glove_vectors(self, model_prefix):
        if model_prefix and not model_prefix.endswith('-'):
            model_prefix = model_prefix + '-'

        model_name = os.path.join(self.directory, '{0}glove-{1}'.format(model_prefix, self.param_str))

        corpus = Corpus()
        # corpus = Corpus.load('glove_corpus_irony_sarcasm.pkl')
        corpus.fit(self.sents, window=self.window, ignore_missing=True)

        # save dictionary and corpus matrix
        #corpus.save(model_name + '.corpus')
        print('GloVe - vocabulary size: {}'.format(len(corpus.dictionary.keys())))

        model = Glove(no_components=self.dim, learning_rate=self.lrate)
        model.fit(corpus.matrix, epochs=self.epochs, no_threads=self.no_threads, verbose=self.verbose)
        model.add_dictionary(corpus.dictionary)

        # save model
        #model.save(model_name + '.model')

        # save embedding and vocab
        word_vectors, word_dct = self.adopt_vocab(model.word_vectors, model.dictionary)
        np.save(model_name, word_vectors)
        save_obj(word_dct, model_name)

        del model
        del word_vectors
        del word_dct
        gc.collect()

        print('Created GloVe vectors ({})'.format(self.param_str))

    def create_word2vec_vectors(self, model_prefix, type):
        # alpha = (initial) learning rate (default: 0.025)
        # sg = 0 (default) --> CBOW
        # hs = 0 (default) --> negative sampling, else hierarchical softmax
        # negative = 5 (default) --> # noise words
        # cbow_mean = 0 --> use sum of context word vectors, 1 (default) --> use mean
        # iter = # iterations (epochs), default: 5

        if model_prefix and not model_prefix.endswith('-'):
            model_prefix = model_prefix + '-'

        model_name = os.path.join(self.directory, '{0}word2vec_{1}-{2}'.format(model_prefix, type.lower(), self.param_str))

        assert type in ['SG', 'CBOW']
        if type == 'SG':
            sg = 1
        else:
            sg = 0
        model = word2vec.Word2Vec(self.sents,
                                  size=self.dim,
                                  window=self.window,
                                  workers=self.no_threads,
                                  min_count=1,
                                  sg=sg,
                                  iter=self.epochs)

        print('Word2Vec {0} - vocabulary size: {1}'.format(type, len(model.wv.index2word)))

        # save model
        #model.save(model_name + '.model')

        # save embeddings and vocab
        # word_vectors.save_word2vec_format('word2vec.model.bin', fvocab='word2vec.vocab', binary=True)
        word_vectors, word_dct = self.adopt_vocab(model.wv.syn0, model.wv.index2word)
        np.save(model_name, word_vectors)  # save word vectors as numpy array
        save_obj(word_dct, model_name)  # save dictionary word->id

        del model
        del word_vectors
        del word_dct
        gc.collect()

        if type == 'CBOW':
            print('Created CBOW Word2Vec vectors ({})'.format(self.param_str))
        else:
            print('Created SG Word2Vec vectors ({})'.format(self.param_str))

    def adopt_vocab(self, matrix, word_list):
        # # add padding and unknown to vocab (and to matrix)
        # vector = np.zeros((2, matrix.shape[1]))  # alternatively: random initialization
        # padded_matrix = np.concatenate((vector, matrix), axis=0)
        #
        # if isinstance(word_list, list):
        #     vocab = ['<PAD>', '<UNK>']
        #     vocab.extend(word_list)
        #     padded_word_list = {word: index for index, word in enumerate(vocab)}
        #
        # elif isinstance(word_list, dict):
        #     padded_word_list = {word: word_list[word] + 2 for word in word_list.keys()}
        #     padded_word_list['<PAD>'] = 0
        #     padded_word_list['<UNK>'] = 1
        # return padded_matrix, padded_word_list

        if isinstance(word_list, list):
            return matrix, {word: index for index, word in enumerate(word_list)}
        elif isinstance(word_list, dict):
            return matrix, word_list

    def create_word_embeddings(self, sents, prefix=''):

        self.sents = sents

        self.create_glove_vectors(prefix)
        self.create_word2vec_vectors(prefix, 'CBOW')
        self.create_word2vec_vectors(prefix, 'SG')


def main(prefix, sents, dim, window, epochs, output_path):
    params = {
        'window': window,
        'epochs': epochs,
        'dim': dim,
        'lrate': 0.05,  # default Glove
        'no_threads': 4,
        'verbose': False,
    }

    emb = WordEmbeddings(**params, directory=output_path)
    emb.create_word_embeddings(sents, prefix=prefix)


if __name__ == '__main__':

    corpus_path = '/big/l/lingj/corpus/'
    output_path = '/big/l/lingj/embeddings/'
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for domain in ['EMEA', 'hansards', 'EMEA_hansards']:
        domain_path = os.path.join(corpus_path, domain, 'train.lowercased.fr')
        sents = MySentences(domain_path)
        print('Loaded sentences for {}'.format(domain))

        main(domain, sents, 64, 5, 50, output_path)
        main(domain, sents, 64, 10, 50, output_path)
        main(domain, sents, 64, 5, 100, output_path)
        main(domain, sents, 64, 10, 100, output_path)

        main(domain, sents, 128, 5, 50, output_path)
        main(domain, sents, 128, 10, 50, output_path)
        main(domain, sents, 128, 5, 100, output_path)
        main(domain, sents, 128, 10, 100, output_path)

    print('Finished')

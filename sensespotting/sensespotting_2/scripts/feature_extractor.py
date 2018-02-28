#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import gzip
import string
from collections import Counter
import numpy as np
import unicodedata
from gensim import corpora, models
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import pickle
import html
import sys
import re
import shutil
from optparse import OptionParser
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import entropy
import operator
import psutil
import time
import datetime
import scipy
import random
from configurator import Configurator

class FeatureExtraction(Configurator):
    def __init__(self, config_file=None, use_existing=False, verbose=False):
        super().__init__(config_file=config_file, use_existing=use_existing, verbose=verbose)

        self.feat_path = os.path.join(self.working_dir, 'features')
        self.check_dir(self.feat_path)

        self.check_focus_type_file()

        self.check_language_models()
        self.check_topic_model()

        self.psd_classifier_path = os.path.join(self.big_files_path, 'psd_classifier')
        self.check_dir(self.psd_classifier_path)

        self.psd_mapping = {'TOKEN_PSD_GLOBAL': 'global_real', 'TOKEN_PSD_GLOBAL_BINNED': 'global_binned',
                            'TOKEN_PSD_LOCAL': 'local_real', 'TOKEN_PSD_LOCAL_BINNED': 'local_binned',
                            'TOKEN_PSD_RATIO': 'psd_ratio'}

        output_path = os.path.join(self.feat_path, self.new_domain)
        for feature in sorted(self.feature_file_configuration):
            self.feature_file_configuration[feature] = '{0}_{1}'.format(output_path,
                                                                        self.feature_file_configuration[feature])

    #################################################
    ### PREPARE NEEDED FILES                      ###
    #################################################

    def check_focus_type_file(self):
        if self.verbose:
            print('\nCHECK for psd files...')

        for domain in self.domain_paths:
            input_file = self.domain_paths[domain]
            output_file = self.psd_files[domain]

            if not self.check_file(output_file):
                output_file = os.path.join(self.working_dir, self.aux_path, '{}.psd'.format(domain))

                if not os.path.isfile(output_file) or not self.use_existing:
                    if self.verbose:
                        print('- create {}.psd'.format(domain))

                    out = open(output_file, 'w', encoding='utf-8')

                    sent_id = 0
                    with open(input_file, encoding='utf-8') as handle:
                        for line in handle:
                            sent_id += 1
                            line = line.strip().split()

                            word_id = -1
                            max = len(line)
                            while word_id + 1 < max:
                                word_id += 1

                                word = line[word_id]
                                if word in self.frtypes:
                                    out.write('\t'.join(
                                        [str(sent_id), str(word_id), str(word_id), '_', '_', word, '_']) + '\n')

                                if word in [word.split()[0] for word in self.frtypes_phrases]:
                                    start_id = word_id
                                    phrase = word
                                    found = False
                                    j = 0
                                    max_j = np.max([len(word.split()) for word in self.frtypes_phrases])
                                    while not found and word_id + 1 < max and j < max_j:
                                        j += 1
                                        word_id += 1
                                        if line[word_id] in [word.split()[j] for word in self.frtypes_phrases if
                                                             len(word.split()) > j]:
                                            phrase += ' ' + line[word_id]
                                        if phrase in self.frtypes_phrases:
                                            found = True
                                            end_id = word_id
                                    if found:
                                        out.write('\t'.join(
                                            [str(sent_id), str(start_id), str(end_id), '_', '_', phrase, '_']) + '\n')

                    out.close()
                    self.psd_files[domain] = output_file

                else:
                    if self.verbose:
                        print('- use existing {}.psd'.format(domain))
            else:
                if self.verbose:
                    print('- {}.psd was provided'.format(domain))

        return

    def check_language_models(self):
        if self.verbose:
            print('\nCHECK for models (arpa, ngrams, ppl)...')

        order_mapping = {1: 'ug', 3: 'ng'}
        orders = [1, 3]

        for domain in self.domain_paths:
            for order in orders:

                language_model = self.language_models[domain][order_mapping[order]]
                ngrams = language_model.replace('arpa', 'ngrams') if language_model else None
                ppl_model = self.ppl_models[domain][order_mapping[order]]

                if self.check_file(language_model) and self.check_file(ngrams):
                    if self.verbose:
                        print('- {0}.{1}.arpa/ngrams was provided'.format(domain, order))
                else:
                    self.build_language_model(domain, order)

                if self.check_file(ppl_model):
                    if self.verbose:
                        print('- {0}.{1}.ppl was provided'.format(domain, order))
                    self.provided_ppl = True
                else:
                    self.build_perplexity_model(domain, order)
                    self.provided_ppl = False

        self.write_token_prob('ug')
        self.write_token_prob('ng')

        return

    def build_language_model(self, domain, order):

        order_mapping = {1: 'ug', 3: 'ng'}

        output_path = os.path.join(self.models_path, 'language_models')
        self.check_dir(output_path)

        corpus_config = self.corpus_suffix[:-1] if self.corpus_suffix.endswith('.') else self.corpus_suffix
        language_model = os.path.join(output_path,
                                      '{domain}.{lang}.{order}gram.{config}.arpa'.format(domain=domain,
                                                                                         lang=self.source_language,
                                                                                         order=order,
                                                                                         config=corpus_config))
        ngrams = language_model.replace('arpa', 'ngrams')

        self.language_models[domain][order_mapping[order]] = language_model
        if order == 3:
            self.ngrams[domain] = ngrams

        if self.remove_stopwords or self.remove_low_frequency_words:
            corpus = self.domain_paths[domain][:-2] + 'no_placeholder.' + self.domain_paths[domain][-2:]
        else:
            corpus = self.domain_paths[domain]

        if not os.path.isfile(language_model) or not os.path.isfile(ngrams) or not self.use_existing:
            if self.verbose:
                print('- build {0}.{1}.arpa/ngrams'.format(domain, order))

            create_language_model = '{srilm}/bin/i686-m64/ngram-count ' \
                                    '-text {corpus} ' \
                                    '-order {order} -lm {lm} ' \
                                    '-write {ngrams}'.format(srilm=self.srilm_path,
                                                             corpus=corpus,
                                                             order=order,
                                                             lm=language_model,
                                                             ngrams=ngrams)

            subprocess.call(create_language_model.split(), stdout=None)

        else:
            if self.verbose:
                print('- use existing {0}.{1}.arpa/ngrams model'.format(domain, order))

        return

    def build_perplexity_model(self, domain, order):
        order_mapping = {1: 'ug', 3: 'ng'}

        output_path = os.path.join(self.models_path, 'ngram_perplexity', self.new_domain)
        self.check_dir(output_path)

        corpus_config = self.corpus_suffix[:-1] if self.corpus_suffix.endswith('.') else self.corpus_suffix
        ppl_model = os.path.join(output_path,
                                 '{domain}.{lang}.{order}gram.{config}.ppl'.format(domain=domain,
                                                                                   lang=self.source_language,
                                                                                   order=order,
                                                                                   config=corpus_config))
        self.ppl_models[domain][order_mapping[order]] = ppl_model

        if self.remove_stopwords or self.remove_low_frequency_words:
            corpus = self.domain_paths[self.new_domain][:-2] + 'no_placeholder.' + self.domain_paths[self.new_domain][
                                                                                   -2:]
        else:
            corpus = self.domain_paths[self.new_domain]

        if not os.path.isfile(ppl_model) or not self.use_existing:
            if self.verbose:
                print('- build {0}.{1}.ppl'.format(domain, order))

            compute_ngram_perplexity = '{srilm}/bin/i686-m64/ngram -order {order} ' \
                                       '-lm {lm} ' \
                                       '-ppl {corpus} ' \
                                       '-debug 2 '.format(srilm=self.srilm_path,
                                                          corpus=corpus,
                                                          lm=self.language_models[domain][order_mapping[order]],
                                                          lm_domain=domain,
                                                          order=order)

            output_file = open(ppl_model, 'w', encoding='utf-8')
            subprocess.call(compute_ngram_perplexity.split(), stdout=output_file)

        else:
            if self.verbose:
                print('- use existing {0}.{1}.ppl model'.format(domain, order))

    def write_token_prob(self, ngram):
        if self.verbose:
            print('\nEXTRACT probabilities for {}...'.format(ngram))

        for domain in self.domain_paths:
            input_path = self.ppl_models[domain][ngram]
            if not os.path.isfile(input_path):
                self.check_for_language_models()

            output_path = input_path.replace('ppl', 'prob')
            if os.path.isfile(output_path) and not self.provided_ppl and self.use_existing:
                if self.verbose:
                    print('- use existing {}.prob'.format(domain))
            else:
                if self.verbose:
                    print('- create {}.prob'.format(domain))
                out = open(output_path, 'w', encoding='utf-8')

                sent_start = True
                in_sent = False
                sent_id = 0

                with open(input_path, encoding='utf-8') as handle:
                    for line in handle:

                        line = line.strip()
                        if line:
                            # line with sentence
                            if sent_start and not in_sent:
                                if 'file' in line and 'sentences' in line and 'words' in line and 'OOVs' in line:
                                    break
                                else:
                                    sent = line.split()
                                    sent_id += 1
                                    in_sent = True
                                    counter = 0
                                    continue

                            # lines with prob per word of sentence -> extract word and prob and save information
                            if sent_start and in_sent:
                                # word_part, prob_part = [element.strip().split() for element in line.split('=')]
                                # word_part, prob_part = [element.strip().split() for element in line.split('|')]
                                word_part, prob_part = [element.strip().split() for element in line.split('\t')]

                                word = word_part[1]
                                prob = prob_part[4]  # [2]

                                out.write('{0}\t{1}\t{2}\n'.format(sent_id, word, prob))

                                # if ngram == searched_ngram:
                                #     if word in sent:
                                #         out.write('{0},{1},{2}'.format(sent_id, word, prob))
                                # elif word in sent:
                                #     out.write('{0},{1}'.format(word, 0.0))

                                counter += 1
                                if counter >= len(sent) + 1:
                                    in_sent = False
                                    sent_start = False
                                    continue
                        else:  # empty line is between old and new sentence and marks start of the new one
                            sent_start = True
                            continue

                out.close()
        return

    def read_token_prob(self, ngram):
        probs = {}
        for domain in self.domain_paths:
            read_in = self.ppl_models[domain][ngram].replace('ppl', 'prob')

            if not os.path.isfile(read_in):
                self.write_token_prob(ngram)

            probs[domain] = {}
            with open(read_in, encoding='utf-8') as handle:
                for line in handle:
                    sent_id, word, prob = line.strip().split('\t')
                    sent_id = int(sent_id)
                    word = word.strip()

                    if word in self.OOVs or word == '<unk>':
                        word = ''

                    # Solution of paper:
                    prob = float(prob.strip())
                    if prob < -6:
                        prob = -6

                    # Different solution (min_prob = -13.xy )??
                    # if prob == '-inf':
                    #     prob = '-15'
                    # prob = float(prob.strip())

                    if sent_id in probs[domain]:
                        probs[domain][sent_id].append((word, prob))
                    else:
                        probs[domain][sent_id] = [(word, prob)]
        return probs

    def check_topic_model(self):
        """
        use LDA and compute topics on the two domains separately (use 100 topics)
        :return:
        """
        # setting seed to make results reproducable
        if self.verbose:
            print('\nCHECK for topic model (lda, type2id)...')
        np.random.seed(1)

        for domain in self.domain_paths:
            topic_model_file = self.topic_model[domain]['model']
            topic_model_dct = self.topic_model[domain]['dct']

            if self.check_file(topic_model_file) and self.check_file(topic_model_dct):
                try:
                    dct = corpora.Dictionary.load(topic_model_dct)
                    if dct:
                        if self.verbose:
                            print('- model for {} was provided'.format(domain))
                except:
                    if self.verbose:
                        print('- problems while reading topic model dictionary -> create new topic model')
                    self.build_topic_model(domain)

            else:
                self.build_topic_model(domain)

    def build_topic_model(self, domain):

        if self.remove_stopwords or self.remove_low_frequency_words:
            corpus_file = self.domain_paths[domain][:-2] + 'no_placeholder.' + self.domain_paths[domain][-2:]
        else:
            corpus_file = self.domain_paths[domain]

        topic_path = os.path.join(self.models_path, 'topic_models')
        self.check_dir(topic_path)

        corpus_config = self.corpus_suffix[:-1] if self.corpus_suffix.endswith('.') else self.corpus_suffix
        topic_model_file = os.path.join(topic_path, '{0}.{1}.lda'.format(domain, corpus_config))
        topic_model_dct = os.path.join(topic_path, '{0}.{1}.type2id'.format(domain, corpus_config))

        if not os.path.isfile(topic_model_file) or not os.path.isfile(topic_model_dct) or not self.use_existing:
            if self.verbose:
                print('- build topic model for {}'.format(domain))
            dictionary = corpora.Dictionary(line.lower().split() for line in open(corpus_file, encoding='utf-8'))
            dictionary.save(topic_model_dct)

            corpus_generator = TopicCorpus(dictionary, corpus_file)
            nb_topics = 100
            # lda_model = models.LdaModel(corpus_generator, id2word=dictionary, num_topics=nb_topics,
            #                             update_every=1, chunksize=10000, passes=1, minimum_probability=0.0)
            lda_model = models.LdaMulticore(corpus_generator, id2word=dictionary, num_topics=nb_topics,
                                            chunksize=10000, passes=10, minimum_probability=0.0,
                                            workers=10)

            lda_model.save(topic_model_file)

        else:
            if self.verbose:
                print('- use existing {} topic model'.format(domain))

        self.topic_model[domain]['model'] = topic_model_file
        self.topic_model[domain]['dct'] = topic_model_dct

    #################################################
    ### FEATURE EXTRACTION                        ###
    #################################################

    def get_type_rel_freq_features(self, return_value=False):
        """
        computes unigram log probabilities (via smoothed relative frequencies) of each word
        under consideration in the old and new domain.
        Return the two log probabilities and their difference as features for each word type
        :param filename: name of file to save features as pickle object
        :return: features[word_type] = (log_prob_old, log_prob_new, diff_log_prob)
        """
        output_path = self.feature_file_configuration['TYPE_REL_FREQ']

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load type relative frequency feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- type relative frequency feature file already exists')
                features = {}

        else:
            if self.verbose:
                print('- compute type relative frequency feature values')

            types = {word: {self.new_domain: -20} for word in self.frtypes}

            # read language model
            for domain in self.domain_paths:
                input_path = self.language_models[domain]['ug']

                if not os.path.isfile(input_path):
                    self.check_for_language_models()

                with open(input_path, encoding='utf-8') as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            try:
                                prob, word = line.split('\t')
                            except:
                                continue

                            if word not in self.frtypes:
                                continue

                            types[word][domain] = float(prob)

            # compute features
            features = {}
            for word_type in types.keys():
                log_prob_new_ug = types[word_type][self.new_domain]

                if self.old_domain in types[word_type].keys():
                    log_prob_old_ug = types[word_type][self.old_domain]
                else:
                    log_prob_old_ug = -20

                log_prob_diff_ug = log_prob_new_ug - log_prob_old_ug
                # features[word_type] = (log_prob_old_ug, log_prob_new_ug, log_prob_diff_ug)
                features[word_type] = {'lpOld': log_prob_old_ug,
                                       'lpNew': log_prob_new_ug,
                                       'lpDiff': log_prob_diff_ug}

            assert self.frtypes.issubset(set(features.keys()))

            self.write_type_features_to_file(output_path, features)
            self.save_pickle_obj(features, output_path)

        # key = list(features.keys())[0]
        # print('Example entry for "{0}": {1}'.format(key, features[key]))
        # print('Feature entries: {}'.format(len(features)))

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def get_type_ngram_prob_features(self, return_value=False):
        """
        log probability under consideration given its N-gram (ng) context
        (using old-domain (old) and new-domain (new) language models)
        and respective unigram (ug) log probabilities
        Used trigram models. => N = 3
        Features:
        (1) lm_new_ng
        (2) lm_new_ng - lm_old_ng
        (3) lm_new_ng - lm_new_ug
        (4) lm_new_ng - lm_new_ug + lm_old_ug - lm_old_ng
        Compute statistics out of this four values over the monolingual text:
        - Mean
        - Std
        - Min and Max value
        - Sum
        :param filename: name of file to save features as pickle object
        :return: features[language] = (mean, std, min, max, sum)
        """

        output_path = self.feature_file_configuration['TYPE_NGRAM_PROB']

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load type ngram probability feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- type ngram probability feature file already exists')
                features = {}

        else:
            if self.verbose:
                print('- compute type ngram probability feature values')

            ug_probs = self.read_token_prob('ug')
            ng_probs = self.read_token_prob('ng')

            statistics = {}
            problematic_terms = []
            sent_id = 0

            input_file = self.domain_paths[self.new_domain]
            if self.remove_stopwords or self.remove_low_frequency_words:
                input_file = input_file[:-2] + 'no_placeholder.' + input_file[-2:]

            with open(input_file, encoding='utf-8') as handle:
                for line in handle:
                    sent_id += 1
                    words = line.strip().split()

                    probs = {'gb': ng_probs[self.old_domain][sent_id],
                             'db': ng_probs[self.new_domain][sent_id],
                             'gs': ug_probs[self.old_domain][sent_id],
                             'ds': ug_probs[self.new_domain][sent_id]}

                    for i in range(len(words)):
                        word = words[i]

                        if word not in self.frtypes:
                            continue

                        if word not in statistics:
                            statistics[word] = {'gb': [], 'gb_gs': [],
                                                'db_gb': [], 'dbgs_gbds': []}

                        statistics[word]['gb'].append(
                            probs['gb'][i][1])
                        statistics[word]['gb_gs'].append(
                            probs['gb'][i][1] - probs['gs'][i][1])
                        statistics[word]['db_gb'].append(
                            probs['db'][i][1] - probs['gb'][i][1])
                        statistics[word]['dbgs_gbds'].append(
                            probs['db'][i][1] + probs['gs'][i][1]
                            - probs['gb'][i][1] - probs['ds'][i][1])

            def get_statistics(values):
                return {'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'sum': np.sum(values),
                        # 'cnt': len(values)
                        }

            features = {}
            for word in statistics:
                features[word] = {}
                for stat in statistics[word]:
                    word_feature_statistics = get_statistics(statistics[word][stat])
                    for name, value in sorted(word_feature_statistics.items()):
                        features[word]['{0}-{1}'.format(stat, name)] = value

            print(set(features.keys()) - self.frtypes)
            print(self.frtypes - set(features.keys()))
            assert self.frtypes.issubset(set(features.keys()))
            self.save_pickle_obj(features, output_path)
            self.write_type_features_to_file(output_path, features)

            # if problematic_terms:
            #     if self.verbose:
            #         print('\tDEBUG: Problematic_terms:')
            #         for ppl_term, psd_term in set(problematic_terms):
            #             print('\t\tppl_file term: {0}, psd_file term: {1}'.format(ppl_term, psd_term))

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def get_type_context_features(self, return_value=False):
        """
        For each word type w compute:
        - N_w = number of n-grams (unigrams, bigrams, trigrams) of the new domain containing w
        - O_w = number of n-grams of the old domain containing w
        => compute |N_w - O_w| / |N_w|
        (not count n-grams containing OOVs)
        :param filename: name of file to save features as pickle object
        :return:
        """

        output_path = self.feature_file_configuration['TYPE_CONTEXT']

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load type context feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- type context feature file already exists')
                features = {}

        else:
            if self.verbose:
                print('- compute type context feature values')
            features = {}
            types = {}

            def contains_OVVs(ngram):
                OOV_ngram = False
                for word_type in ngram:
                    if word_type in self.OOVs:
                        OOV_ngram = True
                        return OOV_ngram

                return OOV_ngram

            for domain in self.domain_paths:

                input_path = self.ngrams[domain]

                if not self.check_file(input_path):
                    self.check_language_models()

                with open(input_path, encoding='utf-8') as handle:
                    for line in handle:
                        line = line.strip()
                        ngram = line.split('\t')[0].split()

                        # if ngram contains OOVs, it is not considered
                        if contains_OVVs(ngram):
                            continue

                        for word_type in ngram:

                            if word_type not in types:
                                types[word_type] = {self.old_domain: set(), self.new_domain: set()}

                            types[word_type][domain].add(tuple(ngram))

            for word_type in types:
                N_w = types[word_type][self.new_domain]
                O_w = types[word_type][self.old_domain]

                if not N_w:
                    features[word_type] = 0.0
                    continue

                if not O_w:
                    features[word_type] = 1.0
                    continue

                # if word_type occurs in both domains; otherwise corresponding set will be empty
                if N_w and O_w:
                    features[word_type] = {'ngram_overlap': len(N_w - O_w) / len(N_w)}
                    continue

            self.save_pickle_obj(features, output_path)
            self.write_type_features_to_file(output_path, features)

        # key = list(features.keys())[0]
        # print('Example entry for "{0}": {1}'.format(key, features[key]))
        # print('Feature entries: {}'.format(len(features)))

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def get_type_topic_features(self, return_value=False):
        """
        For all words w
        - T_o = set of old topics (topics of old domain)
        - T_n = set of new topics (topics of new domain)
        - P_o = old topic distribution
        - P_n = new topic distribution
        - for each pair of old topic t' and new topic t: topic_sim = P_n(t|w) * P_o(t'|w) * cos(t,t')
        use LDA and compute topics on the two domains separately (use 100 topics)
        :param filename:
        :return:
        """
        output_path = self.feature_file_configuration['TYPE_TOPIC']

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load type topic feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- type topic feature file already exists')
                features = {}

        else:
            if self.verbose:
                print('- compute type topic feature values')

            new_domain_topic_path = self.topic_model[self.new_domain]
            old_domain_topic_path = self.topic_model[self.old_domain]

            if not os.path.isfile(new_domain_topic_path['model']) or not os.path.isfile(old_domain_topic_path['model']):
                self.get_topic_model()

            new_domain_dictionary = corpora.Dictionary.load(new_domain_topic_path['dct'])
            old_domain_dictionary = corpora.Dictionary.load(old_domain_topic_path['dct'])

            new_domain_lda_model = models.LdaModel.load(new_domain_topic_path['model'])
            old_domain_lda_model = models.LdaModel.load(old_domain_topic_path['model'])

            T_o_path = old_domain_topic_path['model'][:-4] + '_' + self.new_domain + '.{}topics'.format(
                self.corpus_suffix)
            if os.path.isfile(T_o_path + '.pkl'):
                if self.verbose:
                    print('\t- use existing {}'.format(T_o_path))
                T_o = self.load_pickle_obj(T_o_path)
            else:
                if self.verbose:
                    print('\t- create {}'.format(T_o_path))
                T_o = {topic: [prob for word, prob in sorted(vector) if word in self.common_vocab]
                       for topic, vector in old_domain_lda_model.show_topics(num_topics=-1,
                                                                             num_words=len(old_domain_dictionary),
                                                                             formatted=False)}
                self.save_pickle_obj(T_o, T_o_path)

            T_n_path = new_domain_topic_path['model'][:-4] + '_' + self.old_domain + '.{}topics'.format(
                self.corpus_suffix)
            if os.path.isfile(T_n_path + '.pkl'):
                if self.verbose:
                    print('\t- use existing {}'.format(T_n_path))
                T_n = self.load_pickle_obj(T_n_path)
            else:
                if self.verbose:
                    print('\t- create {}'.format(T_n_path))
                T_n = {topic: [prob for word, prob in sorted(vector) if word in self.common_vocab]
                       for topic, vector in new_domain_lda_model.show_topics(num_topics=-1,
                                                                             num_words=len(new_domain_dictionary),
                                                                             formatted=False)}
                self.save_pickle_obj(T_n, T_n_path)

            assert len(T_o) == 100 and len(T_n) == 100

            features = {}

            for word_type in self.common_vocab:

                features[word_type] = {'topic_sim_sense': 0}

                for t_n in range(len(T_n)):
                    for t_o in range(len(T_o)):
                        new_topic_distribution = {topic_id: prob for topic_id, prob in
                                                  new_domain_lda_model.get_term_topics(
                                                      new_domain_dictionary.token2id[word_type],
                                                      minimum_probability=0.0) if topic_id == t_n}
                        old_topic_distribution = {topic_id: prob for topic_id, prob in
                                                  old_domain_lda_model.get_term_topics(
                                                      old_domain_dictionary.token2id[word_type],
                                                      minimum_probability=0.0) if topic_id == t_o}

                        p_new = new_topic_distribution[t_n] if new_topic_distribution else 0
                        p_old = old_topic_distribution[t_o] if old_topic_distribution else 0

                        if p_new and p_old:
                            cos = cosine_similarity(np.array(T_n[t_n]).reshape(1, -1),
                                                    np.array(T_o[t_o]).reshape(1, -1)).flatten()[-1]
                            topic_sim = p_new * p_old * cos
                            features[word_type]['topic_sim_sense'] += topic_sim

            self.save_pickle_obj(features, output_path)
            self.write_type_features_to_file(output_path, features)

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def get_token_ngram_prob_features(self, return_value=False):

        output_path = self.feature_file_configuration['TOKEN_NGRAM_PROB']

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load token ngram prob feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- token ngram prob feature file already exists')
                features = []

        else:
            if self.verbose:
                print('- compute token ngram probability feature values')

            features = []

            data, last_sent_id, psd_words = self.read_psd_file(self.new_domain)

            probs = self.read_token_prob('ng')
            if self.remove_stopwords or self.remove_low_frequency_words:
                mapping = self.load_pickle_obj(os.path.join(self.corpus_path, self.new_domain,
                                                            '{0}.{1}corpus_mapping'.format(self.new_domain,
                                                                                           self.corpus_suffix)))
            # sent_id starts with 1, position ids with 0
            for current_sent_id in range(1, last_sent_id + 1):

                if current_sent_id not in data:
                    continue

                old_domain_sent = probs[self.old_domain][current_sent_id]  # GEN
                new_domain_sent = probs[self.new_domain][current_sent_id]  # DOM

                # print([word for word, prob in old_domain_sent])
                # print( [word for word, prob in new_domain_sent])
                # print(current_sent_id)
                assert len([word for word, prob in old_domain_sent]) == len([word for word, prob in new_domain_sent])

                for line_nb, start, end in sorted(data[current_sent_id]):
                    word_features = {}

                    if self.lemmatize:
                        source_language_phrase = self.fr_types_mapping[line_nb]
                    else:
                        source_language_phrase, _ = data[current_sent_id][
                            (line_nb, start, end)]

                    if self.remove_stopwords or self.remove_low_frequency_words:
                        prob_start_id = mapping[current_sent_id][start]
                        prob_end_id = mapping[current_sent_id][end]
                    else:
                        prob_start_id = start
                        prob_end_id = end
                    word_phrase, word_prob = probs[self.old_domain][current_sent_id][prob_start_id]
                    old_word_phrase = word_phrase

                    if start == end:
                        assert word_phrase == source_language_phrase
                        # if word_phrase != source_language_phrase:
                        #     print(current_sent_id)
                        #     print(word_phrase)
                        #     print(source_language_phrase)
                        #     print(line_nb)
                        #     print(data[current_sent_id])
                        #     print(mapping[current_sent_id])
                        #     print(start, end)
                        #     exit()

                    # ToDo: handle multi-word phrases
                    # for idx in range(source_language_token_start + 1, source_language_token_end + 1):
                    #     word_phrase += ' ' + probs[self.old_domain][current_sent_id][idx][0]
                    #
                    # # assert word_phrase == source_language_phrase
                    # if word_phrase != source_language_phrase:
                    #     print('ERROR:')
                    #     print(current_sent_id)
                    #     print('psd_file_word: {0}\nprob_file_word: {1}'.format(source_language_phrase, old_word_phrase))
                    #     print('adapted prob_file_word: {0}'.format(word_phrase))
                    #     raise ValueError

                    old_domain_start = probs[self.old_domain][current_sent_id][prob_start_id][1]
                    new_domain_start = probs[self.new_domain][current_sent_id][prob_start_id][1]
                    old_domain_end = probs[self.old_domain][current_sent_id][prob_end_id][1]
                    new_domain_end = probs[self.new_domain][current_sent_id][prob_end_id][1]
                    diff_start = old_domain_start - new_domain_start
                    diff_end = old_domain_end - new_domain_end

                    word_features['gen_st'] = old_domain_start
                    word_features['dom_st'] = new_domain_start
                    word_features['diff_st'] = diff_start

                    word_features['gen_en'] = old_domain_end
                    word_features['dom_en'] = new_domain_end
                    word_features['diff_en'] = diff_end

                    features.append(word_features)

            self.save_pickle_obj(features, output_path)
            self.write_token_features_to_file(output_path, features)

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def get_token_context_features(self, return_value=False):

        if self.feature_extraction_configuration['TOKEN_CONTEXT'] or \
                (self.feature_extraction_configuration['TOKEN_CONTEXT_COUNT'] and \
                         self.feature_extraction_configuration['TOKEN_CONTEXT_PERCENTAGE']):
            token_count = True
            token_percentage = True
            output_path = self.feature_file_configuration['TOKEN_CONTEXT']

            write_all = True

        elif not self.feature_extraction_configuration['TOKEN_CONTEXT_COUNT'] and self.feature_extraction_configuration[
            'TOKEN_CONTEXT_PERCENTAGE']:
            token_count = False
            token_percentage = True
            output_path = self.feature_file_configuration['TOKEN_CONTEXT_PERCENTAGE']

        elif self.feature_extraction_configuration['TOKEN_CONTEXT_COUNT'] and not self.feature_extraction_configuration[
            'TOKEN_CONTEXT_PERCENTAGE']:
            token_count = True
            token_percentage = False
            output_path = self.feature_file_configuration['TOKEN_CONTEXT_COUNT']

        else:
            raise ValueError('Context features are not configured to be extracted!')

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load token context feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- token context feature file already exists')
                features = []

        else:
            if self.verbose:
                print('- compute token context feature values')

            data, last_sent_id, psd_words = self.read_psd_file(self.new_domain)

            distances = {-2: 'llcont', -1: 'lcont', 1: 'rcont', 2: 'rrcont'}
            not_found_value = -1  # previously 0
            features = []

            old_domain_context_count_path = os.path.join(self.aux_path,
                                                         '{0}_token_context_count.{1}'.format(self.old_domain,
                                                                                              self.corpus_suffix[:-1]))

            if os.path.isfile(old_domain_context_count_path + '.pkl'):
                old_domain_context_count = self.load_pickle_obj(old_domain_context_count_path)

            else:
                ## get context words of old domain
                old_domain_context_count = {w_i: {dist: {} for dist in distances} for w_i in psd_words}

                read_old_domain = self.domain_paths[self.old_domain]
                if self.remove_stopwords or self.remove_low_frequency_words:
                    read_old_domain = '{0}no_placeholder.{1}'.format(read_old_domain[:-2], read_old_domain[-2:])

                with open(read_old_domain, encoding='utf-8') as handle:
                    for line in handle:
                        line = line.strip().split()
                        for i in range(2, len(line) - 2):

                            if line[i] not in old_domain_context_count:
                                continue

                            for distance in distances:
                                context_word = line[i + distance]
                                old_domain_context_count[line[i]][distance][context_word] = \
                                    old_domain_context_count[line[i]][distance].get(
                                        context_word, 0) + 1

                self.save_pickle_obj(old_domain_context_count, old_domain_context_count_path)

            read_new_domain = self.domain_paths[self.new_domain]
            if self.remove_stopwords or self.remove_low_frequency_words:
                read_new_domain = '{0}no_placeholder.{1}'.format(read_new_domain[:-2], read_new_domain[-2:])
                mapping = self.load_pickle_obj(os.path.join(self.corpus_path, self.new_domain,
                                                            '{0}.{1}corpus_mapping'.format(self.new_domain,
                                                                                           self.corpus_suffix)))

            sent_id = 0
            with open(read_new_domain, encoding='utf-8') as handle:
                for line in handle:
                    sent_id += 1

                    if sent_id not in data:
                        continue

                    line = line.strip().split()
                    for line_nb, start, end in sorted(data[sent_id]):
                        token_count_percent = 0
                        tmp = {}

                        word, _ = data[sent_id][(line_nb, start, end)]

                        if self.remove_stopwords or self.remove_low_frequency_words:
                            try:
                                corpus_start = mapping[sent_id][start]
                            except:
                                print(line)
                                print(len(line))
                                print(sent_id)
                                print(word)
                                print(data[sent_id])
                                print(mapping[sent_id])
                                print(line_nb, start, end)
                                raise KeyError
                        else:
                            corpus_start = start

                        for distance in distances:
                            # multi-word phrase -> Handle differently?
                            if start != end:
                                tmp[distances[distance]] = 0.0
                            try:
                                assert corpus_start + distance >= 0
                                context_word = line[corpus_start + distance]
                            except (IndexError, AssertionError):
                                tmp[distances[distance]] = not_found_value
                                continue

                            if context_word in old_domain_context_count[word][distance]:
                                token_count_percent += 1

                            if token_count:
                                tmp[distances[distance]] = old_domain_context_count[word][distance].get(context_word,
                                                                                                        not_found_value)
                                if tmp[distances[distance]] > 0:
                                    tmp[distances[distance]] = np.log10(tmp[distances[distance]])

                        if token_count:
                            assert len(tmp) == len(distances)

                        if token_percentage:
                            tmp['perccont'] = token_count_percent / 4.0
                        features.append(tmp)

            self.save_pickle_obj(features, output_path)

            if write_all:
                self.write_context_token_features_to_file(features)
            else:
                self.write_token_features_to_file(features, output_path)

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def get_token_psd_features(self, return_value=False):
        test_filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

        # global_real = self.feature_extraction_configuration['TOKEN_PSD_GLOBAL']
        # global_binned = self.feature_extraction_configuration['TOKEN_PSD_GLOBAL_BINNED']
        # local_real = self.feature_extraction_configuration['TOKEN_PSD_LOCAL']
        # local_binned = self.feature_extraction_configuration['TOKEN_PSD_LOCAL_BINNED']
        # psd_ratio = self.feature_extraction_configuration['TOKEN_PSD_RATIO']
        global_real = 1
        global_binned = 1
        local_real = 1
        local_binned = 1
        psd_ratio = 1

        output_path = self.feature_file_configuration['TOKEN_PSD']

        if os.path.isfile(output_path + '.pkl'):
            if return_value:
                if self.verbose:
                    print('- load global and local psd feature values')
                features = self.load_pickle_obj(output_path)
            else:
                if self.verbose:
                    print('- psd feature file already exists')
                features = []
                for feat_name, value in self.feature_extraction_configuration.items():
                    if not value:
                        continue

                    if not self.check_file(self.feature_file_configuration[feat_name]):
                        self.derive_features_containing(output_path,
                                                        self.feature_file_configuration[feat_name],
                                                        self.psd_mapping[feat_name])

        else:

            features = []

            global_models_path = os.path.join(self.psd_classifier_path, 'global_models')
            global_models = os.listdir(global_models_path)
            fnames = [int(fname.split('_')[-1][:-3]) for fname in global_models if fname.startswith('global_')]
            best_global_model = 'global_{}.vw'.format(max(fnames))
            assert best_global_model in global_models

            model_mapping = {'global': os.path.join(global_models_path, best_global_model),
                             'local': os.path.join(self.psd_classifier_path, 'local_models',
                                                   'local_features')}

            local_mapping = self.load_pickle_obj(os.path.join(self.psd_classifier_path,
                                                              'vw.local.local_models.clfname_nb_mapping'))

            if self.verbose:
                print('- extract global and local psd feature values')

            data, last_sent_id, psd_words = self.read_psd_file(self.new_domain)

            tagging_path = self.tagged_corpus_file[self.new_domain][self.source_language]
            tagged_corpus = self.yield_pos_tags(tagging_path)

            read_new_domain = self.domain_paths[self.new_domain]
            if self.remove_stopwords or self.remove_low_frequency_words:
                corpus_mapping = self.load_pickle_obj(os.path.join(self.corpus_path, self.new_domain,
                                                                   '{0}.{1}corpus_mapping'.format(self.new_domain,
                                                                                                  self.corpus_suffix)))
                read_new_domain = '{0}no_placeholder.{1}'.format(read_new_domain[:-2], read_new_domain[-2:])

            sent_id = 0

            with open(read_new_domain, encoding='utf-8') as handle:
                for line in handle:
                    sent_id += 1

                    sentence = line.strip().split()
                    tagged_sentence = next(tagged_corpus)

                    if sent_id not in data:
                        continue

                    if self.remove_stopwords or self.remove_low_frequency_words:
                        tagged_sentence = [tagged_sentence[i] for i in range(len(tagged_sentence))
                                           if corpus_mapping[sent_id][i] != -1]

                    if self.lemmatize:
                        sentence = [tagged_sentence[i]['word'] for i in range(len(sentence))]

                    for line_nb, start, end in sorted(data[sent_id]):
                        if self.verbose:
                            if line_nb % 1000 == 0:
                                print('\t\t- tested {} instances'.format(line_nb))

                        # ToDo: handle multi-word phrases
                        if start != end:
                            features.append('<ignore>')
                            # prior_out.write('|\n')
                            # posterior_out.write('|\n')
                            continue

                        source_phrase, _ = data[sent_id][(line_nb, start, end)]

                        if self.remove_stopwords or self.remove_low_frequency_words:
                            corpus_start = corpus_mapping[sent_id][start]
                        else:
                            corpus_start = start

                        assert source_phrase == sentence[corpus_start]
                        assert source_phrase == tagged_sentence[corpus_start]['word']

                        X_posterior = self.get_word_features(corpus_start, source_phrase, sentence,
                                                             tagged_sentence)

                        X_prior = {'word': X_posterior['current_word'],
                                   'lemma': X_posterior['current_lemma'],
                                   'POS': X_posterior['current_POS']}

                        posterior_feature_line = '|{}'.format(self.get_vw_feature_line(X_posterior, namespace='FEAT'))
                        prior_feature_line = '|{}'.format(self.get_vw_feature_line(X_prior, namespace='FEAT'))

                        if global_real or global_binned:
                            # prediction on global model
                            global_distribution = self.test_vw(posterior_feature_line, model_mapping['global'],
                                                               filename=test_filename)

                            this_token_features = {}
                            global_psd_features = self.extract_features_from_prob_distribution(global_distribution)

                            if global_real:
                                for name, value in global_psd_features.items():
                                    this_token_features['global_real_{}'.format(name)] = value

                            if global_binned:
                                for name, value in self.get_binned_psd_features(global_psd_features).items():
                                    this_token_features['global_binned_{}'.format(name)] = value

                        if local_real or local_binned or psd_ratio:
                            # prediction on local model with posterior features
                            model_path = '{0}_{1}.train.shuf.vw'.format(model_mapping['local'],
                                                                        local_mapping[source_phrase])
                            posterior_local_distribution = self.test_vw(posterior_feature_line, model_path,
                                                                        filename=test_filename)
                            # print('Distribution: {}'.format(posterior_local_distribution))
                            # print(sum(posterior_local_distribution))


                            # prediction on local model with prior features
                            model_path = '{0}_{1}.train.shuf.vw'.format(model_mapping['local'],
                                                                        local_mapping[source_phrase])
                            prior_local_distribution = self.test_vw(prior_feature_line, model_path,
                                                                    filename=test_filename)

                            if source_phrase in local_mapping:

                                posterior_local_psd_features = self.extract_features_from_prob_distribution(
                                    posterior_local_distribution)

                                if local_real:
                                    for name, value in posterior_local_psd_features.items():
                                        this_token_features['local_real_{}'.format(name)] = value

                                if local_binned:
                                    for name, value in self.get_binned_psd_features(
                                            (posterior_local_psd_features)).items():
                                        this_token_features['local_binned_{}'.format(name)] = value

                                if psd_ratio:
                                    psd_ratio_features = self.compute_psd_ratio_features(prior_local_distribution,
                                                                                         posterior_local_distribution)
                                    for name, value in psd_ratio_features.items():
                                        this_token_features['psd_ratio_{}'.format(name)] = value

                            else:
                                if self.verbose:
                                    print('- PROBLEM: {} not in local mapping!'.format(source_phrase))
                                else:
                                    print('ERROR: {} not in local mapping!'.format(source_phrase))

                        features.append(this_token_features)

            if self.verbose:
                print('- number of features: {}'.format(len(features)))

            self.save_pickle_obj(features, output_path)
            self.write_token_features_to_file(features, output_path)

            token_psd_value = self.feature_extraction_configuration['TOKEN_PSD']
            assert self.check_file(output_path)
            for feat_name, value in self.feature_extraction_configuration.items():
                if not token_psd_value and not value:
                    continue

                if not self.check_file(self.feature_file_configuration[feat_name]):
                    self.derive_features_containing(output_path,
                                                    self.feature_file_configuration[feat_name],
                                                    self.psd_mapping[feat_name])

        if return_value:
            return features
        else:
            del features
            gc.collect()

        return

    def extract_features(self):
        if self.verbose:
            print('\nEXTRACT features...')

        if self.feature_extraction_configuration['TYPE_REL_FREQ']:
            if not self.check_file(self.feature_file_configuration['TYPE_REL_FREQ']):
                self.get_type_rel_freq_features()
            else:
                if self.verbose:
                    print('- use existing type relative frequency feature values')

        if self.feature_extraction_configuration['TYPE_NGRAM_PROB']:
            if not self.check_file(self.feature_file_configuration['TYPE_NGRAM_PROB']):
                self.get_type_ngram_prob_features()
            else:
                if self.verbose:
                    print('- use existing type ngram prob feature values')

        if self.feature_extraction_configuration['TYPE_CONTEXT']:
            if not self.check_file(self.feature_file_configuration['TYPE_CONTEXT']):
                self.get_type_context_features()
            else:
                if self.verbose:
                    print('- use existing type context feature values')

        if self.feature_extraction_configuration['TYPE_TOPIC']:
            if not self.check_file(self.feature_file_configuration['TYPE_TOPIC']):
                self.get_type_topic_features()
            else:
                if self.verbose:
                    print('- use existing type topic feature values')

        if self.feature_extraction_configuration['TOKEN_CONTEXT']:
            if not self.check_file(self.feature_file_configuration['TOKEN_CONTEXT']):
                self.get_token_context_features()
            else:
                if self.verbose:
                    print('- use existing token context feature values')
                if self.feature_extraction_configuration['TOKEN_CONTEXT_COUNT'] and \
                        not self.check_file(self.feature_file_configuration['TOKEN_CONTEXT_COUNT']):
                    if self.verbose:
                        print('- write token context count feature values in separate file')
                    self.derive_features_not_containing(self.feature_file_configuration['TOKEN_CONTEXT'],
                                                        self.feature_file_configuration['TOKEN_CONTEXT_COUNT'],
                                                        'perccont')

                if self.feature_extraction_configuration['TOKEN_CONTEXT_PERCENTAGE'] and \
                        not self.check_file(self.feature_file_configuration['TOKEN_CONTEXT_PERCENTAGE']):
                    if self.verbose:
                        print('- write token context percentage feature values in separate file')
                    self.derive_features_containing(self.feature_file_configuration['TOKEN_CONTEXT'],
                                                    self.feature_file_configuration['TOKEN_CONTEXT_PERCENTAGE'],
                                                    'perccont')

        if self.feature_extraction_configuration['TOKEN_NGRAM_PROB']:
            if not self.check_file(self.feature_file_configuration['TOKEN_NGRAM_PROB']):
                self.get_token_ngram_prob_features()
            else:
                if self.verbose:
                    print('- use existing token ngram prob feature values')

        if self.feature_extraction_configuration['TOKEN_PSD']:
            if not self.check_file(self.feature_file_configuration['TOKEN_PSD']):
                self.get_token_psd_features()
            else:
                if self.verbose:
                    print('- use existing token psd feature values')
                for feat_name, value in self.feature_extraction_configuration.items():
                    if not value:
                        continue

                    if not self.check_file(self.feature_file_configuration[feat_name]):
                        if self.verbose:
                            print('- write {} feature values in separate file'.format(
                                feat_name.lower().replace('_', ' ')))
                        self.derive_features_containing(self.feature_file_configuration['TOKEN_PSD'],
                                                        self.feature_file_configuration[feat_name],
                                                        self.psd_mapping[feat_name])

        return

        #################################################

    #################################################
    ### ADDITIONAL METHODS                        ###
    #################################################

    def extract_features_from_prob_distribution(self, prob_distribution):

        max_prob = max(prob_distribution)
        prob_entropy = entropy(prob_distribution, qk=None) / np.log(len(prob_distribution))
        if prob_entropy == np.nan:
            prop_entropy = -1

        prob_spread = max_prob - min(prob_distribution)

        if max_prob == 0.0:
            prob_confusion = 0
        else:
            prob_confusion = np.median(prob_distribution) / max_prob

        return {'MaxProb': max_prob, 'Entropy': prob_entropy, 'Spread': prob_spread, 'Confusion': prob_confusion}

    def compute_psd_ratio_features(self, prior, posterior):
        features = {}

        prior_min_index, prior_min_value = min(enumerate(prior), key=operator.itemgetter(1))
        prior_max_index, prior_max_value = max(enumerate(prior), key=operator.itemgetter(1))

        posterior_min_index, posterior_min_value = min(enumerate(posterior), key=operator.itemgetter(1))
        posterior_max_index, posterior_max_value = max(enumerate(posterior), key=operator.itemgetter(1))

        features['SameMax'] = 1 if prior_max_index == posterior_max_index else 0
        features['SameMin'] = 1 if prior_min_index == posterior_min_index else 0
        features['X-OR_MinMax'] = 1 if features['SameMin'] != features['SameMax'] else 0

        kl1 = entropy(prior, posterior)
        if kl1 == np.inf:
            features['KL_prior_given_posterior'] = -1
        else:
            features['KL_prior_given_posterior'] = kl1

        kl2 = entropy(posterior, prior)
        if kl2 == np.inf:
            features['KL_posterior_given_prior'] = -1
        else:
            features['KL_posterior_given_prior'] = kl2

        if posterior_max_value == 0.0:
            features['MaxNorm'] = 0
        else:
            features['MaxNorm'] = prior_max_value / float(posterior_max_value)

        spread_prior = prior_max_value - prior_min_value
        spread_posterior = posterior_max_value - posterior_min_value
        if spread_posterior == 0.0:
            features['SpreadNorm'] = 0
        else:
            features['SpreadNorm'] = spread_prior / float(spread_posterior)

        confusion_prior = np.median(prior) / float(prior_max_value)
        confusion_posterior = np.median(posterior) / float(posterior_max_value)
        if confusion_posterior == 0.0:
            features['ConfusionNorm'] = 0
        else:
            features['ConfusionNorm'] = confusion_prior / float(confusion_posterior)

        return features

    def get_binned_psd_features(self, real_features):
        bins = {'Confusion-le': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                'Entropy-le': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
                'MaxProb-gt': [-1e-05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                               0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                'Spread-gt': [-1e-05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                              0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]}

        binned_psd_features = {}

        for key in bins:
            feature_name, relation = key.split('-')
            if relation == 'le':
                binned_psd_features[key] = [value for value in bins[key] if real_features[feature_name] <= value]
            elif relation == 'gt':
                binned_psd_features[key] = [value for value in bins[key] if real_features[feature_name] > value]

        return binned_psd_features

    def test_vw(self, feature_line, model_path, filename='temp'):
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}.txt'.format(filename))

        with open(test_path, mode='w', encoding='utf-8') as out:
            out.write(feature_line + '\n')

        assert self.check_file(test_path)
        assert self.check_file(model_path)

        command = 'sh {0}/test_vw_model.sh {1} {2} {3} {4} {5}'.format(self.script_path,
                                                                       self.vowpal_wabbit_path,
                                                                       model_path,
                                                                       test_path,
                                                                       "--probabilities",
                                                                       "_")

        probs = subprocess.Popen(command.split(), stdout=subprocess.PIPE).communicate()[0]
        probs = str(probs, 'utf-8').split()

        out = [float(prob.split(':')[1]) for prob in probs]

        return out

    def write_type_features_to_file(self, output_path, features):
        assert isinstance(features, dict)

        data, last_sent_id, source_words = self.read_psd_file(self.new_domain)
        with open(output_path, mode='w', encoding='utf-8') as out:
            for sent_id in range(1, last_sent_id + 1):

                if sent_id not in data:
                    continue

                for line_nb, start, end in sorted(data[sent_id]):
                    if self.lemmatize:
                        source_word = self.fr_types_mapping[line_nb]
                        if source_word not in self.frtypes:
                            print(set(self.fr_types_mapping.values()) == self.frtypes)
                            print(self.frtypes)
                            print(data[sent_id][(line_nb, start, end)])
                            print(source_word)
                            exit()
                    else:
                        source_word, _ = data[sent_id][(line_nb, start, end)]

                    print(line_nb, start, end, source_word)
                    print(source_word in self.frtypes)
                    print(source_word in features)

                    if source_word == 'vue des enfants' or source_word == 'vue du enfant':
                        out.write('\n')
                        continue

                    out.write(' '.join(
                        ['{0}:{1}'.format(name, value) for name, value in sorted(features[source_word].items())]))
                    out.write('\n')

    def write_token_features_to_file(self, output_path, feature_list):
        assert isinstance(feature_list, list)

        with open(output_path, mode='w', encoding='utf-8') as out:
            for instance in feature_list:
                if instance == '<ignore>':
                    out.write('\n')
                    continue

                assert isinstance(instance, dict)

                out.write(' '.join(['{0}:{1}'.format(name, value)
                                    if not isinstance(value, list)
                                    else ' '.join(['{0}:{1}'.format(name, v) for v in value])
                                    for name, value in sorted(instance.items())]))
                out.write('\n')

        return

    def write_context_token_features_to_file(self, feature_list):
        assert isinstance(feature_list, list)

        context_out = open(self.feature_file_configuration['TOKEN_CONTEXT'], 'w', encoding='utf-8')
        context_count_out = open(self.feature_file_configuration['TOKEN_CONTEXT_COUNT'], 'w', encoding='utf-8')
        context_percentage_out = open(self.feature_file_configuration['TOKEN_CONTEXT_PERCENTAGE'], 'w',
                                      encoding='utf-8')

        for instance in feature_list:
            if instance == '<ignore>':
                context_out.write('\n')
                context_count_out.write('\n')
                context_percentage_out.write('\n')
                continue

            assert isinstance(instance, dict)

            context_out.write(' '.join(['{0}:{1}'.format(name, value) for name, value in sorted(instance.items())]))

            context_count_out.write(' '.join(['{0}:{1}'.format(name, value)
                                              for name, value in sorted(instance.items())
                                              if name != 'perccont']))

            context_percentage_out.write(' '.join(['{0}:{1}'.format(name, value)
                                                   for name, value in sorted(instance.items())
                                                   if name == 'perccont']))

            context_out.write('\n')
            context_count_out.write('\n')
            context_percentage_out.write('\n')

        context_out.close()
        context_count_out.close()
        context_percentage_out.close()

    def derive_features_containing(self, input_path, output_path, prefix):

        out = open(output_path, 'w', encoding='utf-8')
        with open(input_path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip().split()
                new_line = []
                for feature in line:
                    feat_name, value = feature.split(':')
                    if feat_name.startswith(prefix):
                        new_line.append(feature)

                out.write(' '.join(new_line))
                out.write('\n')
        out.close()

        return

    def derive_features_not_containing(self, input_path, output_path, prefix):

        out = open(output_path, 'w', encoding='utf-8')
        with open(input_path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip().split()
                new_line = []
                for feature in line:
                    feat_name, value = feature.split(':')
                    if not feat_name.startswith(prefix):
                        new_line.append(feature)

                out.write(' '.join(new_line))
                out.write('\n')
        out.close()

        return

    def write_psd_token_features_to_file(self, feature_list, token_psd=False):
        assert isinstance(feature_list, list)

        if token_psd:
            outputs = {fname: open(self.feature_file_configuration[fname], 'w', encoding='utf-8')
                       for fname in self.feature_extraction_configuration if fname.startswith('TOKEN_PSD_')}
        else:
            outputs = {fname: open(self.feature_file_configuration[fname], 'w', encoding='utf-8')
                       for fname, value in self.feature_extraction_configuration.items()
                       if (value and fname.startswith('TOKEN_PSD_'))}

        if token_psd:
            psd_out = open(self.feature_file_configuration['TOKEN_PSD'], 'w', encoding='utf-8')

        for instance in feature_list:
            if instance == '<ignore>':
                for fname, out in outputs.items():
                    out.write('\n')
                continue

            assert isinstance(instance, dict)

            new_line = []
            lines = {fname: [] for fname in outputs}
            for name, value in sorted(instance.items()):
                if not isinstance(value, list):
                    feature_string = '{0}:{1}'.format(name, value)
                else:
                    feature_string = ' '.join(['{0}: {1}'.format(name, v) for v in value])

                new_line.append(feature_string)
                for fname in lines:
                    if name.startswith(self.psd_mapping[fname]):
                        lines[fname].append(feature_string)

            if token_psd:
                psd_out.write(' '.join(new_line))
                psd_out.write('\n')
            for fname, new_line in sorted(lines.items()):
                outputs[fname].write(' '.join(new_line))
                outputs[fname].write('\n')

        for fname, out in outputs.items():
            out.close()

        if token_psd:
            psd_out.close()

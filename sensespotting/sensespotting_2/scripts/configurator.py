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

class ConfigException(Exception):
    pass

class Configurator(object):
    def __init__(self, config_file=None, use_existing=False, verbose=False):
        self.verbose = verbose
        self.uninformative_label = '<uninformative>'

        self.root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
        self.use_existing = use_existing

        if not config_file:
            config_file = os.path.join(self.root, 'config.txt')
            if not self.check_file(config_file):
                print('ERROR: config file is REQUIRED!')
                sys.exit(0)

        self.config_file = config_file

        self.config_paths()

        self.check_required_paths()

        self.corpus_path = os.path.join(self.big_files_path, 'corpus')
        self.check_dir(self.corpus_path)

        self.aux_path = os.path.join(self.working_dir, 'aux_files')
        self.check_dir(self.aux_path)

        self.corpus_suffix = self.get_corpus_suffix()
        assert isinstance(self.corpus_suffix, str)
        assert self.corpus_suffix.endswith('.')

        self.load_fr_types()

        if self.all_combinations:
            self.preprocess_corpus_all(self.source_language)
        elif not self.lemmatize and not self.remove_stopwords and not self.remove_low_frequency_words:
            self.lowercase_corpus(self.source_language)
        elif self.lemmatize and not self.remove_stopwords and not self.remove_low_frequency_words:
            self.lowercase_and_lemmatize_corpus(self.source_language)
        else:
            self.preprocess_corpus(self.source_language)

        self.lowercase_and_lemmatize_corpus(self.target_language)

        self.corpus_file_name = self.corpus_file_name + self.corpus_suffix

        self.domain_paths = {}
        for domain in self.corpus_folder_paths:
            self.domain_paths[domain] = os.path.join(self.corpus_path, domain,
                                                     self.corpus_file_name + self.source_language)

        for feature in self.feature_file_configuration:
            name = self.feature_file_configuration[feature].split('.')
            suffix = name[-1]
            main_name = '.'.join(name[:-1])
            self.feature_file_configuration[feature] = '{0}.{1}{2}'.format(main_name, self.corpus_suffix, suffix)

        self.load_vocab()

        self.models_path = os.path.join(self.big_files_path, 'models')
        self.check_dir(self.models_path)

    #################################################
    ### CONFIGURATION AND PREPROCESSING FUNCTIONS ###
    #################################################

    def config_paths(self):

        configs = self.read_config_file()
        self.map_configs_to_variables(configs)

    def read_config_file(self):

        configs = {}
        with open(self.config_file) as handle:
            for line in handle:

                # skip comments
                if line.startswith('#'):
                    continue

                line = line.strip()
                if line:
                    if line.startswith('['):
                        header = line.strip('[]')

                    else:
                        name, value = line.split('=')
                        configs[':'.join([header, name.strip()])] = value.strip()

        return configs

    def map_configs_to_variables(self, configs):

        self.old_domain = configs.get('GENERAL:old_domain_name', None)
        self.new_domain = configs.get('GENERAL:new_domain_name', None)
        self.source_language = configs.get('GENERAL:source_language', None)
        self.target_language = configs.get('GENERAL:target_language', None)

        self.working_dir = configs.get('GENERAL:working_dir', self.root)
        self.big_files_path = configs.get('GENERAL:big_files_dir', self.working_dir)
        self.script_path = configs.get('GENERAL:script_path',
                                       os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

        corpus_file_name = configs.get('CORPUS:corpus_file', None)

        if not corpus_file_name.endswith('.'):
            self.corpus_file_name = corpus_file_name + '.'
        else:
            self.corpus_file_name = corpus_file_name

        self.corpus_folder_paths = {self.old_domain: configs.get('CORPUS:old_domain_dir', None),
                                    self.new_domain: configs.get('CORPUS:new_domain_dir', None)}

        self.psd_files = {self.old_domain: configs.get('PSD_FILE:old_domain_file', None),
                          self.new_domain: configs.get('PSD_FILE:new_domain_file', None)}

        self.remove_stopwords = int(configs.get('PREPROCESSING:remove_stopwords', 0))
        self.remove_low_frequency_words = int(configs.get('PREPROCESSING:remove_low_frequency_words', 0))
        self.low_freq_border = int(configs.get('PREPROCESSING:low_freq_border', 100))
        self.lemmatize = int(configs.get('PREPROCESSING:lemmatize', 1))
        self.all_combinations = int(configs.get('PREPROCESSING:all_combinations', 0))

        self.feature_file_configuration = {
            'TYPE_REL_FREQ': configs.get('FEATURE_EXTRACTION:type_rel_freq_fname', 'type_rel_freq.feat'),
            'TYPE_NGRAM_PROB': configs.get('FEATURE_EXTRACTION:type_ngram_prob_fname',
                                           'type_ngram_prob.feat'),
            'TYPE_CONTEXT': configs.get('FEATURE_EXTRACTION:type_context_fname', 'type_context.feat'),
            'TYPE_TOPIC': configs.get('FEATURE_EXTRACTION:type_topic_fname', 'type_topic.feat'),
            'TOKEN_NGRAM_PROB': configs.get('FEATURE_EXTRACTION:token_ngram_prob_fname',
                                            'token_ngram_prob.feat'),
            'TOKEN_CONTEXT': configs.get('FEATURE_EXTRACTION:token_context_fname', 'token_context.feat'),
            'TOKEN_CONTEXT_COUNT': configs.get('FEATURE_EXTRACTION:token_context_count_fname',
                                               'token_context_count.feat'),
            'TOKEN_CONTEXT_PERCENTAGE': configs.get('FEATURE_EXTRACTION:token_context_percentage_fname',
                                                    'token_context_percentage.feat'),

            'TOKEN_PSD': configs.get('FEATURE_EXTRACTION:token_psd_fname', 'token_psd.feat'),
            'TOKEN_PSD_GLOBAL': configs.get('FEATURE_EXTRACTION:token_psd_global_fname', 'token_psd_global.feat'),
            'TOKEN_PSD_GLOBAL_BINNED': configs.get('FEATURE_EXTRACTION:token_psd_global_binned_fname',
                                                   'token_psd_global_binned.feat'),
            'TOKEN_PSD_LOCAL': configs.get('FEATURE_EXTRACTION:token_psd_local_fname', 'token_psd_local.feat'),
            'TOKEN_PSD_LOCAL_BINNED': configs.get('FEATURE_EXTRACTION:token_psd_local_binned_fname',
                                                  'token_psd_local_binned.feat'),
            'TOKEN_PSD_RATIO': configs.get('FEATURE_EXTRACTION:token_psd_ratio_fname', 'token_psd_ratio.feat'),
        }

        self.feature_extraction_configuration = {
            'TYPE_REL_FREQ': int(configs.get('FEATURE_EXTRACTION:type_rel_freq', 0)),
            'TYPE_NGRAM_PROB': int(configs.get('FEATURE_EXTRACTION:type_ngram_prob',
                                               0)),
            'TYPE_CONTEXT': int(configs.get('FEATURE_EXTRACTION:type_context', 0)),
            'TYPE_TOPIC': int(configs.get('FEATURE_EXTRACTION:type_topic', 0)),
            'TOKEN_NGRAM_PROB': int(configs.get('FEATURE_EXTRACTION:token_ngram_prob',
                                                0)),
            'TOKEN_CONTEXT': int(
                configs.get('FEATURE_EXTRACTION:token_context', 0)),
            'TOKEN_CONTEXT_COUNT': int(
                configs.get('FEATURE_EXTRACTION:token_context_count', 0)),
            'TOKEN_CONTEXT_PERCENTAGE': int(
                configs.get('FEATURE_EXTRACTION:token_context_percentage', 0)),
            'TOKEN_PSD': int(configs.get('FEATURE_EXTRACTION:token_psd', 0)),
            'TOKEN_PSD_GLOBAL': int(configs.get('FEATURE_EXTRACTION:token_psd_global', 0)),
            'TOKEN_PSD_GLOBAL_BINNED': int(configs.get('FEATURE_EXTRACTION:token_psd_global_binned', 0)),
            'TOKEN_PSD_LOCAL': int(configs.get('FEATURE_EXTRACTION:token_psd_local', 0)),
            'TOKEN_PSD_LOCAL_BINNED': int(configs.get('FEATURE_EXTRACTION:token_psd_local_binned', 0)),
            'TOKEN_PSD_RATIO': int(configs.get('FEATURE_EXTRACTION:token_psd_ratio', 0)),
        }

        self.max_type_freq = configs.get('TRAINING:maximal_type_frequency', None)
        if self.max_type_freq:
            self.max_type_freq = int(self.max_type_freq)

        self.num_folds = int(configs.get('TRAINING:num_folds', 16))
        self.max_buckets = configs.get('TRAINING:max_buckets', None)
        if self.max_buckets:
            self.max_buckets = int(self.max_buckets)

        self.seen_path = configs.get('TRAINING:seen_path', None)
        self.do_cv = int(configs.get('TRAINING:cross_validation', 1))
        self.repeat_cv = int(configs.get('TRAINING:repetition', 1))
        self.do_hyperopt = int(configs.get('TRAINING:hyperparameter_optimization', 1))
        self.use_dev_for = configs.get('TRAINING:use_dev_for', 'training')
        self.use_hold_out = int(configs.get('TRAINING:use_hold_out', 1))
        self.gold_label = configs.get('TRAINING:use_as_gold_label', 'new_sense').replace(' ', '_')

        # print('CV: {0}, Hyperopt: {1}, Holdout: {2}'.format(self.do_cv, self.do_hyperopt, self.use_hold_out))

        self.set_bias = int(configs.get('TRAINING:set_bias', 1))

        self.language_models = {self.old_domain: {'ug': configs.get('LANGUAGE_MODEL:old_domain_arpa_ug', None),
                                                  'ng': configs.get('LANGUAGE_MODEL:old_domain_arpa_ng', None)},
                                self.new_domain: {'ug': configs.get('LANGUAGE_MODEL:new_domain_arpa_ug', None),
                                                  'ng': configs.get('LANGUAGE_MODEL:new_domain_arpa_ng', None)}}

        self.ppl_models = {self.old_domain: {'ug': configs.get('NGRAM_PERPLEXITY:old_domain_ppl_ug', None),
                                             'ng': configs.get('NGRAM_PERPLEXITY:old_domain_ppl_ng', None)},
                           self.new_domain: {'ug': configs.get('NGRAM_PERPLEXITY:new_domain_ppl_ug', None),
                                             'ng': configs.get('NGRAM_PERPLEXITY:new_domain_ppl_ng', None)}}

        self.ngrams = {self.old_domain: configs.get('LANGUAGE_MODEL:old_domain_ngrams', None),
                       self.new_domain: configs.get('LANGUAGE_MODEL:old_domain_ngrams', None)}

        self.topic_model = {self.old_domain: {'model': configs.get('TOPIC_MODEL:old_domain_file', None),
                                              'dct': configs.get('TOPIC_MODEL:old_domain_dict', None)},
                            self.new_domain: {'model': configs.get('TOPIC_MODEL:new_domain_file', None),
                                              'dct': configs.get('TOPIC_MODEL:new_domain_dict', None)}}

        self.tagged_corpus_file = {self.old_domain: {self.source_language: configs.get(
            'TAGGED_CORPUS:source_language_old_domain_file', None),
            self.target_language: configs.get(
                'TAGGED_CORPUS:target_language_old_domain_file', None)},
            self.new_domain: {self.source_language: configs.get(
                'TAGGED_CORPUS:source_language_new_domain_file', None),
                self.target_language: configs.get(
                    'TAGGED_CORPUS:target_language_new_domain_file', None)}}

        self.alignment_file = configs.get('ALIGNMENT:alignment_file', None)
        self.phrase_table_file = configs.get('PHRASE_TABLE:phrase_table_file', None)

        self.srilm_path = configs.get('TOOLS:srilm_dir', None)
        self.tagger_path = configs.get('TOOLS:tree_tagger_dir', None)
        self.aligner_path = configs.get('TOOLS:aligner_dir', None)
        self.moses_path = configs.get('TOOLS:moses_dir', None)
        self.vowpal_wabbit_path = configs.get('TOOLS:vowpal_wabbit', None)

    def check_required_paths(self):

        if not self.old_domain:
            raise ConfigException('REQUIRED parameter <GENERAL:old_domain> in config file (GENERAL)!')

        if not self.new_domain:
            raise ConfigException('REQUIRED parameter <GENERAL:new_domain> in config file (GENERAL)!')

        if not self.source_language:
            raise ConfigException('REQUIRED parameter <GENERAL:source_language> in config file (GENERAL)!')

        if not self.target_language:
            raise ConfigException('REQUIRED parameter <GENERAL:target_language> in config file (GENERAL)!')

        # check if corpus data is found
        for domain in [self.old_domain, self.new_domain]:
            if self.corpus_folder_paths[domain]:
                for language in [self.source_language, self.target_language]:
                    path = os.path.join(self.corpus_folder_paths[domain], self.corpus_file_name + language)
                    if not os.path.isfile(path) and not os.path.isfile(path + '.gz'):
                        raise FileNotFoundError(path)

            if self.psd_files[domain]:
                if not os.path.isfile(self.psd_files[domain]):
                    raise FileNotFoundError(self.psd_files[domain])

        return

    def get_corpus_suffix(self):

        suffix = []

        if self.lemmatize:
            suffix.append('lemmatized')
        else:
            suffix.append('lowercased')

        if self.remove_stopwords:
            suffix.append('removed_stopwords')

        if self.remove_low_frequency_words:
            suffix.append('removed_low_freq')

        return '.'.join(suffix) + '.'

    def load_fr_types(self):
        frtypes_path = self.psd_files[self.new_domain] + '.frtypes'
        if self.check_file(frtypes_path):
            self.frtypes = set([line.strip() for line in open(frtypes_path,
                                                              encoding='utf-8').readlines()])
        else:
            self.frtypes = set([line.strip().split('\t')[5] for line in open(self.psd_files[self.new_domain],
                                                                             encoding='utf-8').readlines()])

    def derive_lemmatized_types(self):

        filename = os.path.join(self.aux_path, 'fr_type_mapping.{}'.format(self.corpus_suffix))

        if self.check_file('{0}pkl'.format(filename)):
            self.fr_types_mapping = self.load_pickle_obj(filename)
        else:
            lowercased_corpus_path = os.path.join(self.corpus_path, 'EMEA', 'train.lowercased.fr')
            lemmatized_corpus_path = os.path.join(self.corpus_path, 'EMEA', 'train.lemmatized.fr')

            # punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
            #                             if unicodedata.category(chr(i)).startswith('P'))

            assert self.check_file(lowercased_corpus_path)
            assert self.check_file(lemmatized_corpus_path)

            self.fr_types_mapping = {}

            data, last_sent_id, source_words = self.read_psd_file(self.new_domain)

            sent_id = 0
            lemmatized_corpus = self.get_next_line(lemmatized_corpus_path)
            with open(lowercased_corpus_path, encoding='utf-8') as handle:
                for line in handle:
                    sent_id += 1
                    lemmatized_line = next(lemmatized_corpus)
                    if sent_id not in data:
                        continue

                    line = line.strip().split()

                    assert len(lemmatized_line) == len(line)

                    for line_nb, start, end in sorted(data[sent_id]):
                        if start == end:
                            assert line[start] == data[sent_id][(line_nb, start, end)][0]
                            lemmatized_fr_type = lemmatized_line[start]
                        else:
                            assert ' '.join(line[start:end + 1]) == data[sent_id][(line_nb, start, end)][0]
                            lemmatized_fr_type = ' '.join(lemmatized_line[start:end + 1])

                        # if self.remove_stopwords or self.remove_low_frequency_words:
                        #     lemmatized_fr_type = lemmatized_fr_type.translate(punctuation)

                        # if lemmatized_fr_type in {'vue du enfant', 'et/ou', 'm.'}:
                        #     print(data[sent_id][(line_nb, start, end)][0], lemmatized_fr_type)

                        self.fr_types_mapping[line_nb] = lemmatized_fr_type

            self.save_pickle_obj(self.fr_types_mapping, filename)

        self.frtypes = set(self.fr_types_mapping.values())
        if self.verbose:
            print('\t- define lemmatized fr types')

    def load_stopwords(self, language):
        path = os.path.join(self.aux_path, 'stopwords_{}.txt'.format(language))
        stopwords = set()
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '#' in line:
                        line = line.split('#')[0].strip()

                    if line in self.frtypes:
                        continue

                    stopwords.add(line)

        stopwords = stopwords - self.frtypes

        return stopwords

    def preprocess_corpus_all(self, language):
        """replace html entities, lowercase and tokenize corpus"""
        if self.verbose:
            print('\nCHECK for preprocessed {} corpus...'.format(language))

        for domain in [self.new_domain, self.old_domain]:
            if self.verbose:
                print('- preprocess {0}.{1}'.format(domain, language))

            input_path = os.path.join(self.corpus_folder_paths[domain], self.corpus_file_name)
            input_file = input_path + language

            output_path = os.path.join(self.corpus_path, domain)
            self.check_dir(output_path)

            lowercased_output_file = os.path.join(output_path, self.corpus_file_name + 'lowercased.' + language)

            lemmatized_output_file = os.path.join(output_path, self.corpus_file_name + 'lemmatized.' + language)

            word_per_line_output = os.path.join(output_path, self.corpus_file_name + 'word_per_line.')

            tagger_input_path = word_per_line_output + language
            tagger_output_path = word_per_line_output + 'tagged.' + language
            self.tagged_corpus_file[domain][language] = tagger_output_path

            # lowercase corpus and save it (since needed as input for tagger)
            if not self.check_file(lowercased_output_file) or not self.use_existing:
                if self.verbose:
                    print('\t- create {}'.format(lowercased_output_file))

                out = open(lowercased_output_file, 'w', encoding='utf-8')
                with open(input_file, encoding='utf-8') as handle:
                    for line in handle:
                        line = html.unescape(line.strip()).lower()
                        out.write(line + '\n')
                out.close()
            else:
                if self.verbose:
                    print('\t- use existing {}'.format(lowercased_output_file))

            if not self.check_file(tagger_output_path) or not self.use_existing:
                self.create_word_per_line_format(lowercased_output_file,
                                                 word_per_line_output + language)

                self.tagged_corpus_file[domain][language] = tagger_output_path
                self.run_tagger(tagger_input_path, tagger_output_path, language)
            else:
                if self.verbose:
                    print('\t- use existing tagged {} corpus'.format(domain))

            if self.check_file(lemmatized_output_file) and self.use_existing:
                if self.verbose:
                    print('\t- use existing {}'.format(lemmatized_output_file))
            else:
                if self.verbose:
                    print('- create {}'.format(lemmatized_output_file))
                tagged_corpus = self.yield_pos_tags(tagger_output_path)

                out = open(lemmatized_output_file, 'w', encoding='utf-8')
                with open(lowercased_output_file, encoding='utf-8') as handle:
                    for sentence in handle:
                        sentence = sentence.strip().split()
                        tagged_sentence = next(tagged_corpus)

                        assert len(tagged_sentence) == len(sentence)
                        lemmatized_sentence = [tagged_sentence[i]['lemma']
                                               if tagged_sentence[i]['lemma'] != '<unknown>'
                                               else sentence[i]
                                               for i in range(len(tagged_sentence))
                                               ]
                        out.write(' '.join(lemmatized_sentence) + '\n')
                out.close()

            if domain == self.new_domain and language == self.source_language:
                self.derive_lemmatized_types()

            removed_stopwords_output_file = 'lowercased.removed_stopwords.'
            removed_low_freq_output_file = 'lowercased.removed_low_freq.'
            removed_stopwords_and_low_freq_out = 'lowercased.removed_stopwords.removed_low_freq.'
            lemmatize_and_low_freq_output_file = 'lemmatized.removed_low_freq.'
            lemmatize_and_stopwords_output_file = 'lemmatized.removed_stopwords.'
            all_output_file = 'lemmatized.removed_stopwords.removed_low_freq.'

            if self.verbose:
                print('\t- collect low frequency words')

            lowercased_low_freq_words = self.collect_low_frequency_words(domain, lowercased_output_file)
            lemmatized_low_freq_words = self.derive_lemmatized_low_freq_words(domain,
                                                                              lowercased_output_file,
                                                                              lemmatized_output_file,
                                                                              lowercased_low_freq_words)

            if self.verbose:
                print('\t- load stopwords')
            stopwords = self.load_stopwords(language)

            self.write_all_corpus_files(domain,
                                        lowercased_output_file,
                                        removed_stopwords_output_file,
                                        removed_low_freq_output_file,
                                        removed_stopwords_and_low_freq_out,
                                        stopwords,
                                        lowercased_low_freq_words,
                                        output_path,
                                        language)

            # if self.verbose:
            #     print('\t\t- finished removal in lowercased file')

            self.write_all_corpus_files(domain,
                                        lemmatized_output_file,
                                        lemmatize_and_stopwords_output_file,
                                        lemmatize_and_low_freq_output_file,
                                        all_output_file,
                                        stopwords,
                                        lemmatized_low_freq_words,
                                        output_path,
                                        language)

            # if self.verbose:
            #     print('\t\t- finished removal in lemmatized file')

        self.create_all_corpus_files_without_placeholder()
        return

    def write_all_corpus_files(self, domain, input_file, out_stopwords, out_low_freq, out_both, stopwords,
                               low_freq_words,
                               output_path, language):

        uninformative_words = stopwords | low_freq_words

        out_stopwords_path = os.path.join(output_path, self.corpus_file_name + out_stopwords + language)
        out_low_freq_path = os.path.join(output_path, self.corpus_file_name + out_low_freq + language)
        out_both_path = os.path.join(output_path, self.corpus_file_name + out_both + language)

        if self.check_file(out_stopwords_path) and self.use_existing:
            if self.verbose:
                print('\t- use existing {}'.format(out_stopwords_path))
            out_stopwords = None
        else:
            if self.verbose:
                print('\t- create {}'.format(out_stopwords_path))
            stopwords_out = open(out_stopwords_path, 'w', encoding='utf-8')

        if self.check_file(out_low_freq_path) and self.use_existing:
            if self.verbose:
                print('\t- use existing {}'.format(out_low_freq_path))
            out_low_freq = None
        else:
            if self.verbose:
                print('\t- create {}'.format(out_low_freq_path))
            low_freq_out = open(out_low_freq_path, 'w', encoding='utf-8')

        if self.check_file(out_both_path) and self.use_existing:
            if self.verbose:
                print('\t- use existing {}'.format(out_both_path))
            out_both = None
        else:
            if self.verbose:
                print('\t- create {}'.format(out_both_path))
            both_out = open(out_both_path, 'w', encoding='utf-8')

        sent_id = 0

        if out_stopwords or out_low_freq or out_both:
            with open(input_file, encoding='utf-8') as handle:
                for sentence in handle:
                    sent_id += 1

                    sentence = sentence.strip().split()

                    if self.verbose and sent_id % 1000000 == 0:
                        print('\t\t- processed {} sentences'.format(sent_id))

                    removed_stopwords_sentence = []
                    removed_low_freq_sentence = []
                    removed_both_sentence = []

                    for i in range(len(sentence)):
                        word = sentence[i]

                        if out_stopwords:
                            if word in stopwords or word in set(string.punctuation):
                                removed_stopwords_sentence.append(self.uninformative_label)
                            else:
                                removed_stopwords_sentence.append(word)

                        if out_low_freq:
                            if word in low_freq_words or word in set(string.punctuation):
                                removed_low_freq_sentence.append(self.uninformative_label)
                            else:
                                removed_low_freq_sentence.append(word)

                        if out_both:
                            if word in uninformative_words or word in set(string.punctuation):
                                removed_both_sentence.append(self.uninformative_label)
                            else:
                                removed_both_sentence.append(word)

                    if out_stopwords:
                        assert len(sentence) == len(removed_stopwords_sentence)
                        stopwords_out.write(' '.join(removed_stopwords_sentence) + '\n')

                    if out_low_freq:
                        assert len(sentence) == len(removed_low_freq_sentence)
                        low_freq_out.write(' '.join(removed_low_freq_sentence) + '\n')

                    if out_both:
                        assert len(sentence) == len(removed_both_sentence)
                        both_out.write(' '.join(removed_both_sentence) + '\n')

            if out_stopwords:
                stopwords_out.close()
                stopword_mapping_path = os.path.join(self.corpus_path, domain,
                                                     '{0}.{1}corpus_mapping'.format(domain, out_stopwords))
                if self.verbose:
                    print('\t\t- save {}'.format(stopword_mapping_path))
                stopword_mapping = self.derive_mapping_from_file(out_stopwords_path)
                self.save_pickle_obj(stopword_mapping, stopword_mapping_path)

            if out_low_freq:
                low_freq_out.close()
                low_freq_mapping_path = os.path.join(self.corpus_path, domain,
                                                     '{0}.{1}corpus_mapping'.format(domain, out_low_freq))
                if self.verbose:
                    print('\t\t- save {}'.format(low_freq_mapping_path))
                low_freq_mapping = self.derive_mapping_from_file(out_low_freq_path)
                self.save_pickle_obj(low_freq_mapping, low_freq_mapping_path)

            if out_both:
                both_out.close()
                both_mapping_path = os.path.join(self.corpus_path, domain,
                                                 '{0}.{1}corpus_mapping'.format(domain, out_both))
                if self.verbose:
                    print('\t\t- save {}'.format(both_mapping_path))
                both_mapping = self.derive_mapping_from_file(out_both_path)
                self.save_pickle_obj(both_mapping, both_mapping_path)

        return

    def preprocess_corpus(self, language):
        if self.verbose:
            print('\nCHECK for preprocessed {} corpus...'.format(language))

        for domain in [self.new_domain, self.old_domain]:

            if self.verbose:
                print('- preprocess {0}.{1}'.format(domain, language))

            input_path = os.path.join(self.corpus_folder_paths[domain], self.corpus_file_name)
            input_file = input_path + language

            output_path = os.path.join(self.corpus_path, domain)
            self.check_dir(output_path)

            lowercased_output_file = os.path.join(output_path, self.corpus_file_name + 'lowercased.' + language)

            word_per_line_output = os.path.join(output_path, self.corpus_file_name + 'word_per_line.')

            tagger_input_path = word_per_line_output + language
            tagger_output_path = word_per_line_output + 'tagged.' + language
            self.tagged_corpus_file[domain][language] = tagger_output_path

            # lowercase corpus and save it (since needed as input for tagger)
            if not self.check_file(lowercased_output_file) or not self.use_existing:
                if self.verbose:
                    print('\t- create {}'.format(lowercased_output_file))

                out = open(lowercased_output_file, 'w', encoding='utf-8')
                with open(input_file, encoding='utf-8') as handle:
                    for line in handle:
                        line = html.unescape(line.strip()).lower()
                        out.write(line + '\n')
                out.close()
            else:
                if self.verbose:
                    print('\t- use existing {}'.format(lowercased_output_file))

            if not self.check_file(tagger_output_path) or not self.use_existing:
                self.create_word_per_line_format(lowercased_output_file,
                                                 word_per_line_output + language)

                self.tagged_corpus_file[domain][language] = tagger_output_path
                self.run_tagger(tagger_input_path, tagger_output_path, language)
            else:
                if self.verbose:
                    print('\t- use existing tagged {} corpus'.format(domain))

            if self.lemmatize:

                lemmatized_output_file = os.path.join(output_path, self.corpus_file_name + 'lemmatized.' + language)

                if self.check_file(lemmatized_output_file) and self.use_existing:
                    if self.verbose:
                        print('\t- use existing {}'.format(lemmatized_output_file))
                else:
                    if self.verbose:
                        print('\t- create {}'.format(lemmatized_output_file))
                    tagged_corpus = self.yield_pos_tags(tagger_output_path)
                    out = open(lemmatized_output_file, 'w', encoding='utf-8')
                    with open(lowercased_output_file, encoding='utf-8') as handle:
                        for sentence in handle:
                            sentence = sentence.strip().split()
                            tagged_sentence = next(tagged_corpus)

                            assert len(tagged_sentence) == len(sentence)
                            lemmatized_sentence = [tagged_sentence[i]['lemma']
                                                   if tagged_sentence[i]['lemma'] != '<unknown>'
                                                   else sentence[i]
                                                   for i in range(len(tagged_sentence))
                                                   ]
                            out.write(' '.join(lemmatized_sentence) + '\n')
                    out.close()

                if domain == self.new_domain and language == self.source_language:
                    self.derive_lemmatized_types()

                if self.remove_stopwords:
                    uninformative_words = self.load_stopwords(language)

                    output_file = os.path.join(output_path, self.corpus_file_name + self.corpus_suffix + language)
                    self.write_corpus_file(domain, lemmatized_output_file, output_file, uninformative_words)

                    without_placeholder_file = os.path.join(output_path,
                                                            self.corpus_file_name + self.corpus_suffix + 'no_placeholder.' + language)
                    self.create_corpus_file_without_placeholder(output_file, without_placeholder_file)

                else:
                    continue
            else:
                if self.remove_stopwords:
                    uninformative_words = self.load_stopwords(language)
                    output_file = os.path.join(output_path, self.corpus_file_name + self.corpus_suffix + language)
                    self.write_corpus_file(domain, lowercased_output_file, output_file, uninformative_words)
                    without_placeholder_file = os.path.join(output_path,
                                                            self.corpus_file_name + self.corpus_suffix + 'no_placeholder.' + language)
                    self.create_corpus_file_without_placeholder(output_file, without_placeholder_file)
                else:
                    continue

        return

    def write_corpus_file(self, domain, input_file, output_file, uninformative_words):

        mapping = {}

        if self.check_file(output_file) and self.use_existing:
            if self.verbose:
                print('\t- use existing {}'.format(output_file))

            mapping_path = os.path.join(self.corpus_path, domain,
                                        '{0}.{1}corpus_mapping'.format(domain, self.corpus_suffix))
            if not self.check_file(mapping_path + '.pkl'):
                if self.verbose:
                    print('\t- create {}'.format(mapping_path))
                mapping = self.derive_mapping_from_file(output_file)
                self.save_pickle_obj(mapping, mapping_path)
            else:
                if self.verbose:
                    print('\t- use existing {}'.format(mapping_path))

            del uninformative_words
            gc.collect()

        else:
            if self.verbose:
                print('\t- create {}'.format(output_file))

            sent_id = 0
            out = open(output_file, 'w', encoding='utf-8')
            punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                                        if unicodedata.category(chr(i)).startswith('P'))
            punctuation_signs = set([chr(i) for i in range(sys.maxunicode)
                                     if unicodedata.category(chr(i)).startswith('P')])

            with open(input_file, encoding='utf-8') as handle:
                for sentence in handle:
                    sent_id += 1
                    sentence = sentence.strip().split()

                    if self.verbose and sent_id % 1000000 == 0:
                        print('\t\t- processed {} sentences'.format(sent_id))

                    cleaned_sentence = []
                    sent_mapping = {}
                    informative_counter = -1
                    for i in range(len(sentence)):
                        word = sentence[i]
                        if word in uninformative_words or word in punctuation_signs or (
                                not re.match("^[\w]*$", word) and not '|' in word and word not in self.frtypes):
                            cleaned_sentence.append(self.uninformative_label)
                            sent_mapping[i] = -1
                        else:
                            informative_counter += 1
                            cleaned_sentence.append(word)
                            sent_mapping[i] = informative_counter

                    mapping[sent_id] = sent_mapping

                    assert len(sentence) == len(cleaned_sentence)
                    cleaned_sentence = ' '.join(cleaned_sentence)
                    # cleaned_sentence = cleaned_sentence.translate(punctuation)

                    out.write(cleaned_sentence + '\n')
            out.close()

            mapping_path = os.path.join(self.corpus_path, domain,
                                        '{0}.{1}corpus_mapping'.format(domain, self.corpus_suffix))
            if self.verbose:
                print('\t\t- save {}'.format(mapping_path))
            self.save_pickle_obj(mapping, mapping_path)

        return

    def lowercase_corpus(self, language):
        """replace html entities, lowercase and tokenize corpus"""
        if self.verbose:
            print('\nCHECK for preprocessed {} corpus...'.format(language))

        for domain in [self.new_domain, self.old_domain]:

            if self.verbose:
                print('- preprocess {0}.{1}'.format(domain, language))

            input_path = os.path.join(self.corpus_folder_paths[domain], self.corpus_file_name)
            input_file = input_path + language

            output_path = os.path.join(self.corpus_path, domain)
            self.check_dir(output_path)

            lowercased_output_file = os.path.join(output_path, self.corpus_file_name + 'lowercased.' + language)

            word_per_line_output = os.path.join(output_path, self.corpus_file_name + 'word_per_line.')

            tagger_input_path = word_per_line_output + language
            tagger_output_path = word_per_line_output + 'tagged.' + language
            self.tagged_corpus_file[domain][language] = tagger_output_path

            # lowercase corpus and save it (since needed as input for tagger)
            if not self.check_file(lowercased_output_file) or not self.use_existing:
                if self.verbose:
                    print('\t- create {}'.format(lowercased_output_file))

                out = open(lowercased_output_file, 'w', encoding='utf-8')
                with open(input_file, encoding='utf-8') as handle:
                    for line in handle:
                        line = html.unescape(line.strip()).lower()
                        out.write(line + '\n')
                out.close()
            else:
                if self.verbose:
                    print('\t- use existing {}'.format(lowercased_output_file))

            if not self.check_file(tagger_output_path) or not self.use_existing:
                self.create_word_per_line_format(lowercased_output_file,
                                                 word_per_line_output + language)

                self.tagged_corpus_file[domain][language] = tagger_output_path
                self.run_tagger(tagger_input_path, tagger_output_path, language)
            else:
                if self.verbose:
                    print('\t- use existing tagged {} corpus'.format(domain))

        return

    def lowercase_and_lemmatize_corpus(self, language):

        """replace html entities, lowercase and tokenize corpus"""
        if self.verbose:
            print('\nCHECK for preprocessed {} corpus...'.format(language))

        for domain in [self.new_domain, self.old_domain]:
            if self.verbose:
                print('- preprocess {0}.{1}'.format(domain, language))

            input_path = os.path.join(self.corpus_folder_paths[domain], self.corpus_file_name)
            input_file = input_path + language

            output_path = os.path.join(self.corpus_path, domain)
            self.check_dir(output_path)

            lowercased_output_file = os.path.join(output_path, self.corpus_file_name + 'lowercased.' + language)

            word_per_line_output = os.path.join(output_path, self.corpus_file_name + 'word_per_line.')

            tagger_input_path = word_per_line_output + language
            tagger_output_path = word_per_line_output + 'tagged.' + language
            self.tagged_corpus_file[domain][language] = tagger_output_path

            # lowercase corpus and save it (since needed as input for tagger)
            if not self.check_file(lowercased_output_file) or not self.use_existing:
                if self.verbose:
                    print('\t- create {}'.format(lowercased_output_file))

                out = open(lowercased_output_file, 'w', encoding='utf-8')
                with open(input_file, encoding='utf-8') as handle:
                    for line in handle:
                        line = html.unescape(line.strip()).lower()
                        out.write(line + '\n')
                out.close()
            else:
                if self.verbose:
                    print('\t- use existing {}'.format(lowercased_output_file))

            if not self.check_file(tagger_output_path) or not self.use_existing:
                self.create_word_per_line_format(lowercased_output_file,
                                                 word_per_line_output + language)

                self.tagged_corpus_file[domain][language] = tagger_output_path
                self.run_tagger(tagger_input_path, tagger_output_path, language)
            else:
                if self.verbose:
                    print('\t- use existing tagged {} corpus'.format(domain))

            lemmatized_output_file = os.path.join(output_path, self.corpus_file_name + 'lemmatized.' + language)

            if self.check_file(lemmatized_output_file) and self.use_existing:
                if self.verbose:
                    print('\t- use existing {}'.format(lemmatized_output_file))
            else:
                if self.verbose:
                    print('\t- create {}'.format(lemmatized_output_file))
                tagged_corpus = self.yield_pos_tags(tagger_output_path)
                out = open(lemmatized_output_file, 'w', encoding='utf-8')
                with open(lowercased_output_file, encoding='utf-8') as handle:
                    for sentence in handle:
                        sentence = sentence.strip().split()
                        tagged_sentence = next(tagged_corpus)

                        assert len(tagged_sentence) == len(sentence)
                        lemmatized_sentence = [tagged_sentence[i]['lemma']
                                               if tagged_sentence[i]['lemma'] != '<unknown>'
                                               else sentence[i]
                                               for i in range(len(tagged_sentence))
                                               ]
                        out.write(' '.join(lemmatized_sentence) + '\n')
                out.close()

        if language == self.source_language:
            self.derive_lemmatized_types()
        return

    def load_vocab(self):
        if self.verbose:
            print('\nLOAD vocabulary...')

        self.frtypes_phrases = [word for word in self.frtypes if len(word.split()) > 1]

        self.old_domain_vocab, self.new_domain_vocab = self.get_domain_vocab()
        self.OOVs, self.common_vocab = self.get_OOVs_and_common_vocab()

    def get_OOVs_and_common_vocab(self):

        compute_OOVs = False;
        compute_common_vocab = False

        OOV_path = os.path.join(self.aux_path, '{0}.{1}oov'.format('_'.join(sorted(self.domain_paths.keys())),
                                                                   self.corpus_suffix))
        common_vocab_path = os.path.join(self.aux_path,
                                         '{0}.{1}common_vocab'.format('_'.join(sorted(self.domain_paths.keys())),
                                                                      self.corpus_suffix))

        if os.path.isfile(OOV_path + '.pkl') and self.use_existing:
            if self.verbose:
                print('- load existing OOVs file')
            OOVs = self.load_pickle_obj(OOV_path)
        else:
            compute_OOVs = True

        if os.path.isfile(common_vocab_path + '.pkl') and self.use_existing:
            if self.verbose:
                print('- load existing common vocab file')
            common_vocab = self.load_pickle_obj(common_vocab_path)
        else:
            compute_common_vocab = True

        if compute_common_vocab or compute_OOVs:

            old_domain_vocab, new_domain_vocab = self.get_domain_vocab()
            new_domain_vocab = set(new_domain_vocab)
            old_domain_vocab = set(old_domain_vocab)

            if compute_OOVs:
                if self.verbose:
                    print('- compute OOVs')
                OOVs = new_domain_vocab - old_domain_vocab
                self.save_pickle_obj(OOVs, OOV_path)

            if compute_common_vocab:
                if self.verbose:
                    print('- compute common vocab')
                common_vocab = old_domain_vocab & new_domain_vocab

                if self.uninformative_label in common_vocab:
                    common_vocab.remove(self.uninformative_label)

                self.save_pickle_obj(common_vocab, common_vocab_path)

        if self.verbose:
            print('- OOVs: {} words'.format(len(OOVs)))
            print('- common vocab: {} words'.format(len(common_vocab)))

        return OOVs, common_vocab

    def get_domain_vocab(self):

        old_domain_path = os.path.join(self.aux_path,
                                       '.'.join([self.old_domain, self.source_language, self.corpus_suffix[:-1],
                                                 'vocab_counter']))
        new_domain_path = os.path.join(self.aux_path,
                                       '.'.join([self.new_domain, self.source_language, self.corpus_suffix[:-1],
                                                 'vocab_counter']))
        if os.path.isfile(old_domain_path + '.pkl') and self.use_existing:
            if self.verbose:
                print('- load existing vocabulary file for old domain')

            old_domain_vocab = self.load_pickle_obj(old_domain_path)
        else:
            if self.verbose:
                print('- build vocabulary from {}'.format(self.domain_paths[self.old_domain]))

            old_domain_vocab = self.get_vocab(self.domain_paths[self.old_domain])
            self.save_pickle_obj(old_domain_vocab, old_domain_path)

        if os.path.isfile(new_domain_path + '.pkl') and self.use_existing:
            if self.verbose:
                print('- load existing vocabulary file for new domain')

            new_domain_vocab = self.load_pickle_obj(new_domain_path)
        else:
            if self.verbose:
                print('- build vocabulary from {}'.format(self.domain_paths[self.new_domain]))

            new_domain_vocab = self.get_vocab(self.domain_paths[self.new_domain])
            self.save_pickle_obj(new_domain_vocab, new_domain_path)

        return old_domain_vocab, new_domain_vocab

    def get_vocab(self, path_to_corpus):

        vocabulary = Counter()

        with open(path_to_corpus, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip().split()
                line = [token.lower() for token in line]
                vocabulary.update(line)

        return vocabulary

    def derive_mapping_from_file(self, file_path):

        mapping = {}
        sent_id = 0
        with open(file_path, encoding='utf-8') as handle:
            for sentence in handle:
                sent_id += 1
                sentence = sentence.strip().split()

                sent_mapping = {}
                informative_counter = -1
                for i in range(len(sentence)):
                    word = sentence[i]
                    if word == self.uninformative_label or word in set(string.punctuation):
                        sent_mapping[i] = -1
                    else:
                        informative_counter += 1
                        sent_mapping[i] = informative_counter

                mapping[sent_id] = sent_mapping

        return mapping

    def get_mapping_after_removing_words(self):

        for domain in self.corpus_folder_paths:
            output_path = os.path.join(self.corpus_path, domain)
            language = 'fr'
            suffixes = ['lowercased.removed_stopwords.', 'lowercased.removed_low_freq.',
                        'lowercased.removed_stopwords.removed_low_freq.',
                        'lemmatized.removed_low_freq.', 'lemmatized.removed_stopwords.',
                        'lemmatized.removed_stopwords.removed_low_freq.']

            for suffix in suffixes:
                output_file = os.path.join(output_path, self.corpus_file_name + suffix + language)
                mapping_path = os.path.join(self.corpus_path, domain, '{0}.{1}corpus_mapping'.format(domain, suffix))
                if not self.check_file(mapping_path):
                    mapping = self.derive_mapping_from_file(output_file)
                    if self.verbose:
                        print('\t\t- save {}'.format(mapping_path))
                    self.save_pickle_obj(mapping, mapping_path)

        return

    def create_all_corpus_files_without_placeholder(self):
        for domain in self.corpus_folder_paths:
            output_path = os.path.join(self.corpus_path, domain)
            language = 'fr'
            suffixes = ['lowercased.removed_stopwords.', 'lowercased.removed_low_freq.',
                        'lowercased.removed_stopwords.removed_low_freq.',
                        'lemmatized.removed_low_freq.', 'lemmatized.removed_stopwords.',
                        'lemmatized.removed_stopwords.removed_low_freq.']

            for suffix in suffixes:
                input_file = os.path.join(output_path, self.corpus_file_name + suffix + language)
                output_file = os.path.join(output_path, self.corpus_file_name + suffix + 'no_placeholder.' + language)

                self.create_corpus_file_without_placeholder(input_file, output_file)

        return

    def create_corpus_file_without_placeholder(self, input_file, output_file):

        if self.check_file(output_file) and self.use_existing:
            if self.verbose:
                print('\t- use existing {}'.format(output_file))
        else:
            if self.verbose:
                print('\t- create {}'.format(output_file))

            out = open(output_file, 'w', encoding='utf-8')
            with open(input_file, encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip().split()
                    new_line = []
                    for word in line:
                        if word != self.uninformative_label:
                            new_line.append(word)

                    if new_line:
                        out.write(' '.join(new_line) + '\n')
                    else:
                        out.write('<empty_sentence>' + '\n')

            out.close()

        return

    #################################################
    ### GENERAL USEFULL FUNCTIONS                 ###
    #################################################

    def load_pickle_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_pickle_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def check_file(self, filename):
        # check if variables are defined = paths to the files were provided
        if filename:
            # True if file is found; else False
            return True if os.path.isfile(filename) else False
        # no path given
        else:
            return False

    def check_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def check_memory(self, state='<unk>'):
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        if self.verbose:
            print('- memory usage after "{0}": {1} GB'.format(state, round(memoryUse, 2)))

    def shuffle_data(self, a, b):
        # Generate the permutation index array.
        permutation = np.random.permutation(a.shape[0])
        # Shuffle the arrays by giving the permutation in the square brackets.
        shuffled_a = a[permutation]
        shuffled_b = b[permutation]
        return shuffled_a, shuffled_b

    #################################################
    ### USEFULL FOR THIS TASK                     ###
    #################################################

    def read_psd_file(self, domain):
        read_in = self.psd_files[domain]

        if not os.path.isfile(read_in):
            self.create_psd()

        last_sent_id = -1
        data = {}
        source_words = []
        line_nb = 0
        with open(read_in, encoding='utf-8') as handle:
            for line in handle:
                sent_id, source_language_token_start, source_language_token_end, \
                target_language_token_start, target_language_token_end, \
                source_language_phrase, target_language_phrase = line.strip().split('\t')

                sent_id = int(sent_id)
                source_language_token_start = int(source_language_token_start)
                source_language_token_end = int(source_language_token_end)
                target_language_token_start = int(target_language_token_start)
                target_language_token_end = int(target_language_token_end)

                if sent_id not in data:
                    data[sent_id] = {}

                data[sent_id][(line_nb, source_language_token_start, source_language_token_end)] = (
                    source_language_phrase, target_language_phrase)

                last_sent_id = sent_id
                source_words.append(source_language_phrase)
                line_nb += 1

        return data, last_sent_id, source_words

    def get_next_line(self, path, split=None):
        while 1:
            with open(path, encoding='utf-8') as handle:
                for line in handle:
                    if split == False:
                        yield line.strip()
                    else:
                        yield line.strip().split(split)

    def write_to_file(self, path, content):
        out = open(path, 'a', encoding='utf-8')
        out.write(str(content) + '\n')
        out.close()

    def get_word_features(self, start_id, source_phrase, sentence, tagged_sentence, mode=None):

        distances = {-2: 'llcont', -1: 'lcont', 1: 'rcont', 2: 'rrcont'}

        word_features = {}

        word_features['current_word'] = source_phrase
        word_features['current_POS'] = tagged_sentence[start_id]['POS']
        word_features['current_lemma'] = tagged_sentence[start_id]['lemma']

        for distance in distances:
            try:
                assert start_id + distance >= 0
                word_features[distances[distance] + '_word'] = tagged_sentence[start_id + distance]['word']
                word_features[distances[distance] + '_POS'] = tagged_sentence[start_id + distance]['POS']
                word_features[distances[distance] + '_lemma'] = tagged_sentence[start_id + distance]['lemma']
            except (IndexError, AssertionError):
                word_features[distances[distance] + '_word'] = '0'
                word_features[distances[distance] + '_POS'] = '0'
                word_features[distances[distance] + '_lemma'] = '0'

        # if start_id > 0:
        #     word_features['-1_word'] = tagged_sentence[start_id - 1]['word']
        #     word_features['-1_POS'] = tagged_sentence[start_id - 1]['POS']
        #     word_features['-1_lemma'] = tagged_sentence[start_id - 1]['lemma']
        # else:
        #     word_features['-1_word'] = 0
        #     word_features['-1_POS'] = 0
        #     word_features['-1_lemma'] = 0
        #
        # if start_id < len(sentence)-1:
        #     word_features['+1_word'] = tagged_sentence[start_id + 1]['word']
        #     word_features['+1_POS'] = tagged_sentence[start_id + 1]['POS']
        #     word_features['+1_lemma'] = tagged_sentence[start_id + 1]['lemma']
        # else:
        #     word_features['+1_word'] = 0
        #     word_features['+1_POS'] = 0
        #     word_features['+1_lemma'] = 0

        word_features['-1+1_word'] = (word_features['lcont_word'], word_features['rcont_word'])
        word_features['-1+1_POS'] = (word_features['lcont_POS'], word_features['rcont_POS'])
        word_features['-1+1_lemma'] = (word_features['lcont_lemma'], word_features['rcont_lemma'])

        word_features['+1+2_word'] = (word_features['rcont_word'], word_features['rrcont_word'])
        word_features['+1+2_POS'] = (word_features['rcont_POS'], word_features['rrcont_POS'])
        word_features['+1+2_lemma'] = (word_features['rcont_lemma'], word_features['rrcont_lemma'])

        assert len(word_features) == 21

        return word_features

    def yield_pos_tags(self, path):
        sentence = []
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if line == '</s>':
                yield sentence
                sentence.clear()
            else:
                word, POS, lemma = line.split('\t')
                # lemma = lemma + '/' + POS[0]
                sentence.append({'word': word, 'POS': POS, 'lemma': lemma})
        return False

    def run_tagger(self, input_path, output_path, language):
        if self.verbose:
            print('\t- run tagger on {}'.format(input_path))
        parameter_file = {'fr': 'french',
                          'en': 'english'}

        # tree-tagger ../lib/french-utf8.par ../test.txt -token -lemma
        tagger_path = os.path.join(self.tagger_path, 'bin', 'tree-tagger')
        parameter_path = os.path.join(self.tagger_path, 'lib', '{}-utf8.par'.format(parameter_file[language]))
        tagger_command = '{tagger} {param} {input} -token -lemma -sgml'.format(tagger=tagger_path,
                                                                               param=parameter_path,
                                                                               input=input_path)
        tagger_output = open(output_path, 'w', encoding='utf-8')
        subprocess.call(tagger_command.split(), stdout=tagger_output)
        tagger_output.close()

    def create_word_per_line_format(self, input_path, output_path):

        if not self.check_file(output_path) or not self.use_existing:
            if self.verbose:
                print('\t- create word-per-line format')
            out = open(output_path, 'w', encoding='utf-8')
            with open(input_path, encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        sentence = line.split()
                        for word in sentence:
                            out.write(word + '\n')
                        out.write('</s>\n')

            out.close()
        else:
            if self.verbose:
                print('\t- use existing word-per-line file')

    def get_vw_feature_line(self, features, namespace=None):

        assert isinstance(features, dict)

        if not namespace:

            feature_line = ' ' + \
                           ' '.join([value
                                     if not isinstance(value, tuple)
                                     else '_'.join(list(value))
                                     for feature, value in sorted(features.items())])
            feature_line = feature_line.replace(':', '-')

        elif namespace == 'FEAT':
            feature_line = '|'.join(['{0} {1} '.format(feature, value)
                                     if not isinstance(value, tuple)
                                     else '{0} {1}'.format(feature, '_'.join(list(value)))
                                     for feature, value in sorted(features.items())])
            feature_line = feature_line.replace(':', '-')

        else:
            assert isinstance(namespace, str)
            feature_line = '{0} {1} '.format(namespace, ' '.join(['{0}:{1}'.format(feature, value)
                                                                  for feature, value in sorted(features.items())]))

        return feature_line
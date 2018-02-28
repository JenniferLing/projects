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
from feature_extractor import FeatureExtraction

class SenseSpottingClassifier(FeatureExtraction):
    def __init__(self, config_file=None, use_existing=False, verbose=False):
        super().__init__(config_file=config_file, use_existing=use_existing, verbose=verbose)

        print('\nCONFIGURATION of final classification...')
        print('- use features: {}'.format(', '.join([feat.lower()
                                                     for feat, val in
                                                     sorted(self.feature_extraction_configuration.items())
                                                     if val])
                                          ))
        if self.do_cv:
            repl = self.num_folds * self.repeat_cv
            print('- perform {0} fold cross validation with {1} repetitions ({2} replicates)'.format(self.num_folds,
                                                                                                     self.repeat_cv,
                                                                                                     repl))
        else:
            print('- perform singel training on defined features and data')

        if self.max_type_freq:
            print('- limit type frequency on {}'.format(self.max_type_freq))

        if self.set_bias:
            print('- add bias to features')
        if self.max_buckets:
            print('- bin features in {} equally heigth buckets'.format(self.max_buckets))

        if self.do_hyperopt:
            print('- optimize hyperparameters on development data')
        else:
            print('- no hyperparameter optimization => use development data for {}'.format(self.use_dev_for))

        if self.use_hold_out:
            print('- use vowpal wabbits holdout function to avoid overfitting while training')
        else:
            print('- no holdout data during training')

        if not self.seen_path:
            print("ERROR: need seen file!")
            sys.exit(1)

        assert self.seen_path.endswith('.gz')

        # self.check_memory(state='init final classifier')

        self.big_files_path = os.path.join(self.big_files_path, 'final_classifier')
        self.check_dir(self.big_files_path)

        self.config_file = config_file
        self.use_existing = use_existing

        self.feature_file_path = None
        self.feature_file_name = None

        self.label_distribution = {}

        self.cross_validation_dir = os.path.join(self.big_files_path, 'cross_validation')
        self.check_dir(self.cross_validation_dir)

        self.cross_validation_data_dir = os.path.join(self.big_files_path, 'cross_validation', 'data')
        self.check_dir(self.cross_validation_data_dir)

        self.cross_validation_model_dir = os.path.join(self.big_files_path, 'cross_validation', 'models')
        self.check_dir(self.cross_validation_model_dir)

        self.feature_name_mapping = {
            'TYPE_REL_FREQ': 'tyr',
            'TYPE_NGRAM_PROB': 'tyn',
            'TYPE_CONTEXT': 'tyc',
            'TYPE_TOPIC': 'tyt',
            'TOKEN_NGRAM_PROB': 'ton',
            'TOKEN_CONTEXT': 'toc',
            'TOKEN_CONTEXT_COUNT': 'tocc',
            'TOKEN_CONTEXT_PERCENTAGE': 'tocp',
            'TOKEN_PSD': 'top',
            'TOKEN_PSD_GLOBAL': 'topg',
            'TOKEN_PSD_GLOBAL_BINNED': 'topgb',
            'TOKEN_PSD_LOCAL': 'topl',
            'TOKEN_PSD_LOCAL_BINNED': 'toplb',
            'TOKEN_PSD_RATIO': 'topr',
        }

        self.experiment_name = 'AllFeatures'

    def extract_gold_labels(self, write=False, verbose=True):
        if self.gold_label == 'new_sense':
            return self.extract_new_sense_labels(write=write, verbose=verbose)
        elif self.gold_label == 'most_frequent_seen':
            return self.extract_most_frequent_seen_label()
        else:
            print('ERROR: gold label type: {} does not exist'.format(self.gold_label))
            sys.exit(1)

    def extract_new_sense_labels(self, write=False, verbose=True):

        new_psd_file = os.path.join(self.aux_path, '{}.psd.labels'.format(self.new_domain))

        old_senses = {}
        with gzip.open(self.seen_path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                fr_word, en_word, _ = line.strip().split('\t')
                if fr_word not in old_senses:
                    old_senses[fr_word] = []

                old_senses[fr_word].append(en_word)

        new_sense_counter = 0
        line_counter = 0
        gold_labels = {}

        if write:
            out = open(new_psd_file, 'w', encoding='utf-8')

        with open(self.psd_files[self.new_domain], encoding='utf-8') as handle:
            for line in handle:
                _, _, _, _, _, fr_word, en_word = line.strip().split('\t')

                if fr_word in old_senses:
                    old_sense = old_senses[fr_word]
                else:
                    continue

                if en_word in old_sense:

                    if write:
                        out.write(line.strip() + '\t-1\n')

                    gold_labels[line_counter] = -1
                else:
                    new_sense_counter += 1

                    if write:
                        out.write(line.strip() + '\t1\n')

                    gold_labels[line_counter] = 1

                line_counter += 1

        if self.verbose and verbose:
            print('- ratio new sense: {}'.format(round(new_sense_counter / float(line_counter), 4)))

        if write:
            out.close()

        return gold_labels

    def extract_most_frequent_seen_label(self):

        old_senses = {}
        with gzip.open(self.seen_path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                fr_word, en_word, prob = line.strip().split('\t')
                if fr_word not in old_senses:
                    old_senses[fr_word] = {}
                old_senses[fr_word][en_word] = prob

        most_frequent_senses = {}
        for fr_word in old_senses:
            max_prob = -1
            for prob in old_senses[fr_word].values():
                if prob > max_prob:
                    max_prob = prob

            if max_prob < 0:
                continue

            for en_word in old_senses[fr_word]:
                if old_senses[fr_word][en_word] >= max_prob:
                    if fr_word not in most_frequent_senses:
                        most_frequent_senses[fr_word] = []
                    most_frequent_senses[fr_word].append[en_word]

        gold_labels = {}
        line_counter = 0

        with open(self.psd_files[self.new_domain], encoding='utf-8') as handle:
            for line in handle:
                _, _, _, _, _, fr_word, en_word = line.strip().split('\t')

                if fr_word in most_frequent_senses:
                    most_frequent_sense = most_frequent_senses[fr_word]
                else:
                    continue

                if en_word in most_frequent_sense:
                    gold_labels[line_counter] = -1
                else:
                    gold_labels[line_counter] = 1

                line_counter += 1

        return gold_labels

    def get_feature(self, type_key, token_key):
        namespace = {}

        for feature in self.feature_extraction_configuration:
            if not self.feature_extraction_configuration[feature]:
                continue

            features = self.load_pickle_obj(self.feature_file_configuration[feature])

            if feature.startswith('TYPE'):
                namespace[feature] = features[type_key]
            elif feature.startswith('TOKEN'):
                namespace[feature] = features[token_key]

            del features
            gc.collect()

        return namespace

    def generate_feature_file_from_file(self):
        freq_counter = {}
        pos = 0
        neg = 0
        all = 0

        feature_names = [name.lower() for name, value in sorted(self.feature_extraction_configuration.items()) if value]

        feature_str = '{0}.{1}'.format('-'.join(feature_names), self.corpus_suffix)

        if feature_str in self.feature_file_mapping:
            filename = self.feature_file_mapping[feature_str]
        else:
            filename = '{0}.{1}'.format('_'.join([self.feature_name_mapping[feature.upper()]
                                                  for feature in sorted(feature_names)]),
                                        self.corpus_suffix[:-1])
            self.feature_file_mapping[feature_str] = filename

        output_path = os.path.join(self.big_files_path, filename)

        if self.check_file(output_path):
            if self.verbose:
                print('- feature file ({}) already exists'.format(output_path))
        else:
            if self.verbose:
                print('- write final features ({}) to file'.format(feature_names))
            data, last_sent_id, source_words = self.read_psd_file(self.new_domain)

            gold_labels = self.extract_gold_labels(write=True)

            generators = {}
            for feature in self.feature_extraction_configuration:
                if not self.feature_extraction_configuration[feature]:
                    continue

                generators[feature] = self.get_next_line(self.feature_file_configuration[feature])

            with open(output_path, mode='w', encoding='utf-8') as out:
                for sent_id in range(1, last_sent_id + 1):
                    if sent_id not in data:
                        continue

                    for line_nb, start, end in sorted(data[sent_id]):
                        source_word, target_word = data[sent_id][(line_nb, start, end)]

                        # if source_word == 'vue des enfants':
                        #     continue

                        namespaces = {feature: next(generators[feature]) for feature in generators}

                        if self.max_type_freq:
                            if source_word not in freq_counter:
                                freq_counter[source_word] = 0

                            if freq_counter[source_word] >= self.max_type_freq:
                                continue

                            freq_counter[source_word] += 1

                        label = gold_labels[line_nb]
                        tag = '{0}-{1}'.format(source_word, line_nb) if len(
                            source_word.split()) == 1 else '{0}-{1}'.format('_'.join(source_word.split()), line_nb)
                        feature_line = '{0} {1}'.format(label, tag)
                        for feature in namespaces:

                            if not namespaces[feature]:
                                continue
                            features = self.line_to_dict(namespaces[feature])
                            feature_line += '|' + self.get_vw_feature_line(features, namespace=feature)
                        feature_line += '\n'

                        out.write(feature_line)

                        if source_word not in self.label_distribution:
                            self.label_distribution[source_word] = {'pos': 0, 'neg': 0, 'all': 0}

                        if label > 0:
                            pos += 1
                            self.label_distribution[source_word]['pos'] += 1
                        else:
                            neg += 1
                            self.label_distribution[source_word]['neg'] += 1
                        all += 1
                        self.label_distribution[source_word]['all'] += 1

            if self.verbose:
                per = round(pos / all * 100, 2)
                if self.verbose:
                    print('- read {0} examples ({1} positive and {2} negative, '
                          'which is {3}% positive)'.format(all, pos, neg, per))

        self.feature_file_path = output_path
        self.feature_file_name = filename

    def line_to_dict(self, line):
        assert isinstance(line, list)
        return {feature.split(':')[0]: feature.split(':')[1] for feature in line}

    def check_num_folds(self):
        log_num_folds = int(np.log2(self.num_folds)) / np.log2(2)
        new_num_folds = int(2 ** log_num_folds)
        if self.verbose and new_num_folds != self.num_folds:
            print('- Warning: Using {0} instead of {1} folds for training '
                  '(need a power of 2 for splitting)'.format(new_num_folds, self.num_folds))

        self.num_folds = new_num_folds

    def get_label_distribution(self):

        gold_labels = self.extract_gold_labels(write=False, verbose=False)

        _, _, source_words = self.read_psd_file(self.new_domain)

        assert len(source_words) == len(gold_labels)

        for i in range(len(source_words)):
            source_word = source_words[i]

            if source_word not in self.label_distribution:
                self.label_distribution[source_word] = {'pos': 0, 'neg': 0, 'all': 0}

            if self.max_type_freq and self.label_distribution[source_word]['all'] >= self.max_type_freq:
                continue

            label = gold_labels[i]

            if label > 0:
                self.label_distribution[source_word]['pos'] += 1
            else:
                self.label_distribution[source_word]['neg'] += 1

            self.label_distribution[source_word]['all'] += 1

    def assign_folds(self, split_tree, current_fold):
        if 'TYPES' in split_tree:
            for word_type in split_tree['TYPES']:
                self.type_to_fold[word_type] = current_fold
            return current_fold + 1

        current_fold = self.assign_folds(split_tree['LEFT'], current_fold)
        current_fold = self.assign_folds(split_tree['RIGHT'], current_fold)

        return current_fold

    def get_type_to_fold_from_perl(self):

        # perl get_folds.pl -dom ~/sensespotting/orig_data/EMEA.psd -k 4
        # -seen ~/sensespotting/aux_files/seen.hansard.gz -max 100 -out ~/sensespotting/aux_files/type_to_fold.txt
        if self.verbose:
            print('- use perl script to get type-to-fold information')

        assert self.check_file(os.path.join(self.script_path, 'get_folds.pl'))

        out_file_path = os.path.join(self.aux_path, 'type_to_fold.txt')

        command = 'perl {script}/get_folds.pl -dom {psd} -k {folds} -seen {seen} -max {max} -out {out}'.format(
            script=self.script_path,
            psd=self.psd_files[self.new_domain],
            folds=self.num_folds,
            seen=self.seen_path,
            max=self.max_type_freq,
            out=out_file_path)

        # log_file = open(log_file_path, 'w', encoding='utf-8')
        # subprocess.call(command.split(), stdout=log_file, stderr=log_file)
        # log_file.close()
        subprocess.call(command.split())

        if self.check_file(out_file_path):
            with open(out_file_path, encoding='utf-8') as handle:
                for line in handle:
                    type, fold = line.strip().split('\t')
                    fold = int(fold)
                    assert fold <= self.num_folds
                    assert type in self.frtypes

                    self.type_to_fold[type] = fold
        else:
            raise FileNotFoundError('Fold information file was not created!')

    def get_fold_information(self, repetition):

        existing = False
        fold_info_path = os.path.join(self.cross_validation_dir, '{0}_fold_information_run_{1}'.format(self.num_folds,
                                                                                                       repetition))

        if self.check_file(fold_info_path + '.pkl') and self.use_existing:
            existing = True
            fold_info = self.load_pickle_obj(fold_info_path)

            if self.verbose:
                print('\nUSE existing {} fold information'.format(self.num_folds))

        else:
            if self.verbose:
                print('\nPARTITION dataset on {} folds'.format(self.num_folds))

            self.check_num_folds()

            if not self.label_distribution:
                self.get_label_distribution()

            self.type_to_fold = {}

            ### Temporary solution: run perl script of SenseSpotting inventors and read fold information from file!
            self.get_type_to_fold_from_perl()

            assert self.type_to_fold

            # print(self.type_to_fold)

            fold_info = {i: {'neg': 0,
                             'pos': 0,
                             'all': 0,
                             'types': set(),
                             'devfold': set(),
                             'testfold': set()}
                         for i in range(self.num_folds)}

            gold_labels = self.extract_gold_labels(write=False)
            _, _, source_words = self.read_psd_file(self.new_domain)

            for i in range(len(source_words)):
                word_type = source_words[i]
                if word_type not in self.type_to_fold:
                    raise ValueError('Type {} did not get a fold!'.format(word_type))

                fold = self.type_to_fold[word_type]

                assert fold <= self.num_folds
                assert fold in fold_info

                fold_info[fold]['neg'] += 0 if gold_labels[i] > 0 else 1
                fold_info[fold]['pos'] += 1 if gold_labels[i] > 0 else 0
                fold_info[fold]['all'] += 1
                fold_info[fold]['types'].add(word_type)
                fold_info[fold]['testfold'].add(word_type)

                next_fold = (fold + 1) % self.num_folds
                fold_info[next_fold]['devfold'].add(word_type)

            self.save_pickle_obj(fold_info, fold_info_path)

        return (existing, fold_info)

    def split_dataset(self, repetition, fold_information, data_dir):

        if self.verbose:
            print('\nSPLIT dataset', end=': ')

        use_old_fold_info = fold_information[0]
        fold_info = fold_information[1]

        fold_information_path = os.path.join(self.cross_validation_dir,
                                             '{0}_folds_type_and_count_information_run_{1}'.format(self.num_folds,
                                                                                                   repetition))

        for fold in range(self.num_folds):
            self.cross_validation_paths['training'][(repetition, fold)] = os.path.join(data_dir,
                                                                                       '{0}.{1}_{2}_fold.run_{3}.train'.format(
                                                                                           self.feature_file_name,
                                                                                           fold,
                                                                                           self.num_folds,
                                                                                           repetition))
            self.cross_validation_paths['dev'][(repetition, fold)] = os.path.join(data_dir,
                                                                                  '{0}.{1}_{2}_fold.run_{3}.dev'.format(
                                                                                      self.feature_file_name,
                                                                                      fold,
                                                                                      self.num_folds,
                                                                                      repetition))
            self.cross_validation_paths['test'][(repetition, fold)] = os.path.join(data_dir,
                                                                                   '{0}.{1}_{2}_fold.run_{3}.test'.format(
                                                                                       self.feature_file_name,
                                                                                       fold,
                                                                                       self.num_folds,
                                                                                       repetition))

        files = [fname for fname in os.listdir(data_dir)
                 if (fname.startswith(self.feature_file_name)
                     and '_{0}_fold.run_{1}'.format(self.num_folds, repetition) in fname
                     and not fname.endswith('.buckets'))]

        # if files already exist
        if len(files) == (
                    self.num_folds * len(self.cross_validation_paths)) and self.use_existing and use_old_fold_info:
            if self.verbose:
                print('found existing files, no new files are created')

            return True

        if self.verbose:
            print('create training, development and test files')

        if isinstance(fold_info, str):
            fold_info = self.load_pickle_obj(fold_info)

        _, _, source_words = self.read_psd_file(self.new_domain)
        type_counter = {}

        # add bias
        if self.set_bias:
            bias = '|BIAS bias:1'
        else:
            bias = ''

        fold_counter = {}

        for fold in range(self.num_folds):
            feature_file_content = self.get_next_line(self.feature_file_path, split=False)
            counter = {'training': {'counts': 0, 'types': set()},
                       'test': {'counts': 0, 'types': set()},
                       'dev': {'counts': 0, 'types': set()}}

            training_file = open(self.cross_validation_paths['training'][(repetition, fold)], mode='w',
                                 encoding='utf-8')
            dev_file = open(self.cross_validation_paths['dev'][(repetition, fold)], mode='w', encoding='utf-8')
            test_file = open(self.cross_validation_paths['test'][(repetition, fold)], mode='w', encoding='utf-8')

            for i in range(len(source_words)):
                source_word = source_words[i]
                if source_word not in type_counter:
                    type_counter[source_word] = 0

                if self.max_type_freq and type_counter[source_word] >= self.max_type_freq:
                    continue
                type_counter[source_word] += 1

                # print(i)
                feature_line = next(feature_file_content)

                if source_word in fold_info[fold]['testfold']:
                    test_file.write(feature_line + bias + '\n')
                    counter['test']['counts'] += 1
                    counter['test']['types'].add(source_word)
                elif source_word in fold_info[fold]['devfold']:
                    dev_file.write(feature_line + bias + '\n')
                    counter['dev']['counts'] += 1
                    counter['dev']['types'].add(source_word)
                else:
                    training_file.write(feature_line + bias + '\n')
                    counter['training']['counts'] += 1
                    counter['training']['types'].add(source_word)

            training_file.close()
            dev_file.close()
            test_file.close()

            for dset in counter:
                if counter[dset]['counts'] == 0:
                    return False

            if self.verbose:
                print("- fold {0}: {1} training types ({2} examples), {3} dev types ({4} examples), " \
                      "{5} test types ({6} examples)".format(fold + 1,
                                                             len(counter['training']['types']),
                                                             counter['training']['counts'],
                                                             len(counter['dev']['types']),
                                                             counter['dev']['counts'],
                                                             len(counter['test']['types']),
                                                             counter['test']['counts']))

            fold_counter[fold] = counter

        self.save_pickle_obj(fold_counter, fold_information_path)

        return True

    def make_buckets(self, data_path):

        feature_vals = {}
        with open(data_path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip().split('|')

                for namespace in line[1:]:
                    namespace = namespace.split()
                    namespace_name = namespace[0]

                    for feature in namespace[1:]:
                        assert ':' in feature
                        feature = feature.strip()
                        fname, val = feature.split(':')
                        val = float(val)

                        key = "{0}--{1}".format(namespace_name, fname)
                        if key not in feature_vals:
                            feature_vals[key] = {}

                        if val not in feature_vals[key]:
                            feature_vals[key][val] = 0

                        feature_vals[key][val] += 1

            for line in handle:
                for fname in feature_vals:
                    _, feature_name = fname.split('--')
                    if feature_name not in line:
                        feature_vals[fname][0] += 1

        bucket_info = {}

        for fname in feature_vals:
            l = sorted(np.hstack([[val] * feature_vals[fname][val] for val in feature_vals[fname]]))
            bin_edges = scipy.stats.mstats.mquantiles(l,
                                                      prob=np.array(range(0, self.max_buckets, 1)) / self.max_buckets)
            assert len(bin_edges) == self.max_buckets
            bucket_info[fname] = sorted(set(bin_edges))

            # i = 0
            #
            # # step = int(len(l)/self.max_buckets)
            # step = round(len(l)/self.max_buckets)
            #
            # if step < 1:
            #     step = 1
            #
            # while i < len(l):
            #     val = l[i]
            #
            #     if fname not in bucket_info:
            #         bucket_info[fname] = []
            #
            #     bucket_info[fname].append(val)
            #
            #     i += step
            #

        return bucket_info

    def apply_buckets(self, bucket_info, input_path, missing_buckets):

        out_path = input_path + '.buckets'

        out = open(out_path, mode='w', encoding='utf-8')

        with open(input_path, encoding='utf-8') as handle:
            for line in handle:
                new_line = ''
                line = line.strip().split('|')
                new_line += line[0]

                for namespace in line[1:]:
                    namespace = namespace.split()
                    new_line += '|{} '.format(namespace[0])
                    namespace_name = namespace[0]

                    for feature in namespace[1:]:
                        assert ':' in feature
                        feature = feature.strip()
                        fname, val = feature.split(':')

                        key = "{0}--{1}".format(namespace_name, fname)

                        if key not in bucket_info:
                            if missing_buckets:
                                print('WARNING: no bucket info for {} --skipping'.format(key))
                            continue

                        val = float(val)

                        for border_val in bucket_info[key]:
                            if val >= border_val:
                                # new_line += '{0}>={1} '.format(fname, border_val)
                                new_line += '{0}-ge:{1} '.format(fname, border_val)

                out.write(new_line + '\n')

        out.close()

        return out_path

    def use_buckets(self, current_repetition, current_fold, data_dir):
        if current_fold == 0:
            files = [fname for fname in os.listdir(data_dir)
                     if (fname.startswith(self.feature_file_name)
                         and '_{0}_fold.run_{1}'.format(self.num_folds, current_repetition) in fname
                         and fname.endswith('.buckets'))]

            # if files already exist
            if len(files) == (self.num_folds * len(self.cross_validation_paths)) and self.use_existing:
                if self.verbose:
                    print('- found existing files, no new files are created')

                for dset in self.cross_validation_paths:
                    for fold in range(self.num_folds):
                        if not self.cross_validation_paths[dset][(current_repetition, fold)].endswith('.buckets'):
                            self.cross_validation_paths[dset][(current_repetition, fold)] = \
                                self.cross_validation_paths[dset][(current_repetition, fold)] + '.buckets'

                return True

            if self.verbose:
                print('- create bucket dataset')

        bucket_info = self.make_buckets(self.cross_validation_paths['training'][(current_repetition, current_fold)])
        for dset in self.cross_validation_paths:
            missing_buckets = 1 if dset == 'training' else 0
            self.cross_validation_paths[dset][(current_repetition, current_fold)] = self.apply_buckets(bucket_info,
                                                                                                       self.cross_validation_paths[
                                                                                                           dset][
                                                                                                           (
                                                                                                           current_repetition,
                                                                                                           current_fold)],
                                                                                                       missing_buckets)

        return False

    def hyperparameter_opt(self, training_path, dev_path, param_to_optimize, min_val, max_val):

        # log_path = os.path.join(self.cross_validation_dir, 'logs')
        # self.check_dir(log_path)
        # log_file_path = os.path.join(log_path, 'hyperparam_opt_{}.log'.format(int(time.time())))
        if self.verbose:
            print('\t- optimize {} hyperparameter'.format(param_to_optimize.strip('-')), end=' => ')

        command = 'sh {script}/hyperparameter_opt.sh {vw} {train} {dev} {param} {min} {max}'.format(
            script=self.script_path,
            vw=self.vowpal_wabbit_path,
            train=training_path,
            dev=dev_path,
            param=param_to_optimize,
            min=min_val,
            max=max_val)

        # log_file = open(log_file_path, 'w',
        #                 encoding='utf-8')
        # subprocess.call(command.split(), stdout=log_file, stderr=log_file)
        # log_file.close()

        FNULL = open(os.devnull, 'w')
        res = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=FNULL).communicate()[0]
        optimal_value = str(res, 'utf-8').strip().split('\t')[0]

        if self.verbose:
            print("{0} = {1}".format(param_to_optimize.strip('-'), optimal_value))

        return optimal_value

    def train_vw_model(self, working_dir, training_path, dev_path, model_path, args, patience, log_file_path,
                       cache_name):

        if self.verbose:
            print('\t- train vw model')

        assert self.check_file(training_path)
        if dev_path != '_':
            assert self.check_file(dev_path)
        assert self.check_file(os.path.join(self.script_path, 'train_vw_model.sh'))

        command = 'sh {0}/train_vw_model.sh {1} {2} {3} {4} {5} {6} {7} {8}'.format(self.script_path,
                                                                                    self.vowpal_wabbit_path,
                                                                                    working_dir,
                                                                                    training_path,
                                                                                    dev_path,
                                                                                    model_path,
                                                                                    patience,
                                                                                    cache_name,
                                                                                    args)

        log_file = open(log_file_path, 'w', encoding='utf-8')
        subprocess.call(command.split(), stdout=log_file, stderr=log_file)
        log_file.close()
        # subprocess.call(command.split())

        assert self.check_file(model_path)

    def test_vw_model(self, test_data, model_path, dev_data):

        if self.verbose:
            print('\t- test vw model')

        assert self.check_file(test_data)
        assert self.check_file(model_path)

        command = 'sh {0}/test_vw_model.sh {1} {2} {3} {4} {5}'.format(self.script_path, self.vowpal_wabbit_path,
                                                                       model_path,
                                                                       test_data, "--binary", dev_data)

        probs = subprocess.Popen(command.split(), stdout=subprocess.PIPE).communicate()[0]
        try:
            probs = str(probs, 'utf-8').split('\n')
            probs = [int(prob.split()[0]) for prob in probs if prob]
        except:
            print(probs)
            sys.exit(1)

        return probs

    def random_baseline_prediction(self, nb_decisions, possible_labels=[-1, 1]):
        return [random.choice(possible_labels) for _ in range(nb_decisions)]

    def constant_baseline_prediction(self, nb_decisions):
        return [1] * nb_decisions

    def type_oracle(self, word_type):

        word_type = word_type.replace('_', ' ')

        type_dist = self.label_distribution[word_type]
        if type_dist['pos'] > type_dist['neg']:
            return 1
        elif type_dist['pos'] < type_dist['neg']:
            return -1
        else:
            # if self.verbose:
            #     print('\t- WARNING: no majority label exists for {}'.format(word_type))
            self.no_majority_label.add(word_type)
            return -1

    def type_oracle_prediction(self, word_types):
        if not self.label_distribution:
            self.get_label_distribution()

        return [self.type_oracle(word_type) for word_type in word_types]

    def cross_validation(self):
        cache_filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

        eval_dir = os.path.join(self.cross_validation_dir, "evaluation", self.corpus_suffix.replace('.', '-')[:-1],
                                self.feature_file_name.split('.')[0])
        self.check_dir(eval_dir)
        model_path = os.path.join(self.cross_validation_model_dir, self.corpus_suffix.replace('.', '-')[:-1],
                                  self.feature_file_name.split('.')[0])
        self.check_dir(model_path)

        log_path = os.path.join(self.cross_validation_dir, 'logs', self.corpus_suffix.replace('.', '-')[:-1],
                                self.feature_file_name.split('.')[0])
        self.check_dir(log_path)

        data_dir = os.path.join(self.cross_validation_data_dir, self.corpus_suffix.replace('.', '-')[:-1],
                                self.feature_file_name.split('.')[0])
        self.check_dir(data_dir)

        feature_scores = {'acc': [], 'roc': [],
                          'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                          'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        random_scores = {'acc': [], 'roc': [],
                         'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                         'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        constant_scores = {'acc': [], 'roc': [],
                           'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                           'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        oracle_scores = {'acc': [], 'roc': [],
                         'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                         'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        self.no_majority_label = set()
        self.zero_performance_word_types = {}

        self.cross_validation_paths = {'training': {}, 'dev': {}, 'test': {}}
        for nb_repetitions in range(self.repeat_cv):
            result = False
            while not result:
                fold_info = self.get_fold_information(nb_repetitions)
                result = self.split_dataset(nb_repetitions, fold_info, data_dir)
                if not result:
                    self.use_existing = 0
            self.use_existing = 1

        assert len(self.cross_validation_paths['training']) == self.repeat_cv * self.num_folds

        for nb_repetitions in range(self.repeat_cv):

            bucket_files_exit = False

            for fold in range(self.num_folds):
                args = {'--l1': None, '--l2': None, '-l': None, '--passes': None}

                if self.max_buckets and not bucket_files_exit:  # bucket_files_exist[fold]:
                    if fold == 0 and self.verbose:
                        print('\nUSE buckets for training')

                    bucket_files_exit = self.use_buckets(nb_repetitions, fold, data_dir)

                if self.verbose and fold == 0 and nb_repetitions == 0:
                    print('\nSTART cross validation...')

                train_data = self.cross_validation_paths['training'][(nb_repetitions, fold)]
                dev_data = self.cross_validation_paths['dev'][(nb_repetitions, fold)]
                test_data = self.cross_validation_paths['test'][(nb_repetitions, fold)]

                _, fname = os.path.split(train_data)

                suffix = '{0}.run_{1}'.format(fname, nb_repetitions)
                eval_file = open(os.path.join(eval_dir, '{}.eval'.format(suffix)),
                                 mode='w', encoding='utf-8')
                model_file = os.path.join(model_path, '{}.vw'.format(suffix))
                log_file_path = os.path.join(log_path, 'train_vw_model.{}.log'.format(suffix))

                if self.verbose:
                    print('- process run {0}/{1}, fold {2}/{3}'.format(nb_repetitions + 1, self.repeat_cv,
                                                                       fold + 1, self.num_folds))

                # number_of_training_examples = sum(
                #     1 for _ in open(self.cross_validation_paths['training'][(nb_repetitions, fold)]))
                # max_reg = 10 / number_of_training_examples
                # step_reg = max_reg / 5

                if self.do_hyperopt:

                    # # Tune L1 regularization on dataset
                    # args['--l1'] = self.hyperparameter_opt(train_data,
                    #                                        dev_data,
                    #                                        '--l1', 0, max_reg)

                    # Tune L2 regularization on dataset
                    args['--l2'] = self.hyperparameter_opt(train_data,
                                                           dev_data,
                                                           '--l2', -1, 1)

                    # # Tune learning rate on dataset
                    # args['-l'] = self.hyperparameter_opt(self.cross_validation_paths['training'][fold],
                    #                                      self.cross_validation_paths['dev'][fold],
                    #                                      '-l', 0, 10)
                    #
                    # # Tune number of iterations on dataset
                    # args['--passes'] = self.hyperparameter_opt(self.cross_validation_paths['training'][fold],
                    #                                            self.cross_validation_paths['dev'][fold],
                    #                                            '--passes', 1, 30)
                    dev_data = ""
                else:
                    args['--passes'] = 20

                args_string = ''
                for arg, val in sorted(args.items()):
                    if val:
                        args_string += '{0} {1} '.format(arg, val)

                if not self.use_hold_out:
                    args_string += '--holdout_off'

                if dev_data:
                    assert self.use_dev_for in ['test', 'training']
                    if self.use_dev_for == 'test':
                        dev = ("_", dev_data)
                    else:
                        dev = (dev_data, "_")
                else:
                    dev = ("_", "_")

                self.train_vw_model(self.cross_validation_dir,
                                    train_data,
                                    dev[0],
                                    model_file,
                                    args_string,
                                    5,
                                    log_file_path,
                                    cache_filename)

                predictions = self.test_vw_model(test_data, model_file, dev[1])

                if dev[1] == "_":
                    label_info = [line.split('|')[0] for line in open(test_data, encoding='utf-8')]
                else:
                    label_info = [line.split('|')[0] for line in open('{}.test_dev'.format(test_data),
                                                                      encoding='utf-8')]

                gold_labels, source_words = zip(*[label.split() for label in label_info])
                gold_labels = [int(label) for label in gold_labels]
                source_words = [word.split('-')[0] for word in source_words]

                assert len(predictions) == len(gold_labels) == len(source_words)

                nb_pred = len(predictions)
                random_prediction = self.random_baseline_prediction(nb_pred)
                constant_prediction = self.constant_baseline_prediction(nb_pred)
                oracle_prediciton = self.type_oracle_prediction(source_words)

                assert set(constant_prediction) == {1}

                assert len(predictions) == len(random_prediction) == len(constant_prediction) == len(oracle_prediciton)

                weighted_micro = False

                feature_scores = self.compute_performance(feature_scores, gold_labels, predictions, source_words,
                                                          weighted_micro=weighted_micro)
                random_scores = self.compute_performance(random_scores, gold_labels, random_prediction, source_words,
                                                         weighted_micro=weighted_micro)
                constant_scores = self.compute_performance(constant_scores, gold_labels, constant_prediction,
                                                           source_words, weighted_micro=weighted_micro)
                oracle_scores = self.compute_performance(oracle_scores, gold_labels, oracle_prediciton, source_words,
                                                         weighted_micro=weighted_micro)

                eval_file.write('prediction\trandom\tconstant\toracle\tgold_label\n')
                eval_file.write("\n".join(["{0}\t{1}\t{2}\t{3}\t{4}".format(pred, random, constant, oracle, gold)
                                           for word, pred, random, constant, oracle, gold in zip(source_words,
                                                                                                 predictions,
                                                                                                 random_prediction,
                                                                                                 constant_prediction,
                                                                                                 oracle_prediciton,
                                                                                                 gold_labels)]))
                eval_file.write('\n')
                eval_file.close()

        if self.verbose:
            print('No majority label word types: {}'.format(self.no_majority_label))
            print('Word types with tp of 0: {}'.format(self.zero_performance_word_types))

        print('\nPERFORMANCE averages (weighted micro: {})'.format(weighted_micro))
        metrics = ['roc', 'prec_macro', 'rec_macro', 'f1_macro', 'prec_micro', 'rec_micro', 'f1_micro']
        self.save_pickle_obj(feature_scores, 'feature_scores_{}'.format(self.experiment_name.lower().replace(':', '_')))

        for metric in metrics:
            assert len(feature_scores[metric]) == (self.num_folds * self.repeat_cv)

            print('- {0} = {1} (+/- {2})'.format(metric,
                                                 round(np.mean(feature_scores[metric]), 2),
                                                 round(np.std(feature_scores[metric]), 2)))

        return

    def compute_performance(self, scores, gold_labels, predictions, source_words, weighted_micro=False, verbose=False):

        acc = accuracy_score(gold_labels, predictions) * 100

        roc = roc_auc_score(gold_labels, predictions) * 100
        prec_macro, rec_macro, f1_macro = self.get_confusion_matrix('<unk>', gold_labels, predictions)
        prec_micro, rec_micro, f1_micro = self.compute_micro_performance(gold_labels, predictions, source_words,
                                                                         weighted=weighted_micro)

        if self.verbose and verbose:
            print("\t- acc = {0}, prec_macro = {1}, rec_macro = {2}, f1_macro = {3}, f1_micro = {4}, "
                  "prec_micro: {5}, rec_micro: {6}, roc: {7}\n".format(acc,
                                                                       prec_macro,
                                                                       rec_macro,
                                                                       f1_macro,
                                                                       f1_micro,
                                                                       prec_micro,
                                                                       rec_micro,
                                                                       roc))
        scores['acc'].append(acc)
        scores['prec_macro'].append(prec_macro)
        scores['rec_macro'].append(rec_macro)
        scores['f1_macro'].append(f1_macro)
        scores['roc'].append(roc)

        scores['f1_micro'].append(f1_micro)
        scores['prec_micro'].append(prec_micro)
        scores['rec_micro'].append(rec_micro)

        return scores

    def compute_micro_performance(self, gold_labels, predictions, source_words, weighted=False):
        word_predictions = {}
        word_weight = {}
        for i in range(len(source_words)):
            word = source_words[i]
            word_weight[word] = word_weight.get(word, 0) + 1

            if word not in word_predictions:
                word_predictions[word] = {'gold': [], 'pred': []}

            word_predictions[word]['gold'].append(gold_labels[i])
            word_predictions[word]['pred'].append(predictions[i])

        performance = {'f1': [], 'prec': [], 'rec': []}
        for word in word_predictions:
            gold = word_predictions[word]['gold']
            pred = word_predictions[word]['pred']

            prec, rec, f1 = self.get_confusion_matrix(word, gold, pred)

            if weighted:
                for _ in range(word_weight[word]):
                    performance['f1'].append(f1)
                    performance['prec'].append(prec)
                    performance['rec'].append(rec)
            else:
                performance['f1'].append(f1)
                performance['prec'].append(prec)
                performance['rec'].append(rec)

        return np.mean(performance['prec']), np.mean(performance['rec']), np.mean(performance['f1'])

    def get_confusion_matrix(self, word, gold, pred):

        assert len(pred) == len(gold)

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(len(pred)):
            if pred[i] == 1 and gold[i] == 1:
                tp += 1
            elif pred[i] == 1 and gold[i] == -1:
                fp += 1
            elif pred[i] == -1 and gold[i] == 1:
                fn += 1
            elif pred[i] == -1 and gold[i] == -1:
                tn += 1
            else:
                raise ValueError('Prediction not binary!')

        #######################################
        if tp == 0:
            self.zero_performance_word_types[word] = self.zero_performance_word_types.get(word, 0) + 1
            return 0, 0, 0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        ########################################
        # if tp == 0:
        #     self.zero_performance_word_types[word] = self.zero_performance_word_types.get(word, 0) + 1
        #     if fp == 0:
        #         prec = 1
        #     else:
        #         prec = 0
        #
        #     if fn == 0:
        #         rec = 1
        #     else:
        #         rec = 0
        # else:
        #     prec = tp / (tp + fp)
        #     rec = tp / (tp + fn)
        ######################################

        if prec == 0 or rec == 0:
            f1 = 0
        else:
            f1 = (2 * prec * rec) / (prec + rec)

        return prec * 100, rec * 100, f1 * 100

    def simple_training(self):

        if self.verbose:
            print('\nTRAIN classifier...')

        cache_filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

        eval_dir = os.path.join(self.cross_validation_dir, "evaluation", self.corpus_suffix.replace('.', '-')[:-1],
                                self.feature_file_name.split('.')[0])
        self.check_dir(eval_dir)
        model_path = os.path.join(self.cross_validation_model_dir, self.corpus_suffix.replace('.', '-')[:-1],
                                  self.feature_file_name.split('.')[0])
        self.check_dir(model_path)

        log_path = os.path.join(self.cross_validation_dir, 'logs', self.corpus_suffix.replace('.', '-')[:-1],
                                self.feature_file_name.split('.')[0])
        self.check_dir(log_path)

        data_dir = os.path.join(self.cross_validation_data_dir, self.corpus_suffix.replace('.', '-')[:-1],
                                self.feature_file_name.split('.')[0])
        self.check_dir(data_dir)

        feature_scores = {'acc': [], 'roc': [],
                          'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                          'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        random_scores = {'acc': [], 'roc': [],
                         'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                         'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        constant_scores = {'acc': [], 'roc': [],
                           'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                           'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        oracle_scores = {'acc': [], 'roc': [],
                         'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                         'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

        self.no_majority_label = set()
        self.zero_performance_word_types = {}

        self.cross_validation_paths = {'training': {}, 'dev': {}, 'test': {}}

        random_repetition = 0
        result = False
        while not result:
            fold_info = self.get_fold_information(random_repetition)
            result = self.split_dataset(random_repetition, fold_info, data_dir)
            if not result:
                self.use_existing = 0
        self.use_existing = 1

        assert len(self.cross_validation_paths['training']) == self.repeat_cv * self.num_folds

        random_fold = random.choice(range(self.num_folds))

        args = {'--l1': None, '--l2': None, '-l': None, '--passes': None}

        train_data = self.cross_validation_paths['training'][(random_repetition, random_fold)]
        dev_data = self.cross_validation_paths['dev'][(random_repetition, random_fold)]
        test_data = self.cross_validation_paths['test'][(random_repetition, random_fold)]

        if self.max_buckets:
            if self.verbose:
                print('- use buckets for training')

            bucket_info = self.make_buckets(train_data)
            train_data = self.apply_buckets(bucket_info, train_data, 1)
            dev_data = self.apply_buckets(bucket_info, dev_data, 0)
            test_data = self.apply_buckets(bucket_info, test_data, 0)

        _, fname = os.path.split(train_data)

        eval_file = open(os.path.join(eval_dir, '{}.eval'.format(fname)),
                         mode='w', encoding='utf-8')
        model_file = os.path.join(model_path, '{}.vw'.format(fname))
        log_file_path = os.path.join(log_path, 'train_vw_model.{}.log'.format(fname))



        if self.do_hyperopt:
            # # Tune L1 regularization on dataset
            # args['--l1'] = self.hyperparameter_opt(train_data,
            #                                        dev_data,
            #                                        '--l1', 0, max_reg)

            # Tune L2 regularization on dataset
            args['--l2'] = self.hyperparameter_opt(train_data,
                                                   dev_data,
                                                   '--l2', -1, 1)

            # # Tune learning rate on dataset
            # args['-l'] = self.hyperparameter_opt(self.cross_validation_paths['training'][fold],
            #                                      self.cross_validation_paths['dev'][fold],
            #                                      '-l', 0, 10)
            #
            # # Tune number of iterations on dataset
            # args['--passes'] = self.hyperparameter_opt(self.cross_validation_paths['training'][fold],
            #                                            self.cross_validation_paths['dev'][fold],
            #                                            '--passes', 1, 30)
            dev_data = ""
        else:
            args['--passes'] = 20

        args_string = ''
        for arg, val in sorted(args.items()):
            if val:
                args_string += '{0} {1} '.format(arg, val)

        if not self.use_hold_out:
            args_string += '--holdout_off'

        if dev_data:
            assert self.use_dev_for in ['test', 'training']
            if self.use_dev_for == 'test':
                dev = ("_", dev_data)
            else:
                dev = (dev_data, "_")
        else:
            dev = ("_", "_")

        self.train_vw_model(self.cross_validation_dir,
                            train_data,
                            dev[0],
                            model_file,
                            args_string,
                            5,
                            log_file_path,
                            cache_filename)

        predictions = self.test_vw_model(test_data, model_file, dev[1])

        if dev[1] == "_":
            label_info = [line.split('|')[0] for line in open(test_data, encoding='utf-8')]
        else:
            label_info = [line.split('|')[0] for line in open('{}.test_dev'.format(test_data),
                                                              encoding='utf-8')]

        gold_labels, source_words = zip(*[label.split() for label in label_info])
        gold_labels = [int(label) for label in gold_labels]
        source_words = [word.split('-')[0] for word in source_words]

        assert len(predictions) == len(gold_labels) == len(source_words)

        nb_pred = len(predictions)
        random_prediction = self.random_baseline_prediction(nb_pred)
        constant_prediction = self.constant_baseline_prediction(nb_pred)
        oracle_prediciton = self.type_oracle_prediction(source_words)

        assert set(constant_prediction) == {1}

        assert len(predictions) == len(random_prediction) == len(constant_prediction) == len(oracle_prediciton)

        weighted_micro = False

        feature_scores = self.compute_performance(feature_scores, gold_labels, predictions, source_words,
                                                  weighted_micro=weighted_micro)
        random_scores = self.compute_performance(random_scores, gold_labels, random_prediction, source_words,
                                                 weighted_micro=weighted_micro)
        constant_scores = self.compute_performance(constant_scores, gold_labels, constant_prediction,
                                                   source_words, weighted_micro=weighted_micro)
        oracle_scores = self.compute_performance(oracle_scores, gold_labels, oracle_prediciton, source_words,
                                                 weighted_micro=weighted_micro)

        eval_file.write('prediction\trandom\tconstant\toracle\tgold_label\n')
        eval_file.write("\n".join(["{0}\t{1}\t{2}\t{3}\t{4}".format(pred, random, constant, oracle, gold)
                                   for word, pred, random, constant, oracle, gold in zip(source_words,
                                                                                         predictions,
                                                                                         random_prediction,
                                                                                         constant_prediction,
                                                                                         oracle_prediciton,
                                                                                         gold_labels)]))
        eval_file.write('\n')
        eval_file.close()

        if self.verbose:
            print('No majority label word types: {}'.format(self.no_majority_label))
            print('Word types with tp of 0: {}'.format(self.zero_performance_word_types))

        # only print statement which is always printed!
        print('\nPERFORMANCE averages (weighted micro: {})'.format(weighted_micro))
        metrics = ['roc', 'prec_macro', 'rec_macro', 'f1_macro', 'prec_micro', 'rec_micro', 'f1_micro']
        self.save_pickle_obj(feature_scores, 'feature_scores_{}'.format(self.experiment_name.lower().replace(':', '_')))

        for metric in metrics:
            print('- {0} = {1}'.format(metric, round(feature_scores[metric], 2)))


        return

    def run(self):

        self.extract_features()

        if 'TOKEN_CONTEXT' in self.feature_extraction_configuration:
            self.feature_extraction_configuration['TOKEN_CONTEXT'] = 0

        if 'TOKEN_PSD' in self.feature_extraction_configuration:
            self.feature_extraction_configuration['TOKEN_PSD'] = 0

        if self.verbose:
            print('\nGENERATE general feature file...')

        feature_file_mapping_path = os.path.join(self.aux_path, 'feature_file.mapping')
        if self.check_file(feature_file_mapping_path):
            self.feature_file_mapping = self.load_pickle_obj(feature_file_mapping_path)
        else:
            self.feature_file_mapping = {}

        self.generate_feature_file_from_file()

        self.save_pickle_obj(self.feature_file_mapping, feature_file_mapping_path)

        if self.do_cv:
            self.cross_validation()
        else:
            self.simple_training()

    def perform_ablation_study(self):

        for feature in sorted(self.feature_file_configuration):

            self.feature_extraction_configuration[feature] = 1

        for feature in self.feature_file_configuration:

            print('\nPERFORM ablation study: AllFeature - {}'.format(feature.lower()))
            self.experiment_name = feature.lower()

            self.feature_extraction_configuration[feature] = 0

            self.run()

            self.feature_extraction_configuration[feature] = 1

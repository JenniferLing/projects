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

class PSDClassifier(Configurator):
    def __init__(self, domain=None, language=None, config_file=None, use_existing=False, verbose=False):
        super().__init__(config_file=config_file, use_existing=use_existing, verbose=verbose)

        if self.verbose:
            print('\nSTART processing of PSD classifier...')

        self.check_memory(state='init psd classifier')

        self.big_files_path = os.path.join(self.big_files_path, 'psd_classifier')
        self.check_dir(self.big_files_path)

        if not domain:
            self.domain = self.old_domain
            self.language = self.source_language
        else:
            self.domain = domain
            self.language = language

        self.cleaned_corpus_path = os.path.join(self.corpus_path, self.domain)
        self.check_dir(self.cleaned_corpus_path)

        self.max_length = 80

        self.check_memory(state='leave init psd classifier')

        self.psd_tagged_corpus_file = None

    def read_phrase_table(self):
        if self.verbose:
            print('\nEXTRACT phrase table...')

        output_path = os.path.join(self.big_files_path, 'phrase_table.psd.frtypes')

        if not self.check_file(output_path + '.pkl'):
            if self.verbose:
                print('- derive phrase table from file')

            phrase_table = dict()
            with gzip.open(self.phrase_table_file, 'rt', encoding='utf-8') as handle:
                for line in handle:
                    sent_id, start_fr, end_fr, start_en, end_en, fr_word, en_word = line.strip().split('\t')

                    if fr_word not in self.frtypes:
                        continue

                    sent_id = int(sent_id)
                    start_fr = int(start_fr)
                    end_fr = int(end_fr)

                    # use only words not phrases!
                    if start_fr != end_fr:
                        continue

                    if sent_id not in phrase_table:
                        phrase_table[sent_id] = {}

                    phrase_table[sent_id][start_fr] = (fr_word, en_word)

            self.save_pickle_obj(phrase_table, output_path)
        else:
            if self.verbose:
                print('- use existing phrase table')

        self.check_memory(state='finished reading phrase table')

    def preprocess_for_feature_extraction(self):
        if self.verbose:
            print('\nCHECK preprocessing for PSD classifier...')

        self.clean_corpus()

        if not self.check_file(self.psd_tagged_corpus_file):
            self.tag_corpus()
        else:
            if self.verbose:
                print('- tagged corpus was provided...')

        self.check_phrase_table()

    def clean_corpus(self):

        # define needed parameters
        filename = self.corpus_file_name[:-1] if self.corpus_file_name.endswith('.') else self.corpus_file_name
        self.cleaned_corpus_file = os.path.join(self.cleaned_corpus_path, filename + '.cleaned')

        if not os.path.isfile('.'.join([self.cleaned_corpus_file, self.language])) or not self.use_existing:
            # clean corpus for Moses (using Moses script)
            if self.verbose:
                print('- clean {} corpus'.format(self.domain))

            script_path = os.path.join(self.moses_path, 'scripts', 'training')
            cleaning_script_path = os.path.join(script_path, 'clean-corpus-n.perl')
            corpus = os.path.join(self.corpus_path, self.domain, filename)

            clean_command = '{SCRIPT} {INPUT} {F_EXT} {E_EXT} {OUTPUT} 0 {MAX_LENGTH}'.format(
                SCRIPT=cleaning_script_path,
                INPUT=corpus,
                F_EXT=self.source_language,
                E_EXT=self.target_language,
                OUTPUT=self.cleaned_corpus_file,
                MAX_LENGTH=self.max_length)

            subprocess.call(clean_command.split(), stdout=None)
        else:
            if self.verbose:
                print('- use existing cleaned corpus')

    def tag_corpus(self):
        # if self.verbose:
        #     print('\nTAGGING of corpus...')

        tagger_input_path = '.'.join([self.cleaned_corpus_file, 'word_per_line', self.language])
        tagger_output_path = '.'.join([self.cleaned_corpus_file, 'word_per_line', 'tagged', self.language])
        self.psd_tagged_corpus_file = tagger_output_path

        if os.path.isfile(tagger_output_path) and self.use_existing:
            if self.verbose:
                print('- use existing tagged corpus')
        else:
            if not os.path.isfile(tagger_input_path) or not self.use_existing:
                word_per_line_input_path = '.'.join([self.cleaned_corpus_file, self.language])
                self.create_word_per_line_format(word_per_line_input_path, tagger_input_path)

            if self.verbose:
                print('- create tagged corpus')

            self.run_tagger(tagger_input_path, tagger_output_path, self.language)

    def get_alignment(self):
        if self.check_file(self.alignment_file):
            if self.verbose:
                print('- alignment for {} was provided'.format(self.domain))
        else:
            self.build_alignment()

    def build_alignment(self):

        alignment_path = os.path.join(self.phrase_table_path, self.domain, 'alignment')
        self.check_dir(alignment_path)

        aligned_corpus_file = os.path.join(alignment_path, '{}.fr-en'.format(self.domain))

        def create_aligned_corpus(input_path, output_path):

            input_path = input_path if input_path.endswith('.') else input_path + '.'

            out = open(output_path, 'w', encoding='utf-8')

            sentences_source_language = open(input_path + self.source_language, encoding='utf-8').readlines()
            sentences_target_language = open(input_path + self.target_language, encoding='utf-8').readlines()

            assert len(sentences_source_language) == len(sentences_target_language)

            for i in range(len(sentences_source_language)):
                out.write(
                    sentences_source_language[i].strip() + ' ||| ' + sentences_target_language[i].strip() + '\n')
            out.close()

        if not os.path.isfile(aligned_corpus_file) or not self.use_existing:
            if self.verbose:
                print('- build aligned {} corpus'.format(self.domain))
            create_aligned_corpus(self.cleaned_corpus_file, aligned_corpus_file)
        else:
            if self.verbose:
                print('- use existing aligned {} corpus'.format(self.domain))

        fast_align = os.path.join(self.aligner_path, 'build', 'fast_align')
        atools_path = os.path.join(self.aligner_path, 'build', 'atools')

        if not os.path.isfile('{}.forward.align'.format(aligned_corpus_file)) or not self.use_existing:
            if self.verbose:
                print('- build forward alignment for {}'.format(self.domain))

            forward_align_command = '{fast_align} -i {input} -d -o -v'.format(
                fast_align=fast_align, input=aligned_corpus_file)
            forward_align_output = open('{}.forward.align'.format(aligned_corpus_file), 'w',
                                        encoding='utf-8')
            subprocess.call(forward_align_command.split(), stdout=forward_align_output)
            forward_align_output.close()
        else:
            if self.verbose:
                print('- use existing forward alignment for {}'.format(self.domain))

        if not os.path.isfile('{}.reverse.align'.format(aligned_corpus_file)) or not self.use_existing:
            if self.verbose:
                print('- build reverse alignment for {}'.format(self.domain))

            reverse_align_command = '{fast_align} -i {input} -d -o -v -r'.format(
                fast_align=fast_align, input=aligned_corpus_file)
            reverse_align_output = open('{}.reverse.align'.format(aligned_corpus_file), 'w', encoding='utf-8')
            subprocess.call(reverse_align_command.split(), stdout=reverse_align_output)
            reverse_align_output.close()

        else:
            if self.verbose:
                print('- use existing reverse alignment for {}'.format(self.domain))

        if os.path.isfile('{}.forward.align'.format(aligned_corpus_file)) \
                and os.path.isfile('{}.reverse.align'.format(aligned_corpus_file)):

            if not os.path.isfile('{}.symm.align'.format(aligned_corpus_file)) or not self.use_existing:
                if self.verbose:
                    print('- build symmetric alignment for {}'.format(self.domain))

                symmetrize_command = '{atools} -i {input}.forward.align ' \
                                     '-j {input}.reverse.align ' \
                                     '-c grow-diag-final-and'.format(atools=atools_path,
                                                                     input=aligned_corpus_file)

                symmetrize_output = open('{}.symm.align'.format(aligned_corpus_file), 'w', encoding='utf-8')
                subprocess.call(symmetrize_command.split(), stdout=symmetrize_output)
                symmetrize_output.close()

            else:
                if self.verbose:
                    print('- use existing symmetric alignment for {}'.format(self.domain))

            self.alignment_file = '{}.symm.align'.format(aligned_corpus_file)

        else:
            print('- ERROR: input files for symmetric alignment of {} were not created correctly!'.format(self.domain))
            sys.exit(1)

    def check_phrase_table(self):
        if self.verbose:
            print('\nCHECK phrase table...')
        if self.check_file(self.phrase_table_file):
            if self.verbose:
                print('- phrase table file was provided')
                self.read_phrase_table()
        else:
            self.build_phrase_table()

    def build_phrase_table(self):

        self.phrase_table_path = os.path.join(self.big_files_path, 'phrase_table')
        self.check_dir(self.phrase_table_path)

        moses_output_path = os.path.join(self.phrase_table_path, self.domain, 'model')
        self.check_dir(moses_output_path)

        phrase_table_file = os.path.join(moses_output_path, 'extract.psd.gz')
        self.phrase_table_file = phrase_table_file

        if os.path.isfile(phrase_table_file) and self.use_existing:
            if self.verbose:
                print('- use existing phrase table for {}'.format(self.domain))

        else:
            # print('- build phrase table for {}'.format(domain))
            # define needed parameters

            filename = self.corpus_file_name[:-1] if self.corpus_file_name.endswith('.') else self.corpus_file_name
            corpus = os.path.join(self.corpus_path, self.domain, filename)

            script_path = os.path.join(self.moses_path, 'scripts', 'training')
            training_script_path = os.path.join(script_path, 'train-model.perl')
            aligner_path = os.path.join(self.aligner_path, 'build')
            working_dir = os.path.join(self.phrase_table_path, self.domain)

            if self.verbose:
                print('- check for fake language model')
            fake_model_path = os.path.join(moses_output_path, 'fake.lm')
            if not os.path.exists(fake_model_path):
                out = open(fake_model_path, 'w')
                out.close()

            self.get_alignment()

            if not os.path.isfile(os.path.join(moses_output_path, 'aligned.grow-diag-final-and')):
                shutil.copy2(self.alignment_file,
                             os.path.join(moses_output_path, 'aligned.grow-diag-final-and'))
                # print('- put alignment file in correct folder for Moses')

            # run Moses to create phrase table
            if self.verbose:
                print('- run Moses to create table for {}'.format(self.domain))
            training_command = '{SCRIPT} -cores 16 -parallel -root-dir {WORKINGDIR} ' \
                               '-external-bin-dir {BINDIR} -corpus {INPUT} -f {F} -e {E} ' \
                               '-alignment grow-diag-final-and -reordering msd-bidirectional-fe  ' \
                               '-lm 0:5:{LM} -parallel -first-step 4 -last-step 6 ' \
                               '-max-phrase-length {MAX_PHRASE_LENGTH}'.format(SCRIPT=training_script_path,
                                                                               WORKINGDIR=working_dir,
                                                                               BINDIR=aligner_path,
                                                                               INPUT=self.cleaned_corpus_file,
                                                                               F=self.source_language,
                                                                               E=self.target_language,
                                                                               LM=fake_model_path,
                                                                               MAX_PHRASE_LENGTH=self.max_length)

            subprocess.call(training_command.split(), stdout=None)

            # check if phrase table was created
            if os.path.isfile(self.phrase_table_file):
                if self.verbose:
                    print('- created phrase table successfully!')
            else:
                print('- ERROR: something went wrong... process will be stopped!')
                sys.exit(1)

    def extract_word_features(self, output_path, suffix='_word'):

        if self.verbose:
            print('\nEXTRACT word psd features...')

        if self.check_file(output_path + '_global' + suffix + '.pkl') and self.check_file(
                                        output_path + '_local' + suffix + '.pkl'):
            if self.verbose:
                print('- use existing psd features')
            # global_features = self.load_pickle_obj(output_path + '_global' + suffix)
            # local_features = self.load_pickle_obj(output_path + '_local' + suffix)
            global_features = output_path + '_global' + suffix
            local_features = output_path + '_local' + suffix
        else:
            if self.verbose:
                print('- compute psd features')
            first_line = True

            phrase_table_path = os.path.join(self.big_files_path, 'phrase_table.psd.frtypes')

            tagged_corpus = self.yield_pos_tags(self.psd_tagged_corpus_file)

            global_features = {'X': [], 'y': []}
            # prior_global_features = {'X': [], 'y': []}
            local_features = {}

            sent_id = 0

            word_mapping = dict()
            row_nb = 0

            truth_labels = {0: '<unk>'}

            phrase_table = self.load_pickle_obj(phrase_table_path)

            with open('.'.join([self.cleaned_corpus_file, self.language]), encoding='utf-8') as corpus:
                for sentence in corpus:
                    sentence = sentence.strip().split()
                    tagged_sentence = next(tagged_corpus)

                    # assert len(sentence) == len(tagged_sentence)
                    if len(sentence) != len(tagged_sentence):
                        print('ERROR')
                        print(sentence)
                        print([elem['word'] for elem in tagged_sentence])
                        raise ValueError

                    sent_id += 1

                    if sent_id not in phrase_table:
                        continue

                    for start_id in phrase_table[sent_id]:

                        source_phrase, target_phrase = phrase_table[sent_id][start_id]

                        if target_phrase not in truth_labels:
                            truth_labels[target_phrase] = len(truth_labels)

                        # assert source_phrase == sentence[start_id]
                        # assert source_phrase == tagged_sentence[start_id]['word']
                        if source_phrase != sentence[start_id] or source_phrase != tagged_sentence[start_id]['word']:
                            print('ERROR')
                            print(source_phrase)
                            print(sentence[start_id])
                            print(tagged_sentence[start_id]['word'])
                            raise ValueError

                        word_features = self.get_word_features(start_id, source_phrase, sentence, tagged_sentence,
                                                               mode='train')

                        global_features['X'].append(word_features)
                        global_features['y'].append(truth_labels[target_phrase])

                        if source_phrase not in local_features:
                            local_features[source_phrase] = {'X': [], 'y': []}

                        local_features[source_phrase]['X'].append(word_features)
                        local_features[source_phrase]['y'].append(truth_labels[target_phrase])

                        word_mapping[row_nb] = source_phrase

                        # if first_line:
                        #     content = ','.join(
                        #         [feat for feat, val in sorted(word_features.items())]) + '\ttruth_label\n'
                        #     self.write_to_file(output_path + '.txt', content)
                        #     first_line = False
                        #
                        # self.write_to_file(output_path + '.txt', ','.join([str(val) for feat, val in sorted(
                        #     word_features.items())]) + '\t' + target_phrase + '\n')

                # self.save_pickle_obj(prior_global_features, output_path + '_global_prior')
                # print('- save prior global psd features')

                self.save_pickle_obj(global_features, output_path + '_global' + suffix)
                if self.verbose:
                    print('- save global psd features')

                self.save_pickle_obj(local_features, output_path + '_local' + suffix)
                if self.verbose:
                    print('- save local psd features')

                self.save_pickle_obj(word_mapping, output_path + '_global' + suffix + '.row_word_mapping')

                self.save_pickle_obj(truth_labels, output_path + '_global' + suffix + '.truth_labels')

        return global_features, local_features

    def write_vw_features(self, features, filename='global_features.train', feature_type='global', namespace=True):
        # print('- postprocess word features (write to file)')

        X = features['X']
        y = np.array(features['y']) + 1

        del features
        gc.collect()

        if feature_type == 'local':
            mapping = {0: -1}
            transformed_y = []
            for val in y:
                if val not in mapping:
                    mapping[val] = len(mapping)
                transformed_y.append(mapping[val])

            assert len(transformed_y) == len(y)

            transformed_y = np.array(transformed_y)
            self.save_pickle_obj(mapping, filename + '.mapping')
        else:
            transformed_y = y

        # print('- number of different truth labels: {}'.format(len(set(transformed_y))))

        del y
        gc.collect()

        assert len(transformed_y) == len(X)

        # print('- write features to file: {}'.format(filename))
        with open(filename, mode='w', encoding='utf-8') as out:
            for i in range(len(X)):
                assert isinstance(X[i], dict)

                feature_line = '{0} |{1}'.format(transformed_y[i],
                                                 self.get_vw_feature_line(X[i], namespace=namespace))

                out.write(feature_line)
                out.write('\n')

    def train_local_classifier(self, features, namespace=None, suffix="", holdout=1):
        solver = 'vw'

        name_mapping = dict()
        counter = 0
        folder = os.path.join(self.big_files_path, 'local{suffix}'.format(suffix=suffix))
        self.check_dir(folder)

        if isinstance(features, str):
            feature_path = features
            if self.verbose:
                print('- load local features from {}.pkl'.format(feature_path))
            features = self.load_pickle_obj(features)
            self.check_memory('loading local features')

        files = [fname for fname in os.listdir(folder) if (fname.endswith('.train') or
                                                           fname.endswith('.train.mapping.pkl'))]

        # local feature + mapping
        if len(files) == (len(features) * 2):
            print('- use existing local feature files')

        else:
            if self.verbose:
                print('- create feature files for {} source phrases'.format(len(features)))

            for source_phrase in features:
                counter += 1

                name_mapping[source_phrase] = counter
                if counter % 100 == 0:
                    if self.verbose:
                        print('- processed {} source phrases'.format(counter))

                output_path = os.path.join(folder, 'local_features_{}.train'.format(counter))
                self.write_vw_features(features[source_phrase],
                                       filename=output_path,
                                       feature_type='local',
                                       namespace=namespace)

                self.save_pickle_obj(name_mapping,
                                     os.path.join(self.big_files_path,
                                                  solver + '.local{suffix}.clfname_nb_mapping'.format(
                                                      suffix=suffix)))

            files = [fname for fname in os.listdir(folder) if (fname.endswith('.train') or
                                                               fname.endswith('.train.mapping.pkl'))]

            # local feature + mapping
            assert len(files) == (len(features) * 2)

        del features
        gc.collect()
        self.check_memory('finishing feature files')

        self.train_local_vw(folder, holdout=holdout, suffix=suffix)

        return

    def train_local_vw(self, data_path, holdout=1, suffix=''):
        if self.verbose:
            print('- start training local psd classifier')

        command = 'sh {scripts}/psd_local_vw.sh {vw} {dir} {holdout}'.format(scripts=self.script_path,
                                                                             vw=self.vowpal_wabbit_path,
                                                                             dir=data_path,
                                                                             holdout=holdout)

        log_file = open(os.path.join(self.big_files_path, 'train_local_vw{suffix}.log'.format(suffix=suffix)),
                        mode='w',
                        encoding='utf-8')
        subprocess.call(command.split(), stdout=log_file, stderr=log_file)
        log_file.close()

        return

    def train_global_vw(self, training_path, model_path):

        if self.verbose:
            print('- start training global psd classifier')

        assert self.check_file(training_path)
        assert self.check_file(model_path)

        command = 'sh {scripts}/psd_global_vw.sh {vw} {dir} {train} {model} {passes}'.format(
            scripts=self.script_path,
            vw=self.vowpal_wabbit_path,
            dir=self.big_files_path,
            train=training_path,
            model=model_path,
            passes=20)

        log_file = open(os.path.join(self.big_files_path, 'train_global_vw_{}.log'.format(int(time.time()))), 'w',
                        encoding='utf-8')
        subprocess.call(command.split(), stdout=log_file, stderr=log_file)
        log_file.close()

    def train_test_global_vw(self, name, training_path, model_path, patience=3):
        max_pass = 20
        best_acc = 0
        nb_epochs = 1
        results = []
        should_stop = False

        log_file_name = os.path.join(self.big_files_path, 'train_test_global_vw_{}.log'.format(name))

        log_file = open(log_file_name, 'a', encoding='utf-8')
        log_file.write('START training global classifier ({})\n'.format(datetime.datetime.now()))
        log_file.write('\nPASS {}\n'.format(nb_epochs))
        log_file.close()

        model_output_path = '{0}_{1}.vw'.format(model_path[:-3], nb_epochs)

        if self.verbose:
            print('- start training global psd classifier')
            print('\t- pass {0}/{1} ({2})'.format(nb_epochs, max_pass, model_output_path))

        # first run:

        command = 'sh {scripts}/psd_global_vw.sh {vw} {dir} {train} {model} {passes}'.format(
            scripts=self.script_path,
            vw=self.vowpal_wabbit_path,
            dir=self.big_files_path,
            train=training_path,
            model=model_output_path,
            passes=1)

        if self.verbose:
            print('\t\t- start training')

        log_file = open(log_file_name, 'a', encoding='utf-8')
        subprocess.call(command.split(), stdout=log_file, stderr=log_file)
        log_file.close()

        # create smaller test set to evaluate performance of training
        test_path = training_path.replace('train', 'test')
        out = open(test_path, mode='w', encoding='utf-8')
        line_counter = 0
        with open('{}.shuf'.format(training_path), encoding='utf-8') as handle:
            for line in handle:
                line_counter += 1
                out.write(line)

                if line_counter == 1000:
                    break
        out.close()

        if self.verbose:
            print('\t\t- start testing')

        acc = self.test_vw_model(test_path, model_output_path)
        results.append(acc)

        log_file = open(log_file_name, 'a', encoding='utf-8')
        log_file.write('\nRESULT: Model {0} achieved accuracy of {1}\n'.format(model_output_path, acc))
        log_file.close()

        if self.verbose:
            print('\t\t- achieved acc: {}\n'.format(acc))

        while not should_stop and nb_epochs < max_pass:

            model_input_path = '{0}_{1}.vw'.format(model_path[:-3], nb_epochs)
            nb_epochs += 1
            model_output_path = '{0}_{1}.vw'.format(model_path[:-3], nb_epochs)

            log_file = open(log_file_name, 'a', encoding='utf-8')
            log_file.write('\nPASS {}\n\n'.format(nb_epochs))
            log_file.close()

            if self.verbose:
                print('\t- pass {0}/{1} ({2})'.format(nb_epochs, max_pass, model_output_path))

            command = 'sh {scripts}/psd_global_vw_retrain.sh {vw} {dir} {train} {model_input} {model_output} {passes}'.format(
                scripts=self.script_path,
                vw=self.vowpal_wabbit_path,
                dir=self.big_files_path,
                train=training_path,
                model_input=model_input_path,
                model_output=model_output_path,
                passes=1)

            if self.verbose:
                print('\t\t- start training')

            log_file = open(log_file_name, 'a', encoding='utf-8')
            subprocess.call(command.split(), stdout=log_file, stderr=log_file)
            log_file.close()

            if self.verbose:
                print('\t\t- start testing')

            # acc = self.test_vw_model('{}.shuf'.format(training_path), model_output_path)
            acc = self.test_vw_model_stepwise(name, test_path, model_output_path)
            results.append(acc)

            log_file = open(log_file_name, 'a', encoding='utf-8')
            log_file.write('\nRESULT: Model {0} achieved accuracy of {1}\n'.format(model_output_path, acc))
            log_file.close()

            if self.verbose:
                print('\t\t- achieved acc: {}\n'.format(acc))

            if (acc > best_acc):
                stopping_step = 0
                best_acc = acc
                best_acc_model = model_output_path
            else:
                stopping_step += 1

            if stopping_step >= patience or acc == 1.0:
                should_stop = True

        log_file = open(log_file_name, 'a', encoding='utf-8')
        log_file.write('\nSTOPPED training of global model after {0} passes ' \
                       'with best accuracy of {1} (model: {2})\n'.format(nb_epochs,
                                                                         best_acc,
                                                                         best_acc_model))
        log_file.close()
        if self.verbose:
            print("- stopped training of global model after {0} passes" \
                  " with best accuracy of {1} (model: {2})".format(nb_epochs,
                                                                   best_acc,
                                                                   best_acc_model))
        return best_acc_model

    def test_vw_model(self, test_path, model_path):

        assert self.check_file(model_path)

        command = 'sh {0}/test_vw_model.sh {1} {2} {3} {4} {5}'.format(self.script_path,
                                                                       self.vowpal_wabbit_path,
                                                                       model_path,
                                                                       test_path,
                                                                       '--probabilities',
                                                                       '_')

        out = subprocess.Popen(command.split(), stdout=subprocess.PIPE).communicate()[0]
        out = [test_example.split() for test_example in str(out, 'utf-8').split('\n') if test_example]
        assert len(out) > 0

        probs = []
        for test_example in out:
            example_probs = sorted([(int(label_prob.split(':')[0]), float(label_prob.split(':')[1]))
                                    for label_prob in test_example])
            example_probs = [prob for label, prob in example_probs]
            probs.append(example_probs)

        predictions = np.argmax(probs, axis=1) + 1

        gold_labels = [int(line.split('|')[0]) for line in open(test_path, encoding='utf-8')]

        acc = accuracy_score(gold_labels, predictions)

        return acc

    def test_vw_model_stepwise(self, name, test_path, model_path):

        assert self.check_file(model_path)

        gold_labels = [int(line.split('|')[0]) for line in open(test_path, encoding='utf-8')]
        predictions = []
        with open(test_path, encoding='utf-8') as handle:
            for line in handle:
                temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}_temp.txt'.format(name))

                with open(temp_path, mode='w', encoding='utf-8') as out:
                    out.write(line + '\n')

                command = 'sh {0}/test_vw_model.sh {1} {2} {3} {4} {5}'.format(self.script_path,
                                                                               self.vowpal_wabbit_path,
                                                                               model_path,
                                                                               temp_path,
                                                                               '--probabilities',
                                                                               '_')

                probs = subprocess.Popen(command.split(), stdout=subprocess.PIPE).communicate()[0]
                probs = str(probs, 'utf-8').split()

                probs = np.array([float(prob.split(':')[1]) for prob in probs])
                # print(probs)
                pred = np.argmax(probs) + 1
                predictions.append(pred)

        acc = accuracy_score(gold_labels, predictions)
        return acc

    def run(self):

        self.preprocess_for_feature_extraction()
        self.check_memory(state='finished preprocessing for psd classifier')
        self.read_phrase_table()

        output_path = os.path.join(self.big_files_path, 'psd_features')

        global_features, local_features = self.extract_word_features(output_path)

        # self.extract_features_global_multiple(output_path)
        self.check_memory(state='finished extracting features for psd classifier')

        #### GLOBAL PSD MODEL ####
        if self.verbose:
            print('\nGLOBAL psd classifer')

        name = 'global'  # 'global.namespace'
        global_model_folder = os.path.join(self.big_files_path, 'global_models')
        self.check_dir(global_model_folder)
        global_model_path = os.path.join(global_model_folder, '{}.vw'.format(name))
        path_to_global_features = os.path.join(self.big_files_path, '{}.train'.format(name))
        if not self.check_file(path_to_global_features):

            if self.verbose:
                print('- write global psd features')

            if isinstance(global_features, str):
                global_features = self.load_pickle_obj(global_features)
                self.check_memory(state='finished loading features for psd classifier')

            if 'namespace' in name:
                self.write_vw_features(global_features, filename=path_to_global_features, feature_type='global',
                                       namespace='FEAT')
            else:
                self.write_vw_features(global_features, filename=path_to_global_features, feature_type='global',
                                       namespace=False)

        else:
            if self.verbose:
                print('- use existing global feature file')

        # self.train_global_vw(path_to_global_features, global_model_path)
        self.train_test_global_vw(name, path_to_global_features, global_model_path, patience=5)

        #### LOCAL ####
        if self.verbose:
            print('\nLOCAL psd classifer')

        self.train_local_classifier(local_features, namespace=None, suffix=".no_namespace.holdout",
                                    holdout=1)

        return

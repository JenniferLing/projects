import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Input, Convolution1D, MaxPooling1D, Reshape, GRU, Bidirectional, \
    Dropout, Flatten, Concatenate
import os
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
import time
import pickle

BIG_FILES_PATH = '/mounts/work/lingj/sensespotting'
PSD_PATH = '/mounts/Users/student/lingj/sensespotting/orig_data/EMEA.psd'
CORPUS_PATH = '/mounts/work/lingj/sensespotting/corpus/EMEA/train.lowercased.fr'


def do_cross_validation(path, repeat=10, k=16):
    epochs = 20
    batch_size = 100
    window_size = 5

    feature_scores = {'acc': [], 'roc': [],
                      'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                      'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

    # prefix = 'tocc_tocp_ton_topg_topgb_topl_toplb_topr_tyc_tyn_tyr_tyt.lowercased'  # all_features, flen=64
    # nb_features = 64
    # name = 'all_features'

    # prefix = 'tocc_tocp_ton_topg_topgb_topl_toplb_topr.lowercased' # token_only, flen=35
    # nb_features  = 35
    # name = 'token_only'

    prefix = 'tyc_tyn_tyr_tyt.lowercased' # type_only, flen=29
    nb_features = 29
    name = 'type_only'

    global_embedding_file = os.path.join(BIG_FILES_PATH, 'embeddings', 'hansards-word2vec_cbow-dim_128-epochs_100-lrate_0.05-window_10')
    psd_path = PSD_PATH

    for r in range(repeat):
        for i in range(k):
            print('\nRepetition {0}, Fold {1}'.format(r + 1, i + 1))
            sensespotting_file = os.path.join(path, '{0}.{1}_{2}_fold.run_{3}'.format(prefix, i, k, r))

            contexts = get_contexts(CORPUS_PATH, psd_path, window_size)
            word_dict = create_word_dict(sensespotting_file, global_embedding_file, contexts)

            words_train, X_train, context_X_train, y_train = load_data('{}.train'.format(sensespotting_file),
                                                                       nb_features, contexts, word_dict, window_size)
            if isinstance(X_train, list):
                print('Any error in loading data; adapt flen!')
                exit()

            words_dev, X_dev, context_X_dev, y_dev = load_data('{}.dev'.format(sensespotting_file), nb_features,
                                                               contexts, word_dict, window_size)
            words_test, X_test, context_X_test, y_test = load_data('{}.test'.format(sensespotting_file), nb_features,
                                                                   contexts,
                                                                   word_dict, window_size)


            model = build_model(nb_features, global_embedding_file, window_size)

            model.fit([context_X_train, X_train], y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=([context_X_dev, X_dev], y_dev), verbose=2)

            X = np.concatenate((X_dev, X_test))
            y = np.concatenate((y_dev, y_test))

            context_X = np.concatenate((context_X_dev, context_X_test))
            words = np.concatenate((words_dev, words_test))

            y_pred = model.predict([context_X, X])
            y_pred = [int(round(y[0])) for y in y_pred]

            feature_scores = compute_performance(feature_scores, y, y_pred, words)

        del model

    print('\nPERFORMANCE averages ')
    metrics = ['roc', 'prec_macro', 'rec_macro', 'f1_macro', 'prec_micro', 'rec_micro', 'f1_micro']
    save_pickle_obj(feature_scores, 'context_nn_features_scores_{0}'.format(name)) #str(time.time()).split('.')[0]))
    for metric in metrics:
        assert len(feature_scores[metric]) == (k * repeat)
        print('- {0} = {1} (+/- {2})'.format(metric,
                                             round(np.mean(feature_scores[metric]), 2),
                                             round(np.std(feature_scores[metric]), 2)))

    return

def build_model(nb_features, emb_path, window_size, verbose = False):
    context_rnn_units = [32, 16, 8]
    context_layer = 'biLSTM'
    context_dropout = 0.1
    context_dense_units = [128, 64, 32, 16]
    optimizer = 'adam'

    input_shape = {'context': (window_size * 2,), 'sense_indicators': (nb_features,)}

    context_input = Input(shape=input_shape['context'], name='context_input')


    print('Initialize with pretrained embeddding weights')
    pretrained_embeddings = np.load(emb_path + '-adapted.npy')

    embedding_layer = Embedding(input_dim=pretrained_embeddings.shape[0],
                                output_dim=pretrained_embeddings.shape[1],
                                weights=[pretrained_embeddings],
                                input_length=None,
                                trainable=True,
                                mask_zero=False,
                                name='word_embedding')


    hidden = embedding_layer(context_input)

    if verbose:
        print('Embedding layer output shape: {}'.format(embedding_layer.output_shape))

    type_to_func = {'LSTM': LSTM, 'GRU': GRU, 'biLSTM': LSTM, 'biGRU': GRU, 'BiLSTM': LSTM, 'BiGRU': GRU}

    for i in range(len(context_rnn_units)):
        units = context_rnn_units[i]
        if 'bi' in context_layer:
            rnn_layer = Bidirectional(type_to_func[context_layer](units, return_sequences=True))
        else:
            rnn_layer = type_to_func[context_layer](units, return_sequences=True)
        hidden = rnn_layer(hidden)

        if context_dropout:
            hidden = Dropout(context_dropout)(hidden)

        if verbose:
            print('RNN layer output shape: {}'.format(rnn_layer.output_shape))

    flatten_layer = Flatten()
    hidden = flatten_layer(hidden)
    if verbose:
        print('Flatten layer output shape: {}'.format(flatten_layer.output_shape))

    sense_indicator_input = Input(shape=input_shape['sense_indicators'], name='sense_indicator')

    combination_layer = Concatenate(axis=-1, name='combine_context_and_sense_indicators')
    hidden = combination_layer([hidden, sense_indicator_input])
    if verbose:
        print('Combined knowledge output shape: {}'.format(combination_layer.output_shape))

    activations = ['relu', 'tanh', 'relu', 'tanh', None, 'tanh']
    for i in range(len(context_dense_units)):
        units = context_dense_units[i]
        dense_layer = Dense(units=units, activation=activations[i])
        hidden = dense_layer(hidden)
        if verbose:
            print('Dense layer output shape: {}'.format(dense_layer.output_shape))

            if context_dropout:
                hidden = Dropout(context_dropout)(hidden)

    dense_layer = Dense(1,
                        activation='sigmoid',
                        name='context_output',
                        # kernel_regularizer=l2(0.001),
                        # activity_regularizer=l1(0.01),
                        )

    main_output = dense_layer(hidden)

    model = Model(inputs=[context_input, sense_indicator_input], outputs=main_output)

    if verbose:
        print(model.summary())

    model.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return model

def load_data(path, flen, contexts, word_dict, window_size):

    X = []
    Y = []
    context_X = []
    words = []
    first_row = True
    gold_labels = {-1: 0, 1: 1}
    counter = 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            counter += 1

            line = line.strip().split('|')

            label, word = line[0].split()
            word, lineID = word.split('-')
            y = gold_labels[int(label)]

            feature_vals = []
            for feature_type in line[1:]:
                if 'bias' in feature_type.lower():
                    continue

                feature_vals.extend([float(feature.split(':')[1]) for feature in feature_type.split()
                                     if len(feature.split(':')) > 1])

            if len(feature_vals) != flen:
                continue

            left_context, right_context = contexts[(int(lineID), word)]
            context = left_context + right_context

            context_features = []
            for _ in range(window_size - len(left_context)):
                context_features.append(word_dict['<PAD>'])

            for context_word in context:
                context_features.append(word_dict.get(context_word, 1))

            for _ in range(window_size - len(right_context)):
                context_features.append(word_dict['<PAD>'])

            assert len(context_features) == window_size * 2

            if first_row:
                X.append(feature_vals)
                context_X.append(context_features)
                first_row = False
            else:

                X = np.concatenate([X, [feature_vals]], axis=0)
                context_X = np.concatenate([context_X, [context_features]], axis=0)

            Y.append(y)
            words.append(word)

    return np.array(words), X, context_X, np.array(Y)

def load_pickle_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def read_psd_file(psd_path):

    last_sent_id = -1
    data = {}
    source_words = []
    line_nb = 0
    with open(psd_path, encoding='utf-8') as handle:
        for line in handle:
            sent_id, source_language_token_start, source_language_token_end, \
            target_language_token_start, target_language_token_end, \
            source_language_phrase, target_language_phrase = line.strip().split('\t')

            sent_id = int(sent_id)
            source_language_token_start = int(source_language_token_start)
            source_language_token_end = int(source_language_token_end)

            if sent_id not in data:
                data[sent_id] = {}

            data[sent_id][(line_nb, source_language_token_start, source_language_token_end)] = (
                source_language_phrase, target_language_phrase)

            last_sent_id = sent_id
            source_words.append(source_language_phrase)
            line_nb += 1

    return data, last_sent_id, source_words

def get_contexts(corpus_path, psd_path, window_size):
    contexts = {}
    data, last_sent_id, _ = read_psd_file(psd_path)

    sent_id = 0
    with open(corpus_path, encoding='utf-8') as handle:
        for line in handle:
            sent_id += 1
            if sent_id not in data:
                continue

            line = line.strip().split()
            for line_nb, start, end in sorted(data[sent_id]):
                word, _ = data[sent_id][(line_nb, start, end)]
                word_in_line = line[start] if start == end else ' '.join(line[start:end+1])

                assert word == word_in_line

                left_context = line[max(start-window_size, 0): start]
                right_context = line[end+1: min(end+1+window_size, len(line))]

                contexts[(line_nb, word)] = (left_context, right_context)
                assert len(left_context) <= window_size and len(right_context) <= window_size

    return contexts

def create_word_dict(sensespotting_feature_file, global_embedding_file, contexts):
    word_dict = {'<PAD>': 0, "<OOV>": 1}

    global_embedding_dct = load_pickle_obj(global_embedding_file)
    global_embedding_vectors = np.load(global_embedding_file + ".npy")

    padding_vector = np.zeros((1, global_embedding_vectors.shape[1]))
    unknown_words_vector = np.ones((1, global_embedding_vectors.shape[1]))

    pretrained_embeddings = np.concatenate((padding_vector, unknown_words_vector),
                                               axis=0)

    with open('{0}.train'.format(sensespotting_feature_file), encoding='utf-8') as handle:
        for line in handle:
            line = line.strip().split('|')

            label, word = line[0].split()
            word, lineID = word.split('-')

            # skip phrases
            if '_' in word:
                continue

            left_context, right_context = contexts[(int(lineID), word)]
            context = left_context + right_context
            
            for context_word in context:

                if context_word not in word_dict and context_word in global_embedding_dct:

                    word_dict[context_word] = len(word_dict)

                    globalID = global_embedding_dct[context_word]

                    pretrained_word_vector = global_embedding_vectors[globalID]

                    pretrained_word_vector = pretrained_word_vector.reshape(1, -1)

                    pretrained_embeddings = np.concatenate((pretrained_embeddings, pretrained_word_vector),
                                                           axis=0)
                    assert len(word_dict) == pretrained_embeddings.shape[0]

    assert len(word_dict) == pretrained_embeddings.shape[0]
    adapted_embeddings_file = global_embedding_file + '-adapted'
    np.save(adapted_embeddings_file, pretrained_embeddings)

    return word_dict

def compute_performance(scores, gold_labels, predictions, source_words, weighted_micro=False, verbose=False):
    acc = accuracy_score(gold_labels, predictions) * 100

    roc = roc_auc_score(gold_labels, predictions) * 100
    prec_macro, rec_macro, f1_macro = get_confusion_matrix('<unk>', gold_labels, predictions)
    prec_micro, rec_micro, f1_micro = compute_micro_performance(gold_labels, predictions, source_words,
                                                                     weighted=weighted_micro)


    print("acc = {0}, prec_macro = {1}, rec_macro = {2}, f1_macro = {3}, f1_micro = {4}, "
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


def compute_micro_performance(gold_labels, predictions, source_words, weighted=False):
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

        prec, rec, f1 = get_confusion_matrix(word, gold, pred)

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

def get_confusion_matrix(word, gold, pred):
    assert len(pred) == len(gold)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(pred)):
        if pred[i] == 1 and gold[i] == 1:
            tp += 1
        elif pred[i] == 1 and gold[i] == 0:
            fp += 1
        elif pred[i] == 0 and gold[i] == 1:
            fn += 1
        elif pred[i] == 0 and gold[i] == 0:
            tn += 1
        else:
            raise ValueError('Prediction not binary!')

    if tp == 0:
        return 0, 0, 0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    if prec == 0 or rec == 0:
        f1 = 0
    else:
        f1 = (2 * prec * rec) / (prec + rec)

    return prec * 100, rec * 100, f1 * 100

def save_pickle_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    data_path = os.path.join(BIG_FILES_PATH, 'neural_net', 'data')

    do_cross_validation(data_path)
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Input, merge, Convolution1D, MaxPooling1D, Reshape
import os
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
import time
import pickle

BIG_FILES_PATH = '/mounts/work/lingj/sensespotting'

def do_cross_validation(path, repeat=10, k=16):
    dense_layers = [100, 90, 80, 75, 50, 25]  # [128, 64, 32, 16]
    epochs = 20
    batch_size = 100

    feature_scores = {'acc': [], 'roc': [],
                      'prec_macro': [], 'rec_macro': [], 'f1_macro': [],
                      'prec_micro': [], 'rec_micro': [], 'f1_micro': []}

    # prefix = 'tocc_tocp_ton_topg_topgb_topl_toplb_topr_tyc_tyn_tyr_tyt.lowercased'  # all_features, flen=64
    # nb_features = 64

    # prefix = 'tocc_tocp_ton_topg_topgb_topl_toplb_topr.lowercased' # token_only, flen=35
    # nb_features  = 35

    prefix = 'tyc_tyn_tyr_tyt.lowercased' # type_only, flen=29
    nb_features = 29

    for r in range(repeat):
        for i in range(k):
            print('\nRepetition {0}, Fold {1}'.format(r + 1, i + 1))
            words_train, X_train, y_train = load_data(os.path.join(path, '{0}.{1}_{2}_fold.run_{3}.train'.format(prefix, i, k, r)),
                                         nb_features)
            if isinstance(X_train, list):
                print('Any error in loading data; adapt flen!')
                exit()

            words_dev, X_dev, y_dev = load_data(os.path.join(path, '{0}.{1}_{2}_fold.run_{3}.dev'.format(prefix, i, k, r)), nb_features)
            words_test, X_test, y_test = load_data(os.path.join(path, '{0}.{1}_{2}_fold.run_{3}.test'.format(prefix, i, k, r)), nb_features)

            model = Sequential()
            model.add(Dense(units=dense_layers[0], input_shape=(X_train.shape[1],), activation='tanh'))
            for j in range(1, len(dense_layers)):
                model.add(Dense(units=dense_layers[j], activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev), verbose=2)

            X = np.concatenate((X_dev, X_test))
            y = np.concatenate((y_dev, y_test))
            words = np.concatenate((words_dev, words_test))

            y_pred = model.predict(X)
            y_pred = [int(round(y[0])) for y in y_pred]


            feature_scores = compute_performance(feature_scores, y, y_pred, words)

        del model

    print('\nPERFORMANCE averages ')
    metrics = ['roc', 'prec_macro', 'rec_macro', 'f1_macro', 'prec_micro', 'rec_micro', 'f1_micro']
    save_pickle_obj(feature_scores, 'mlp_features_scores_type_only'.format(str(time.time()).split('.')[0]))
    for metric in metrics:
        assert len(feature_scores[metric]) == (k * repeat)
        print('- {0} = {1} (+/- {2})'.format(metric,
                                             round(np.mean(feature_scores[metric]), 2),
                                             round(np.std(feature_scores[metric]), 2)))

    return


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

def load_data(path, flen=64):
    X = []
    Y = []
    words = []
    first_row = True
    gold_labels = {-1: 0, 1: 1}
    counter = 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            counter += 1

            line = line.strip().split('|')

            label, word = line[0].split()

            y = gold_labels[int(label)]

            feature_vals = []
            for feature_type in line[1:]:
                if 'bias' in feature_type.lower():
                    continue

                feature_vals.extend([float(feature.split(':')[1]) for feature in feature_type.split()
                                     if len(feature.split(':')) > 1])

            if len(feature_vals) != flen:
                continue

            if first_row:
                X.append(feature_vals)
                first_row = False
            else:
                X = np.concatenate([X, [feature_vals]], axis=0)
            Y.append(y)
            words.append(word.split('-')[0])

    return np.array(words), X, np.array(Y)

def save_pickle_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    data_path = os.path.join(BIG_FILES_PATH, 'neural_net', 'data')

    do_cross_validation(data_path)
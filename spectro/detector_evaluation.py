import os
import json
import librosa
import argparse
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from processing import load_audio, SAMPLE_SIZE, melspectrogram, SR
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, plot_confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report


TESTSET = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/old_Tobia/test.npy"

def load_data(f, s=False):
    data = np.load(f, allow_pickle=True)
    x = []
    y = []

    for sample in data:
        x.append(sample[0])
        y.append(to_categorical(sample[1], num_classes=2))

    x = np.array(x)
    y = np.array(y)

    shape = (x.shape[1], x.shape[2], 1)

    x = x.reshape(x.shape[0], shape[0], shape[1], shape[2])

    if s:
        return x, y, shape

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25)

    return x_train, y_train, x_val, y_val, shape

class Detector(object):
    def __init__(self, m="model.json", w="model.h5", extra=None):
        self.m = m
        self.w = w
        self.x_extra = extra[0]
        self.y_extra = extra[1]

        json_file = open(self.m, 'r')
        self.model = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(self.model)
        self.model.load_weights(self.w)
#        os.system("clear")
        self.model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=['accuracy']
        )

    def test(self, x_test, y_test, extra=True):

        test_err, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('real_acc_ %f' % test_acc)
        #predict probabilities for test set


        yhat_probs = self.model.predict(x_test, verbose=0)
        # predict crisp classes for test set
        yhat_classes = np.argmax(self.model.predict(x_test, verbose=0),axis=-1)
        y_test = np.argmax(y_test,axis=-1)
        acc = round(accuracy_score(y_test, yhat_classes), 4) * 100
        print('Accuracy: %f' % acc)
        precision = round(precision_score(y_test, yhat_classes), 4) * 100
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = round(recall_score(y_test, yhat_classes), 4) * 100
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = round(f1_score(y_test, yhat_classes), 4) * 100
        print('F1 score: %f' % f1)
        #area under roc curve
        auroc = round(roc_auc_score(y_test, yhat_classes), 4) * 100
        print('Auroc: %f' % auroc)
        tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
        confusion_matrix_abs_nums = {"FP": fp, "FN": fn, "TN": tn, "TP": tp}

        # print (classification_report(Y_test, y_pred))

        score = {
            "Accuracy %": acc,
            "Precision %": precision,
            "Recall %": recall,
            "F1-Score %": f1,
            "AUROC %": auroc,
            "ConfusionMatrixCounts": confusion_matrix_abs_nums
        }
        return score

    def detect(self, audio, out, visual):
        self.out = out
        try:
            signal = load_audio(audio)

        except ValueError:
            if visual:
                print("Recording too short!")
            return

        if len(signal) < SAMPLE_SIZE:
            if visual:
                print("Recording too short!")
            return

        current = 0
        end = False
        predictions = []
        pieces = []

        while not end:
            if current+SAMPLE_SIZE > len(signal):
                sample = signal[len(signal)-SAMPLE_SIZE:]
                end = True
            else:
                sample = signal[current:current+SAMPLE_SIZE]
                current += SAMPLE_SIZE

            mel = melspectrogram(sample)
            x = np.array(mel)
            x = x[np.newaxis, ...]
            x = np.expand_dims(x, axis=3)
            pred = np.argmax(self.model.predict(x), axis=1)
            predictions.append(pred)
            pieces.append(sample)

        for i in range(len(predictions)):
            if predictions[i][0] == 0:
                librosa.output.write_wav(self.out + str(i) + os.path.split(audio)[1], pieces[i], sr=SR)

        if visual:
            end = '\033[0m'
            green = '\033[92m'
            length = 50
            each = "|" * (length // len(predictions))
            output = ""
            for p in predictions:
                if p[0] == 0:
                    output += green + each + end
                else:
                    output += each

            print(output)
        else:
            return pieces, predictions


if __name__ == '__main__':
    x_extra, y_extra, _ = load_data(TESTSET, s=True)
    GRAPH = "/Users/andreatamburri/Desktop/Voicemed/Detector/sweep/swift-salad-2/model.json"
    WEIGHTS = "/Users/andreatamburri/Desktop/Voicemed/Detector/sweep/swift-salad-2/model.h5"
    d = Detector(GRAPH, WEIGHTS, extra = (x_extra,y_extra))
    score = d.test(x_extra,y_extra,extra=True)
    print(score)


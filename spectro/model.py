import os
import wandb
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from wandb.keras import WandbCallback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, plot_confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, LeakyReLU, SpatialDropout2D, GlobalAveragePooling2D


DATASET = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/Andrea/dataset.npy"
TESTSET = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/Andrea/test.npy"
global model

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

class ExtraCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        score = model.test(model.x_extra, model.y_extra, extra=True)
        try:
            wandb.log({'score':score})
            print("  score of CNN" + wandb.run.name + ":"+str(score))
        except:
            pass
        #print("Evaluating over our own dataset, accuracy: "+str(acc))

class Model(object):
    def __init__(self, name, config, hyper=False, hyper_project="", extra=None):
        self.name = name
        self.x_extra = extra[0]
        self.y_extra = extra[1]

        if hyper:
            wandb.init(config=config, project=hyper_project)
            wandb.run.save("/Users/andreatamburri/Desktop/Voicemed/Detector/")
            self.config = wandb.config
            self.callback = WandbCallback(data_type="image", validation_data=extra)
            try:
                os.system("mkdir /Users/andreatamburri/Desktop/Voicemed/Detector/sweep/"+wandb.run.name)
            except:
                pass
        else:
            self.config = config
            log_dir = "logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.extra_callback = ExtraCallBack()

    def build(self, input_shape):
        self.model = keras.Sequential([
            Conv2D(self.config["conv1"], self.config["kernel1"], kernel_regularizer=l2(self.config["l2_rate"]), input_shape=input_shape),
            LeakyReLU(alpha=self.config["alpha"]),
            BatchNormalization(),

            SpatialDropout2D(self.config["drop1"]),
            Conv2D(self.config["conv2"], self.config["kernel2"], kernel_regularizer=l2(self.config["l2_rate"])),
            LeakyReLU(alpha=self.config["alpha"]),
            BatchNormalization(),

            MaxPooling2D(self.config["pool1"], padding='same'),

            SpatialDropout2D(self.config["drop1"]),
            Conv2D(self.config["conv3"], self.config["kernel3"], kernel_regularizer=l2(self.config["l2_rate"])),
            LeakyReLU(alpha=self.config["alpha"]),
            BatchNormalization(),

            SpatialDropout2D(self.config["drop2"]),
            Conv2D(self.config["conv4"], self.config["kernel4"], kernel_regularizer=l2(self.config["l2_rate"])),
            LeakyReLU(alpha=self.config["alpha"]),
            BatchNormalization(),

            GlobalAveragePooling2D(),

            Dense(2, activation='softmax')
        ])

    def train(self, x_train, y_train, validation):
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.config["lr"],
            beta_1=self.config["beta_1"],
            beta_2=self.config["beta_2"]
        )
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )

        self.model.summary()

        self.model.fit(
            x_train,
            y_train,
            validation_data=validation,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[self.callback, self.extra_callback]
        )

    def test(self, x_test, y_test, extra=True):
        #test_err, test_acc = self.model.evaluate(x_test, y_test, verbose=0)

        # predict probabilities for test set
        yhat_probs = self.model.predict(x_test, verbose=0)
        # predict crisp classes for test set
        yhat_classes = self.model.predict_classes(x_test, verbose=0)

        Y_test = np.argmax(y_test, axis=-1)
        acc = round(accuracy_score(Y_test, yhat_classes), 4) * 100
        print('Accuracy: %f' % acc)
        precision = round(precision_score(Y_test, yhat_classes), 4) * 100
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = round(recall_score(Y_test, yhat_classes), 4) * 100
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = round(f1_score(Y_test, yhat_classes), 4) * 100
        print('F1 score: %f' % f1)
        # area under roc curve
        auroc = round(roc_auc_score(Y_test, yhat_classes), 4) * 100
        print('Auroc: %f' % auroc)
        tn, fp, fn, tp = confusion_matrix(Y_test, yhat_classes).ravel()
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

        if extra:
            return score
        else:
            print("Accuracy on testing data: "+str(test_acc))

    def save(self):
        folder = "/Users/andreatamburri/Desktop/Voicemed/Detector/sweep/"+wandb.run.name+"/"
        with open(folder+"model.json", "w") as json_file:
            json_file.write(self.model.to_json())

        self.model.save_weights(folder+"model.h5")
        print("Saved model '"+self.name+"-"+wandb.run.name+"' to disk")


if __name__ == '__main__':
    should_train = True

    if should_train:
        x_train, y_train, x_val, y_val, shape = load_data(DATASET)
        x_extra, y_extra, _ = load_data(TESTSET, s=True)

        config = dict(
            conv1 = 32,
            kernel1 = (3,3),
            drop1 = 0.07,

            conv2 = 32,
            kernel2 = (3,3),

            pool1 = (2,2),

            conv3 = 64,
            kernel3 = (3,3),

            drop2 = 0.14,

            conv4 = 64,
            kernel4 = (3,3),

            batch_size = 128,
            epochs = 70,

            lr = 1e-4,
            beta_1 = 0.99,
            beta_2 = 0.999,
            l2_rate = 0.001,

            alpha = 0.1
        )

        model = Model("Spectro3", config, hyper=True, hyper_project="CoughDetectionv3", extra=(x_extra, y_extra))
        model.build(shape)
        model.train(x_train, y_train, (x_val, y_val))
        model.save()

    else:
        pass

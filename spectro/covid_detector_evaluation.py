import pickle
import joblib
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score, plot_confusion_matrix, precision_recall_curve, roc_curve, auc, classification_report
import time
import warnings

warnings.filterwarnings('ignore')

import pprint
pp = pprint.PrettyPrinter(indent=4)

DATASET = "/Users/andreatamburri/Desktop/Voicemed/Covid/Mfcc/dataset_mfcc.npy"
MODEL = "/Users/andreatamburri/Desktop/Voicemed/Covid/BestModel/svm_model.sav"
UMAP = "/Users/andreatamburri/Desktop/Voicemed/Covid/Umap/umap_reducer.sav"
FIT=False


#Splitting the data into train/test
def data_split(data_features, data_labels):
    X_train, X_test, Y_train, Y_test = train_test_split(data_features,
                                                        data_labels,
                                                        test_size=0.20,
                                                        stratify= data_labels,
                                                        random_state = 42)
    return X_train, X_test, Y_train, Y_test



#Function to transform data to umap features
def umap_transform(features, filename, labels=None, fit=False, display=False):
    if fit:
        reducer = umap.UMAP(n_neighbors=30, min_dist = 0.3, random_state=42)
        transformed_features = reducer.fit(features, y=labels)
        print("Saving UMAP reducer")
        joblib.dump(reducer, filename)
        if display:
            plt.scatter(transformed_features.embedding_[:, 0], transformed_features.embedding_[:, 1], s= 2, c=labels, cmap='Spectral')
            plt.title('UMAP Features')
            plt.savefig('UMAP Features.png')
        return transformed_features.embedding_
    else:
        reducer = joblib.load(filename)
        transformed_features = reducer.transform(features)

        if display:
            plt.scatter(transformed_features[:, 0], transformed_features[:, 1], s=2, c=labels,
                        cmap='Spectral')
            plt.title('UMAP Features')
            plt.savefig('UMAP Features.png')

        return transformed_features


#Function to scale
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def get_evaluation_scores(y_pred, Y_test):
    a = round(accuracy_score(Y_test, y_pred), 4) * 100
    p = round(precision_score(Y_test, y_pred), 4) * 100
    r = round(recall_score(Y_test, y_pred), 4) * 100
    f1 = round(f1_score(Y_test, y_pred), 4) * 100
    auroc = round(roc_auc_score(Y_test, y_pred), 4) * 100

    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    confusion_matrix_abs_nums = {"FP": fp, "FN": fn, "TN": tn, "TP": tp}

    # print (classification_report(Y_test, y_pred))

    score = {
        "Accuracy %": a,
        "Precision %": p,
        "Recall %": r,
        "F1-Score %": f1,
        "AUROC %": auroc,
        "ConfusionMatrixCounts": confusion_matrix_abs_nums
    }
    return score


def plot_ROC(Y_test, y_pred):
    false_pos_rate, true_pos_rate, thresholds = roc_curve(Y_test, y_pred)
    roc_auc = auc(false_pos_rate, true_pos_rate, )
    plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver Operating Characteristic Curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def support_vector_machine(X_train, X_test, Y_train, Y_test, load=False):
    print ("\nSupport Vector Machine:")
    start_time = time.time()
    if not load:
        print('--- Training SVM model for covid detection ---')
        svm_model = SVC(class_weight = 'balanced')
        svm_model.fit(X_train, Y_train)
        pickle.dump(svm_model, open(MODEL, 'wb'))
        print("--- training %s seconds ---" % (time.time() - start_time))
    else:
        print('--- Loading SVM model for covid detection ---')
        svm_model = pickle.load(open(MODEL, 'rb'))
        print("--- inference %s seconds ---" % (time.time() - start_time))
    y_pred = svm_model.predict(X_test)
    svm_score = get_evaluation_scores(y_pred, Y_test)
    pp.pprint(svm_score)
    return svm_score





def mfcc_preprocessing(DATASET, visualize = False, umap = False, scale = False):
    mfcc_ar = np.array(np.load(DATASET, allow_pickle=True))
    mfcc = pd.DataFrame(data=mfcc_ar, columns=['sample', 'label'])
    mfcc['label'] = mfcc['label'].replace({0: 1, 1: 0})
    num_features = len(mfcc['sample'].values[0])
    data_features = pd.DataFrame(mfcc["sample"].to_list(), columns=list(range(num_features)))
    data_labels = mfcc["label"]


    if visualize:
        print(mfcc.head())
        print ("Number of Covid Samples: ", len(mfcc[mfcc['label'] == 1]))
        print ("Number of Non-Covid Samples: ", len(mfcc[mfcc['label'] == 0]))
        print(data_features.head())

    X_train, X_test, Y_train, Y_test = data_split(data_features, data_labels)
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    if umap :
        start_time = time.time()
        print("UMAP Dimension Reduction for Train Data")
        X_train = umap_transform(features=X_train, labels=Y_train, display=visualize, filename=UMAP, fit=FIT)
        X_test = umap_transform(features=X_test, labels=Y_test, display=visualize, filename=UMAP, fit=FIT)
        print("--- %s seconds ---" % (time.time() - start_time))

    if scale :
        X_train, X_test = scale_features(X_train, X_test)

    data = (X_train, X_test, Y_train, Y_test)

    return data

if __name__ == '__main__':

    X_train,X_test,Y_train,Y_test = preprocessing(DATASET, visualize=False, umap=False, scale=True)
    support_vector_machine(X_train,X_test,Y_train,Y_test,load=True)
''' first draft to unify all preprocessing settings for cough and covid detectors'''
''' to check augmentation working only for covid labeled samples'''

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import floor
from random import shuffle
from speechpy.processing import cmvn

SR = 44000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
SILENCE = 0.0018
SAMPLE_LENGTH = 0.5 #s
SAMPLE_SIZE = int(np.ceil(SR*SAMPLE_LENGTH))
NOISE_RATIO = 0.3
MFCC = True
N_MFCC = 14

LABELS = ["covid", "cough"]
exclude_list = ['LICENSE.md', '.DS_Store', 'README.md', 'combined_data.csv', 'file_name.tar.gz', '.git', 'metadata_compiled.csv']
coswara_covid_list = ['positive_asymp', 'positive_mild', 'positive_moderate']
coswara_cough_list = ['healthy', 'recovered_full']
coughvid_covid_list = ['COVID-19']
coughvid_cough_list = ['healthy']

AUGMENT = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/noise/"
AUGMENT2 = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/static_noise/"
COSWARA = "/Users/andreatamburri/Desktop/Voicemed/Dataset/CoswaraDataset2/"
COUGHVID = "/Users/andreatamburri/Desktop/Voicemed/Dataset/COUGHVID/"
DATA_FOLDER = "/Users/andreatamburri/Desktop/Voicemed/MainDataset/"
SUB_DATA_FOLDER = "CovidClassification"
HOLDOUT_FOLDER = "/Users/andreatamburri/Desktop/Voicemed/HoldOutDataset/"
noises = []

def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    # Create aggregated mean
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for m in y_mean:
        mask.append(m > thresh)

    return mask

def load_audio(path):
    signal, rate = librosa.load(path, sr=SR)
    mask = envelope(signal, rate, SILENCE)
    signal = signal[mask]

    return signal

def melspectrogram(signal):
    signal = librosa.util.normalize(signal)
    spectro = librosa.feature.melspectrogram(
        signal,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT
    )
    spectro = librosa.power_to_db(spectro)
    spectro = spectro.astype(np.float32)
    return spectro

def load_noises(n=4):
    ns = []
    ids = []
    for _ in range(n):
        while True:
            i = np.random.choice(len(noises))
            if i in ids:
                continue
            ids.append(i)
            noise, _ = librosa.load(noises[i], sr=SR)
            if len(noise) < SAMPLE_SIZE:
                continue
            ns.append(noise)
            break

    return ns

def augment(sample, ns, MFCC=True):
    augmented = []
    for noise in ns:
        gap = len(noise)-len(sample)
        point = 0
        if gap > 0:
            point = np.random.randint(low=0, high=len(noise)-len(sample))
        noise = noise[point:point+len(sample)]
        final = []
        for f in range(len(sample)):
            n = noise[f]*NOISE_RATIO
            final.append(sample[f]+n)

        augmented.append(final)
    if MFCC:
        augmented = np.asarray(augmented, dtype=np.float32)

    return augmented

def process(audio, aug=False):
    """
    This function takes as input the path of the audio and returns the extracted features, which are the ù
    melspectrogram values of the signal.
    """

    try:
        signal = load_audio(audio)
    except:
        return []

    if len(signal) < SAMPLE_SIZE:
        return []

    current = 0
    end = False
    features = []

    """
    The loop slides across the signal and splits the audio into smaller signals
    of SAMPLE_LENGTH length. When it reaches the end, the last sample is taken
    from the end of the signal backwards.
    """
    if aug:
        ns = load_noises()

    while not end:
        if current + SAMPLE_SIZE > len(signal):
            sample = signal[len(signal) - SAMPLE_SIZE:]
            end = True
        else:
            sample = signal[current:current + SAMPLE_SIZE]
            current += SAMPLE_SIZE

        if MFCC:
            mfcc = librosa.feature.mfcc(sample, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
            scaler = MinMaxScaler()
            scaler.fit(mfcc)
            mfcc = scaler.transform(mfcc)
            for frame in mfcc:  # this will be of length 14
                features.append(frame)
            if aug:
                signals = augment(sample, ns, MFCC=MFCC)
                for s in signals:
                    mfcc = librosa.feature.mfcc(s, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
                    scaler = MinMaxScaler()
                    scaler.fit(mfcc)
                    mfcc = scaler.transform(mfcc)
                    for frame in mfcc:  # this will be of length 14
                        features.append(frame)
        else:
            m = melspectrogram(sample)
            scaler = MinMaxScaler()
            scaler.fit(m)
            m = scaler.transform(m)
            features.append(m)
            if aug:
                signals = augment(sample, ns, MFCC=MFCC)
                for s in signals:
                    m = melspectrogram(s)
                    scaler = MinMaxScaler()
                    scaler.fit(m)
                    m = scaler.transform(m)
                    features.append(m)

    return features

def generate_dataset(folder, aug=False):
    data = [] #contains [mel, label]
    for i, label in enumerate(LABELS):
        print("Processing: "+label)
        for audio in tqdm(os.listdir(folder+label)):
            if os.path.splitext(audio)[-1] != ".wav":
                continue

            features = process(folder+label+"/"+audio, aug=aug and i == 0) #i==0 augmentation solo sample non covid
            for feat in features:
                data.append([feat, i])

    return data

def generate_dataset(folder, aug=False):
    data = [] #contains [mel, label]
    for i, label in enumerate(LABELS):
        print("Processing: "+label)
        for audio in tqdm(os.listdir(folder+label)):
            if os.path.splitext(audio)[-1] != ".wav":
                continue

            features = process(folder+label+"/"+audio, aug=aug and i == 0)
            for feat in features:
                data.append([feat, i])

    return data


def import_from_coswara(data_source, sub_data_source, output_source, exclude_list, interest_label, cough_labels, covid_labels):
    for folder in os.listdir(data_source+sub_data_source+"/"):
        if folder not in exclude_list:
            if folder == interest_label:
                print("processing " + folder)
                for subfolder in tqdm(os.listdir(data_source+sub_data_source+"/"+folder+"/")):
                    if subfolder in cough_labels:
                        print("processing non covid coughs, labels " + subfolder)
                        for audio in os.listdir(data_source+sub_data_source+"/"+folder+"/"+subfolder+"/"):
                            if audio not in exclude_list:
                                os.system("cp "+data_source+sub_data_source+"/"+folder+"/"+subfolder+"/"+audio+" "+output_source+sub_data_source+"/"+"/cough")
                    elif subfolder in covid_labels:
                        print("processing covid coughs, labels " + subfolder)
                        for audio in os.listdir(data_source+sub_data_source+"/" + folder + "/" + subfolder + "/"):
                            if audio not in exclude_list:
                                os.system( "cp "+data_source+sub_data_source+"/"+folder+"/"+subfolder+"/"+audio+ " "+output_source+sub_data_source+"/"+"/covid")
                    else:
                        continue
            else:
                continue

def import_from_coughvid(data_source, sub_data_source, output_source, exclude_list, cough_labels, covid_labels):
    for folder in os.listdir(data_source+sub_data_source+"/"):
        if folder in cough_labels:
            print("processing non covid label " + folder)
            for audio in tqdm(os.listdir(data_source+sub_data_source+"/"+folder+"/")[:1100]):
                if audio not in exclude_list:
                    os.system("cp "+data_source+sub_data_source+"/"+folder+"/"+audio+" "+output_source+"/cough")
        elif folder in covid_labels:
            print("processing covid label " + folder)
            for audio in tqdm(os.listdir(data_source+sub_data_source+"/" + folder + "/")[:1100]):
                if audio not in exclude_list:
                    os.system("cp " + data_source+sub_data_source+"/" + folder + "/" + audio + " " + output_source + "/covid")
        else:
            continue

def generate_holdout_set(data_source, sub_data_source, output_source, exclude_list, split_ratio):
    for folder in os.listdir(data_source+sub_data_source+"/"):
        if folder not in exclude_list:
            print("generating holdout dataset from MainDataset for " +sub_data_source+" "+ folder)
            main_set = os.listdir(data_source + sub_data_source + "/" + folder + "/")
            holdout_len = floor(len(main_set)*split_ratio)
            shuffle(main_set)
            holdout_set = main_set[:holdout_len]
            for audio in tqdm(holdout_set):
                if audio not in exclude_list:
                    os.system("mv "+data_source+sub_data_source+"/"+folder+"/"+audio+" "+output_source+sub_data_source+"/"+folder)


if __name__ == '__main__':

    generate_holdout_set(DATA_FOLDER, SUB_DATA_FOLDER, HOLDOUT_FOLDER, exclude_list, split_ratio=0.1)

    #import_from_coswara(COSWARA, DATA_FOLDER, exclude_list, "cough", coswara_cough_list, coswara_covid_list)
    #import_from_coughvid(COUGHVID, DATA_FOLDER, exclude_list, coughvid_cough_list, coughvid_covid_list)
    # for audio in os.listdir(AUGMENT):
    #     if os.path.splitext(audio)[-1] != ".wav":
    #         continue
    #
    #     noises.append(AUGMENT + audio)
    #
    # for audio2 in os.listdir(AUGMENT2):
    #     if os.path.splitext(audio2)[-1] != ".wav":
    #         continue
    #
    #     noises.append(AUGMENT2 + audio2)
    #
    # DATA_FOLDER = "/Users/andreatamburri/Desktop/Voicemed/Dataset/CoughDataset/"
    # #TEST_FOLDER = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/test/"
    #
    # data = generate_dataset(DATA_FOLDER,aug=False)
    #
    # #extra = generate_dataset(EXTRA_FOLDER)
    # #data += extra
    # np.random.shuffle(data)
    # np.save("/Users/andreatamburri/Desktop/Voicemed/Covid/Mfcc/dataset_mfcc.npy", data)
    #
    # #test = generate_dataset(TEST_FOLDER)
    # #np.save("test.npy", test)

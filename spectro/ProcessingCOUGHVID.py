import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa.display
import json
from pydub import AudioSegment


SR = 44000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 60
SILENCE = 0.0018
SAMPLE_LENGTH = 0.5 #s
SAMPLE_SIZE = int(np.ceil(SR*SAMPLE_LENGTH))
NOISE_RATIO = 0.3

LABELS = ["None", "Healty", "Symptomatic", "Covid"]

ROOT = "/Users/andreatamburri/Downloads/public_dataset/"
NEW_PATH = "/Users/andreatamburri/Desktop/Voicemed/Dataset/COUGHVID/"

AUGMENT = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/noise/"
AUGMENT2 = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/static_noise/"
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

def augment(sample, ns):
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

    return augmented

def process(audio, aug=False):
    signal = load_audio(audio)

    if len(signal) < SAMPLE_SIZE:
        return []

    current = 0
    end = False
    features = []

    if aug:
        ns = load_noises()

    while not end:
        if current+SAMPLE_SIZE > len(signal):
            sample = signal[len(signal)-SAMPLE_SIZE:]
            end = True
        else:
            sample = signal[current:current+SAMPLE_SIZE]
            current += SAMPLE_SIZE

        features.append(melspectrogram(sample))

        if aug:
            signals = augment(sample, ns)
            for s in signals:
                features.append(melspectrogram(s))

    return features

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

if __name__ == '__main__':

    # file
    print("loading COUGHVID dataset...")
    directory = os.listdir(ROOT)

    for meta in tqdm(directory):

     if os.path.splitext(meta)[-1] == ".json":
         with open(ROOT + meta) as json_file:
             data = json.load(json_file)
             try:
                 sound = AudioSegment.from_file(ROOT + os.path.splitext(meta)[0] + ".webm")
             except:
                 sound = AudioSegment.from_file(ROOT + os.path.splitext(meta)[0] + ".ogg")

             sound.export(NEW_PATH+str(data.get("status"))+"/"+os.path.splitext(meta)[0] + ".wav",format="wav")

     if os.path.splitext(meta)[-1] != ".json":
             continue



    #test = "0a1b4119-cc22-4884-8f0f-34e8207c31d1.json"
    #with open(ROOT+test) as json_file:
     #   data = json.load(json_file)

    #data = json.loads(ROOT+test)
    #print(data.get('cough_detected'))
    #src = "fb07cc85-4342-48ab-a748-6f71ac975aff.ogg"
    #dst = "fb07cc85-4342-48ab-a748-6f71ac975aff.wav"

    # convert wav to mp3
    #sound = AudioSegment.from_file(root + src)
    #sound.export(new_path + dst, format="wav")

    # for audio in os.listdir(AUGMENT):
    #     if os.path.splitext(audio)[-1] != ".wav":
    #         continue
    #
    #     noises.append(AUGMENT+audio)
    #
    # for audio2 in os.listdir(AUGMENT2):
    #     if os.path.splitext(audio2)[-1] != ".wav":
    #         continue
    #
    #     noises.append(AUGMENT2 + audio2)
    #
    # DATA_FOLDER = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/dataset/"
    # TEST_FOLDER = "/Users/andreatamburri/Desktop/Voicemed/Detector/CoughModelData/test/"
    #
    # data = generate_dataset(DATA_FOLDER, True)
    #
    # #extra = generate_dataset(EXTRA_FOLDER)
    # #data += extra
    # np.random.shuffle(data)
    # np.save("dataset.npy", data)
    #
    # test = generate_dataset(TEST_FOLDER)
    # np.save("test.npy", test)


import os
from tqdm import tqdm
import sys
import detector
import json


coswara_path = "/Users/andreatamburri/Documents/GitHub/Coswara-Data/"
out_path = "/Users/andreatamburri/Desktop/Voicemed/Dataset/CoswaraDataset2/"
spectro_path = "/Users/andreatamburri/Documents/GitHub/CoughAPI/cough_detector/"




def detector_extacrtion(spectro_path, coswara_path, out_path):

    sys.path.insert(0, spectro)
    cough = detector.Detector(spectro_path + "new_model.json", spectro_path + "new_model.h5")

    out = outpath+"AllSamples/"
    dataset = path+"DataSet/"

    for folder in os.listdir(coswara_path):
        if not os.path.isdir(coswara_path+folder):
            continue
        print("Extracting "+folder)
        for f in os.listdir(coswara_path+folder):
            if "tar.gz" in f:
                os.system("tar xvf "+coswara_path+folder+"/"+f+" -C "+out)
                os.system("mv "+out_path+folder+"/* "+out)
                os.system("rm -r "+out_path+folder)

    patients = os.listdir(out_path)
    empty = 0
    covid = 0
    for p in tqdm(patients):
        if not os.path.isdir(out_path + p):
            continue

        content = os.listdir(out_path + p)
        if len(content) == 0:
            empty += 1
            continue

        label = "cough"
        if "metadata.json" in content:
            with open(out + p + "/metadata.json", "r") as meta:
                data = json.load(meta)
                if data["covid_status"] != "healthy":
                    label = "covid"
                    covid += 1

        valid = False
        audios = os.listdir(out_path + p)
        for f in range(len(audios)):
            if ".wav" in audios[f] and "cough" in audios[f]:
                try:
                    pieces, _ = cough.detect(out_path + p + "/" + audios[f], dataset + label + "/" + str(f), False)
                    if len(pieces) > 0:
                        valid = True
                except:
                    continue

        if not valid:
            empty += 1

    print("Out of " + str(len(patients)) + " samples, " + str(empty) + " are empty")
    print("Total covid samples: " + str(covid))

def complete_extraction(root_path, out_path):

    print("loading COSWARA dataset...")
    directory = os.listdir(root_path)
    exclude_list = ['LICENSE.md', '.DS_Store', 'README.md', 'combined_data.csv','file_name.tar.gz', '.git']
    final_dir = [element for element in directory if element not in exclude_list]
    print(final_dir)
    for sub in final_dir:
        os.system("cat " + coswara_path + sub + "/*.tar.gz.* > " + coswara_path + sub + "file_name.tar.gz")
        os.system("tar -xvzf " + coswara_path + sub + "file_name.tar.gz")

    for sub in tqdm(final_dir):
        if os.path.isdir(tmp_path + sub) == True:
            tmp_dir = os.listdir(tmp_path + sub)
            final_tmp_dir = [element for element in tmp_dir if element not in exclude_list]
            for sub2 in final_tmp_dir:
                if os.path.isdir(tmp_path + sub + "/" + sub2) == True:
                    tmp_dir2 = os.listdir(tmp_path + sub + "/" + sub2)
                    for meta in tmp_dir2:
                        print(meta)
                        if os.path.splitext(meta)[-1] == ".json":
                            with open(tmp_path + sub + "/" + sub2 + "/" + meta) as json_file:
                                data = json.load(json_file)
                                label = str(data.get("covid_status"))

                                os.system(
                                    "cp -r " + tmp_path + sub + "/" + sub2 + "/cough*.wav " + out_path + "cough/" + label + "/")
                                os.system(
                                    "mv " + out_path + "cough/" + label + "/cough-heavy.wav " + out_path + "cough/" + label + "/cough-heavy-" + sub2 + ".wav")
                                os.system(
                                    "mv " + out_path + "cough/" + label + "/cough-shallow.wav " + out_path + "cough/" + label + "/cough-shallow-" + sub2 + ".wav ")

                                os.system(
                                    "cp -r " + tmp_path + sub + "/" + sub2 + "/breathing*.wav " + out_path + "breath/" + label + "/")
                                os.system(
                                    "mv " + out_path + "breath/" + label + "/breathing-deep.wav " + out_path + "breath/" + label + "/breathing-deep-" + sub2 + ".wav")
                                os.system(
                                    "mv " + out_path + "breath/" + label + "/breathing-shallow.wav " + out_path + "breath/" + label + "/breathing-shallow-" + sub2 + ".wav ")

                        if os.path.splitext(meta)[-1] != ".json":
                            continue

if __name__ == '__main__':

method = "complete"

if method == "complete":
    complete_extraction(coswara_path, out_path)
else:
    detector_extacrtion(spectro_path, coswara_path, out_path)


 #temporary workaround, use it if needed
    # print("loading COSWARA dataset...")
    # tmp_path = "/Users/andreatamburri/Desktop/Voicemed/Dataset/CoswaraDataset2/tmp_dataset/"
    # directory = os.listdir(tmp_path)
    # exclude_list = ['LICENSE.md', '.DS_Store', 'README.md', 'combined_data.csv', 'file_name.tar.gz', '.git']
    # final_dir = [element for element in directory if element not in exclude_list]
    #
    # for sub in tqdm(final_dir):
    #     tmp_dir = os.listdir(tmp_path + sub)
    #     final_tmp_dir = [element for element in tmp_dir if element not in exclude_list]
    #     for sub2 in final_tmp_dir:
    #         if os.path.isdir(tmp_path + sub + "/" + sub2) == True:
    #             tmp_dir2 = os.listdir(tmp_path + sub + "/" + sub2)
    #             for meta in tmp_dir2:
    #                 print(meta)
    #                 if os.path.splitext(meta)[-1] == ".json":
    #                     with open(tmp_path + sub + "/" + sub2 + "/" + meta) as json_file:
    #                         data = json.load(json_file)
    #                         label = str(data.get("covid_status"))
    #
    #                         os.system("cp -r " + tmp_path + sub + "/" + sub2 + "/cough*.wav " + out_path + "cough/" + label + "/")
    #                         os.system(
    #                             "mv " + out_path + "cough/" + label + "/cough-heavy.wav "+ out_path + "cough/" + label + "/cough-heavy-" + sub2 + ".wav")
    #                         os.system(
    #                             "mv " + out_path + "cough/" + label + "/cough-shallow.wav " + out_path + "cough/" + label + "/cough-shallow-" + sub2 + ".wav ")
    #
    #                         os.system("cp -r " + tmp_path + sub + "/" + sub2 + "/breathing*.wav " + out_path + "breath/" + label + "/")
    #                         os.system(
    #                             "mv " + out_path + "breath/" + label + "/breathing-deep.wav " + out_path + "breath/" + label + "/breathing-deep-" + sub2 + ".wav")
    #                         os.system(
    #                             "mv " + out_path + "breath/" + label + "/breathing-shallow.wav " + out_path + "breath/" + label + "/breathing-shallow-" + sub2 + ".wav ")
    #
    #                 if os.path.splitext(meta)[-1] != ".json":
    #                     continue

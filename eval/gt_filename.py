import os
import glob
import numpy as numpy
import argparse

BASE_DIR = os.path.abspath("__file__/..")
ROOT_DIR = os.path.dirname(BASE_DIR)

def get_subfolder_names(path):
    subfolders = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            p = os.path.join(path,f)
            subfolders.append(p)
    return subfolders
def get_subfolder_names_cath(path):
    subfolders = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)) and not os.path.join(path, f).endswith("CATH"):
            p = os.path.join(path,f)
            subfolders.append(p)
    return subfolders
def main(args):
    eval_data = []
    data_path = f"{ROOT_DIR}/job_specs/vessel.txt"
    with open(data_path, "r") as file:
        for line in file:
            eval_data.append(line.strip())
    # print(eval_data[0][:9])

    path = f"{ROOT_DIR}/xca_dataset"
    output_txt_path = "gt_path.txt"
    output_predict_path = "out_path.txt"
    all_data=[]

    for i in range(len(eval_data)):
        newpath = os.path.join(path,eval_data[i][:9],"ground_truth")
        data = os.path.join(newpath,eval_data[i])
        all_data.append(data)
    # print(all_data)
    image_data = []
    for i in all_data:
        p = os.path.join(i,"*")
        tmp = sorted(glob.glob(p))
        image_data += tmp
    with open(output_txt_path, "w") as file:
        for d in image_data:
            file.write(d + "\n")
    out_data = []
    for image_id in image_data:
        image_name = image_id.split("/")[-1]
        file_name  = image_id.split("/")[-2]
        predict_path = f"{ROOT_DIR}/outputs/dev/custom-{file_name}-gap1-2l"
        output_name = args.dir
        predict_path = os.path.join(predict_path,output_name)
        for f in os.listdir(predict_path):
            if f.endswith("val_refine"):
                p = os.path.join(predict_path,f,"masks_0",image_name)
                out_data.append(p)
    with open(output_predict_path, "w") as file:
        for d in out_data:
            file.write(d + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dir")
    args = parser.parse_args()
    main(args)
from imutils import paths
import argparse
import shutil
import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dataset", type=str, required=True, help="Path to the CASIA dataset")
args = parser.parse_args()

total_persons = 10572
train_persons = 10000

dataset_path = "casia_webface"

for i in range(total_persons):
    print("[INFO] Copying person: {}/{}".format(i+1,total_persons))
    if i < train_persons:
        set_ = "train"
    else:
        set_ = "val"
    
    create_dir(os.path.join(dataset_path, set_, "person_{}".format(i)))
    src_image_paths = list(paths.list_images(os.path.join(args.dataset, "person_{}".format(i))))
    
    for src_image_path in src_image_paths:
        image_name = src_image_path.split("/")[-1]
        dst_image_path = os.path.join(dataset_path, set_, "person_{}".format(i), image_name)
        shutil.copy(src_image_path, dst_image_path)
        
    

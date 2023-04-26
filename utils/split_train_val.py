import os
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, help="Path to root directory", required=True)
    args = parser.parse_args()

    folders = ["target_a", "target_b", "target_c", "source", "source_aug_6", "masks"]

    assert os.path.exists(os.path.join(args.root_dir, "masks")), "Make sure you downloaded masks folder and placed it into same folder containing target_a, target_b, target_c etc. If using masks_360p, make sure to rename."
    assert all(os.path.exists(os.path.join(args.root_dir, folder)) for folder in folders), "Provided dir is not train/val PPR10k dataset"

    for folder in folders:
        data_root = os.path.join(args.root_dir, folder)
        target_train_root = os.path.join(args.root_dir, "train", folder)
        target_val_root = os.path.join(args.root_dir, "val", folder)

        os.makedirs(target_train_root, exist_ok=True)
        os.makedirs(target_val_root, exist_ok=True)

        count_train_file, count_val_file = 0, 0
        files = os.listdir(data_root)

        for file in tqdm(files, total=len(files), desc=f"Splitting {folder} into train/val..."):
            if file[0] == ".":
                continue
            source_path = os.path.join(data_root, file)
            if int(file.split('_')[0]) < 1356:
                target_path = os.path.join(target_train_root, file)
                count_train_file += 1
            else:
                target_path = os.path.join(target_val_root, file)
                count_val_file += 1
            os.rename(source_path, target_path)

import argparse
import os
import pathlib
import cv2
import random
import csv


SKIP_FIRST_N_FRAMES = "4"
SPLIT_RATIO = 0.75

def write_to_lists(lists, directory, stimulus_class):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if pathlib.Path(f).suffix == ".avi":
            video = cv2.VideoCapture(f)
            num_frames = str(int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1) # subtract 1 from total count in case of corruption
            row = [f, stimulus_class, "4", num_frames]
            lists.append(row)
    return lists
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type = str)
    parser.add_argument("5g_dir", type = str)
    parser.add_argument("control_dir", type = str)
    args = parser.parse_args()
    return args.test_dir, args.control_dir

if __name__ == "__main__":
    test, test_5g, control = parse_args()
    header_row = ["file", "class", "begin frame", "end frame"]
    lists = []
    lists = write_to_lists(lists, test, "2")
    lists = write_to_lists(lists, test_5g, "3")
    lists = write_to_lists(lists, control, "1")
    random.shuffle(lists)
    split = int(len(lists)*SPLIT_RATIO)
    train_list = lists[:split]
    eval_list = lists[split:]
    train_list.insert(0, header_row)
    eval_list.insert(0, header_row)

    with open("train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train_list)

    with open("eval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(eval_list)
    
    

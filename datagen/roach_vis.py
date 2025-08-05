import cvzone
import pandas as pd
import argparse
import cv2
import numpy as np


SKIP_N_ROWS = 3 #to get headers from the txt file
REMOVE_N_ROWS = 60 #discard 2 seconds of video
VID_WIDTH = 1280 #pixels
VID_HEIGHT = 720 #pixels
VID_FPS = 30

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()
    return args

def make_frame(row):
    X = int(row['X'])
    Y = int(row['Y'])
    angle = row['Angle (deg)']
    #print(X, Y, angle)
    img_back = np.zeros((VID_HEIGHT, VID_WIDTH,3), np.uint8)*255
    img_roach = cv2.imread("bug.png", cv2.IMREAD_UNCHANGED)
    img_roach = cvzone.rotateImage(img_roach, angle)
    add_transparent_image(img_back, img_roach, X, Y)
    img_back[img_back!=0.0] = 255
    return img_back

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def generate_video(data):
    df = pd.read_csv(data, sep="(\s{3,})", skiprows=SKIP_N_ROWS, engine='python')
    df.drop(index=df.index[:REMOVE_N_ROWS], inplace=True)
    vid_filename = data.replace(".txt", ".avi")
    out_vid = cv2.VideoWriter(vid_filename, cv2.VideoWriter_fourcc(*'DIVX'), VID_FPS, (VID_WIDTH, VID_HEIGHT))
    prev_angle = df['Angle (deg)'].iloc[0] # pull the first timestamps's angle as the prev_angle
    flipped_180 = False
    for idx,row in df.iterrows():
        curr_angle = row['Angle (deg)']
        delta_angle = curr_angle - prev_angle
        if abs(delta_angle) > 45: # this can be any number probably greater than 1 degree and less than 180 since each normal frame should only have tenths of angle difference
            flipped_180 = not flipped_180 # if a flip happens, then flipped_180 will be true until it flips back to the correct value
        if flipped_180: # skip frame if flipped_180 is true
            row['Angle (deg)']-= delta_angle # adjust angle back to actual
            print("THERE'S A FLIPPING ROACH!")
        img = make_frame(row)
        out_vid.write(img)
        prev_angle = curr_angle
    out_vid.release()

if __name__ == "__main__":
    args=parse_args()
    generate_video(args.file)

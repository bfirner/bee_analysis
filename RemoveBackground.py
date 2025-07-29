#!/usr/bin/python3

# take a file as input, and output a new file with the background subtracted
# R. P. Martin, (c) 2023, GPL 2 

# For the algorithm, see: https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html

import cv2 
import argparse
import datetime

parser = argparse.ArgumentParser(description='This program removes the background from a video')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='input.avi')
parser.add_argument('--output', type=str, help='output file name', default='output.avi')
parser.add_argument('--lognum', type=int, help='log frame number', default=0)
parser.add_argument('--alg', type=str, help='Background subtraction algorithm, KNN or MOG2.', default='MOG2')

args = parser.parse_args()

outputFileName = args.output
frameLogNumber = args.lognum

if args.alg == 'MOG2':
    backSubAlg = cv2.createBackgroundSubtractorMOG2()
else:
    backSubAlg = cv2.createBackgroundSubtractorKNN()

capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open file: ' + args.input)
    exit(0)


frame_num = 0
ret, frame = capture.read()
if frame is None:
    print('Warning! No initial frame!')
height, width, channels = frame.shape

size = (width, height)
fps = int(capture.get(cv2.CAP_PROP_FPS))
dateNow = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print ('Log: Processing video \"%s\" size %d:%d fps %d at %s' % (args.input,size[0],size[1],fps,dateNow) )

# file to write out 
out = cv2.VideoWriter(outputFileName,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#out = cv2.VideoWriter(outputFileName,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSubAlg.apply(frame)
    masked = cv2.bitwise_and(frame, frame, mask=fgMask)
    out.write(masked)
    frame_num = frame_num + 1

    # debugging code 
    #cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    #cv2.imshow("Mask Applied to Image", masked)
    #cv2.imshow('Frame', frame)
    #cv2.imshow('FG Mask', fgMask)

    #keyboard = cv2.waitKey(30)
    #if keyboard == 'q' or keyboard == 27:
        #break

    if (frameLogNumber > 0):
        dateNow = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if (frame_num % frameLogNumber) == 1 :
            print("Log: converted frame %d at: %s " % (frame_num,dateNow))
            

print("Log: done with video")
cv2.destroyAllWindows()
out.release()


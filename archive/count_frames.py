#!/usr/bin/python3

# This Python peforms two functions (1)creates the counts.csv needed by the make_csv.py script.
# and (2) create background subtracted videos ( noBackground - noBg ) 
# Both functions use GNU parallel to utilize multiple cores
# 
# For Frame counts, it runs ffprobe to count all the frames in a directory of h264 files, using 
# the parallel program to run ffprobe 60 copies at a time.
# It then cleans up the output of parallel to get a csv files of file, framecount
# to get a counts.csv file 
#
# 

# Author: R. P. Marin

import argparse 
import sys 
import os
import subprocess

parallel_command_count = [  "/research/projects/grail/rmartin/bin/parallel","--jobs","60","--results","pcounts.csv","ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=nb_read_frames","-of","csv=p=0",":::" ]

parallel_command_noBg = [  "/research/projects/grail/rmartin/bin/parallel","--jobs","15","--results","noBg.csv","/koko/system/anaconda/envs/python38/bin/python3", "/research/projects/grail/rmartin/analysis-results/code/bee_analysis6/RemoveBackground.py","--input","{}","--output","{}.avi" ,":::" ]

#/common/home/rmartin/.bash_H.rlab2:awk -F, '{print $1}' < dataset-time.csv | sed 's/\/research\/projects\/grail\/rmartin\/bee-videos\/yard\/2023-10-18_2023-10-21/../g' | parallel --jobs 20 /koko/system/anaconda/envs/python38/bin/python3 ~/grail/analysis-results/code/bee_analysis5/RemoveBackground.py --input  {} --output {}.avi --lognum 100000

parallel_background = [  "/research/projects/grail/rmartin/bin/parallel","--jobs","60","--results","bgresults.csv","/research/projects/grail/rmartin/bin/RemoveBackground.py","--input","{}","--output","{}.avi",":::" ]

cleanup_shell_cmd = "awk -F, '{printf(\"%s,%s\\n\",$10,$11)} ' < pcounts.csv  | sed 's/V1/filename/g' | sed 's/Stdout/frames/g' | sed 's/\",\"/XYZ/g' | sed s/,//g | egrep 'frames|h264' | sed 's/\"//g' | sed 's/XYZ/,/g' | sed 's/frames/,frames/' > counts.csv"

def main():
    bgalg = 'keepBg'
    which = 'count'

    parser = argparse.ArgumentParser(description='This program processes video files to either count frames or remove the background. Assumes video files end in .h254 extension')
    parser.add_argument('--dir', type=str, help='Directory of videos.', default='.')
    parser.add_argument('--which', type=str, help='which operation: <count|noBg>', default='count')
    parser.add_argument('--bgalg', type=str, help='Background subtraction algorithm, KNN or MOG2.', default='MOG2')
    parser.add_argument('--fps', type=int, help='frames/sec', default=3)

    args = parser.parse_args()

    directory = args.dir
    which = args.which
    bgalg = args.bgalg 
    fps = args.fps

    # cd to the current directory 
    try:
        os.chdir(directory)
    except OSError as e:
        print(f"Error with directory: {e}, exiting")
        sys.exit(-1)
        
    # get a list of all the video files
    h264_files = [f for f in os.listdir() if f.endswith('.h264')]
    # append the list to the parallel command args

    if (len(h264_files) == 0):
        print("No h264 files in the directory %s, exiting" % (directory) )
        sys.exit(-1)

    # figure out if we are running the count frames or background subtraction
    if (which == 'count'):
        parallel_command = parallel_command_count
    elif (which == 'noBg'):
        parallel_command = parallel_command_noBg
    else:
        print ("error, no command specified. -h for help")
        sys.exit(-1)

    # add the filenames to the end of the gnu parallel command 
    for filename in h264_files:
            parallel_command.append(filename)
            
    print ("about to run ffprobe in parallel: ", parallel_command)
    # run the parallel command 
    process = subprocess.Popen(parallel_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decode the output and split it into lines
    parallel_output = stdout.decode('utf-8').splitlines()

    # run the cleanup command to generate the counts.csv that holds the frame count for each file
    if (which == 'count'):
        print("about to run:", cleanup_shell_cmd)
        cleanup_result = subprocess.run(cleanup_shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        cleanup_output = cleanup_result.stdout.decode('utf-8')
    else:
        print("Ran Background Subtraction", cleanup_shell_cmd)        

# run the main program 
if __name__ == "__main__":
    main()


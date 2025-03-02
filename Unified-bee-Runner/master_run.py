"""
TODO: Work to add common pitfalls
TODO: make sure everything executes
TODO: Finish implementing debugging for the pipeline
TODO: ADD the ability to skip steps
"""

import os
import logging
import subprocess
from ArgParser import get_args
from test_steps import test_step_0, test_step_1, test_step_2, test_step_3, test_step_4

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

try:
    args = get_args()

    # getting the path working
    subprocess.run(
        "export PATH=/koko/system/anaconda/envs/python38/bin:$PATH", shell=True
    )

    path = args.data_path

    os.chdir(path)
    logging.info("truncating BEERUN.log")
    file_list = os.listdir()

    # truncate the file, resets the logs
    if "BEERUN.log" in file_list:
        to_truncate = open("BEERUN.log", "r+")
        to_truncate.truncate(0)
        to_truncate.close()

    logging.info("(0) Starting the pipeline")
except Exception as e:
    logging.error(f"Error: {e}")
    raise "Something went wrong in the beginning"
# convert the videos

if args.start > args.end:
    raise "You can't have the start be higher than the end"

#  if the videos a .h264, convert to .mp4, else, just make a counts.csv
if args.start <= 0 and args.end >= 0:
    logging.info("(0) Starting the video conversions, always defaulting to .mp4")
    try:
        if "Video_Frame_Counter" in file_list:
            subprocess.run("rm -rf Video_Frame_Counter", shell=True)

        subprocess.run(
            "git clone https://github.com/Elias2660/Video_Frame_Counter.git >> CLONES.log 2>&1",
            shell=True,
        )
        file_list = os.listdir(path)
        contains_h264 = True in [
            ".h264" in file for file in file_list
        ]  # if there is at least a single h264 file
        contains_mp4 = True in [
            ".mp4" in file for file in file_list
        ]  # if there is a single mp4 file

        if "CONVERSION_STEP_0.log" in file_list:
            to_truncate = open("CONVERSION_STEP_0.log", "r+")
            to_truncate.truncate(0)
            to_truncate.close()

        arguments = f"--max-workers {args.max_workers_frame_counter}"
        if contains_h264 and contains_mp4:
            raise "Both types of file are in this directory, please remove one"
        elif contains_h264:
            logging.info(
                "Converting .h264 to .mp4, old h264 files can be found in the h264_files folder"
            )
            subprocess.run(
                f"python Video_Frame_Counter/h264tomp4.py {arguments} >> CONVERSION_STEP_0.log 2>&1",
                shell=True,
            )
        elif contains_mp4:
            logging.info("No conversion needed, making counts.csv")
            subprocess.run(
                f"python Video_Frame_Counter/make_counts.py {arguments} >> CONVERSION_STEP_0.log 2>&1",
                shell=True,
            )
        else:
            raise "Something went wrong with the file typing, as it seems that there are no .h264 or .mp4 files in the directory"

        # truncating chmoding.log, if it exists
        if "chmoding.log" in file_list:
            to_truncate = open("chmoding.log", "r+")
            to_truncate.truncate(0)
            to_truncate.close()

        subprocess.run("chmod -R 777 . >> chmoding.log 2>&1", shell=True)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise "Something went wrong in step 0"
else:
    logging.info(
        f"Skipping step 0, given the start ({args.start}) and end ({args.end}) values"
    )


if args.start <= 1 and args.end >= 1:
    logging.info("(1) Starting the background subtraction")
    try:
        if args.background_subtraction_type is not None:
            logging.info("Starting the background subtraction")
            # truncating the subtraction logging file
            if "SUBTRACTION_STEP_1.log" in file_list:
                to_truncate = open("SUBTRACTION_STEP_1.log", "r+")
                to_truncate.truncate(0)
                to_truncate.close()

            # removing the background subtraction folder if it exists
            if "Video_Subtractions" in file_list:
                subprocess.run("rm -rf Video_Subtractions", shell=True)

            subprocess.run(
                "git clone https://github.com/Elias2660/Video_Subtractions.git >> CLONES.log 2>&1",
                shell=True,
            )

            arguments = f"--subtractor {args.background_subtraction_type} --max-workers {args.max_workers_background_subtraction}"
            subprocess.run(
                f"python Video_Subtractions/Convert.py {arguments} >> SUBTRACTION_STEP_1.log 2>&1",
                shell=True,
            )

        else:
            logging.info("No background subtraction type given, skipping this step")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise "Something went wrong in step 1"
else:
    logging.info(
        f"Skipping step 1, given the start ({args.start}) and end ({args.end}) values"
    )

if args.start <= 2 and args.end >= 2:
    logging.info("(2) Starting the dataset creation")
    try:
        log_list = [
            file.strip()
            for file in os.listdir()
            if file.strip() == "logNo.txt"
            or file.strip() == "logPos.txt"
            or file.strip() == "logNeg.txt"
        ]
        logging.info(f"Creating the dataset with the files: {log_list}")

        if "Dataset_Creator" in file_list:
            subprocess.run("rm -rf Dataset_Creator", shell=True)

        subprocess.run(
            "git clone https://github.com/Elias2660/Dataset_Creator.git >> CLONES.log 2>&1",
            shell=True,
        )
        if args.files is None:
            string_log_list = ",".join(log_list).strip().replace(" ", "")
        else:
            string_log_list = args.files

        logging.info("Truncating the MAKE_DATASET_2.log to zero, if it exists")
        if "MAKE_DATASET_2.log" in file_list:
            to_truncate = open("MAKE_DATASET_2.log", "r+")
            to_truncate.truncate(0)
            to_truncate.close()

        arguments = f"--files '{string_log_list}' --starting-frame {args.starting_frame} --frame-interval {args.frame_interval}"
        subprocess.run(
            f"python Dataset_Creator/Make_Dataset.py {arguments} >> MAKE_DATASET_2.log 2>&1",
            shell=True,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise "Something went wrong in step 2"
else:
    logging.info(
        f"Skipping step 2, given the start ({args.start}) and end ({args.end}) values"
    )

logging.info("(3) Splitting up the data")
if args.start <= 3 and args.end >= 3:
    try:
        # !!! VERY IMPORTANT !!!, change the path_to_file to the path of the file that was created in the last step

        if "working_bee_analysis" in file_list:
            subprocess.run("rm -rf working_bee_analysis", shell=True)

        BEE_ANALYSIS_CLONE = "https://github.com/Elias2660/working_bee_analysis.git"
        subprocess.run(f"git clone {BEE_ANALYSIS_CLONE} >> CLONES.log 2>&1", shell=True)
        dir_name = BEE_ANALYSIS_CLONE.split(".")[1].strip().split("/")[-1].strip()

        logging.info("truncating DATASET_SPLIT_3.log, if it exists")

        if "DATASET_SPLIT_3.log" in file_list:
            to_truncate = open("DATASET_SPLIT_3.log", "r+")
            to_truncate.truncate(0)
            to_truncate.close()

        arguments = f"--k {args.k} --model {args.model} --seed {args.seed} --width {args.width} --height {args.height} --path_to_file {dir_name} --frames_per_sample {args.frames_per_sample} --crop_x_offset {args.crop_x_offset} --crop_y_offset {args.crop_y_offset} --epochs {args.epochs}"
        if args.only_split:
            arguments += " --only_split"
        if args.training_only:
            arguments += " --training_only"
        subprocess.run(
            f"python {dir_name}/make_validation_training.py {arguments} >> DATASET_SPLIT_3.log 2>&1",
            shell=True,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise "Something went wrong in step 3"
else:
    logging.info(
        f"Skipping step 3, given the start ({args.start}) and end ({args.end}) values"
    )

logging.info("(4) Starting the tar sampling")
if args.start <= 4 and args.end >= 4:
    try:
        subprocess.run("python Dataset_Creator/dataset_checker.py", shell=True)

        if "VideoSamplerRewrite" in file_list:
            subprocess.run("rm -rf VideoSamplerRewrite", shell=True)


        subprocess.run(
            f"git clone https://github.com/Elias2660/VideoSamplerRewrite.git >> CLONES.log 2>&1",
            shell=True,
        )

        # ? No need to truncate dataprep.log because the Dataprep package already truncates
        subprocess.run(
            "chmod -R 777 . >> chmoding.log 2>&1", shell=True
        )  # keep chmoding to make sure that the permissions are correct to sample videos
        arguments = f"--frames-per-sample {args.frames_per_sample} --number-of-samples {args.number_of_samples} --normalize {args.normalize} --out-channels {args.out_channels} --max-workers {args.max_workers_video_sampling}"
        if args.crop:
            arguments += f" --crop --x-offset {args.crop_x_offset} --y-offset {args.crop_y_offset} --out-width {args.width} --out-height {args.height}"
        if args.debug:
            arguments += " --debug"
        subprocess.run(
            f"python VideoSamplerRewrite/Dataprep.py {arguments} >> dataprep.log 2>&1",
            shell=True,
        )
        
    except Exception as e:
        logging.error(f"Error: {e}")
        raise "Something went wrong in step 4"
    finally:
        test_step_4("VideoSamplerRewrite")
        
else:
    logging.info(
        f"Skipping step 4, given the start ({args.start}) and end ({args.end}) values"
    )

logging.info("(5) Starting the model training")
if args.start <= 5 and args.end >= 5:
    try:
        summary = """
        Running model training can be tricky. As this runs, make sure that you're running the correct scripts
        
        Additionally, a big problem that can come up is that the wrong python environment is being used. It's possible that you might have to switch from python38 to 39, or vice versa
        """
        logging.info("")
        subprocess.run("chmod -R 777 . >> chmoding.log 2>&1", shell=True)
        subprocess.run("./training-run.sh", shell=True)
        subprocess.run("chmod -R 777 . >> chmoding.log 2>&1", shell=True)

        logging.info("Pipeline complete, training is occuring")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise "Something went wrong in step 5"
else:
    logging.info(
        f"Skipping step 5, given the start ({args.start}) and end ({args.end}) values"
    )

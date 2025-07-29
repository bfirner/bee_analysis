import typer
import pathlib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

LOG_TIME_FORMAT = "%Y%m%d_%H%M%S"
FILE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_VIDEO_LENGTH = 1201 # number of seconds

app = typer.Typer()

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)

def _assert_logs(files_dir: pathlib.Path):
    assert (files_dir / "logNeg.txt").is_file()
    assert (files_dir / "logNo.txt").is_file()
    assert (files_dir / "logPos.txt").is_file()

def parse_frame_counts(files_dir: pathlib.Path):
    """
    counts.csv should have the header: 'filename,frames'
    """
    assert (files_dir / "counts.csv").is_file()
    counts_df = pd.read_csv(files_dir / "counts.csv")
    counts_df["filename"] = counts_df["filename"].apply(lambda x: files_dir / x)
    return counts_df.sort_values(by="filename")

def parse_logs(files_dir: pathlib.Path) -> pd.DataFrame:
    """Read in all log (Pos, Neg, No) and store into dataframe.

    Args:
        files_dir: directory containing all data

    Returns:
        pd.DataFrame: contains all of the various events sorted by time
    """
    _assert_logs(files_dir)
    events_list = []
    for file_path in files_dir.glob("*.txt"):
        event_type = file_path.stem
        with open(file_path, 'r') as file:
            for line in file:
                timestamp = line.strip()
                event = {
                    "event_type" : event_type,
                    "ts" : timestamp
                }
                events_list.append(event)
    return pd.DataFrame(events_list).sort_values(by="ts").reset_index(drop=True)

def _find_latest_video(
    logged_timestamp: str,
    file_epoch_map_df: pd.DataFrame
) -> pathlib.Path:
    """Finds video file containing the given logged timestamp.

    Assumes that the closest video filename < given logged timestamp is said video.

    Args:
        logged_timestamp: timestamp from the log*.txt
        file_epoch_map_df: contains the filename to epoch time mappings

    Returns:
        pathlib.Path: said video file name.
    """
    epoch_timestamp = datetime.strptime(logged_timestamp, LOG_TIME_FORMAT).timestamp()
    latest_filename = file_epoch_map_df.loc[file_epoch_map_df['epoch_ts'] <= epoch_timestamp, 'filename'].iloc[np.argmin(np.abs(file_epoch_map_df.loc[file_epoch_map_df['epoch_ts'] <= epoch_timestamp, 'epoch_ts'] - epoch_timestamp))]
    return latest_filename

def map_file_names_to_epoch(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Create a df containing the timestamped filenames mapped to epoch time.

    Args:
        counts_df: use filenames from here to map to epoch

    Returns:
        file_epoch_map_df: contains the filename - epoch map.
    """
    epoch_list = []
    for index, row in counts_df.iterrows():
        filename = row["filename"].stem
        filename_remove_micro = filename.split(".")[0]
        epoch_timestamp = datetime.strptime(filename_remove_micro, FILE_TIME_FORMAT).timestamp()
        epoch_names = {
            "filename" : row["filename"],
            "epoch_ts" : int(epoch_timestamp)
        }
        epoch_list.append(epoch_names)
    file_epoch_map_df = pd.DataFrame(epoch_list)

    return file_epoch_map_df

def _add_event_end_info(events_df: pd.DataFrame, counts_df: pd.DataFrame, fps: int) -> pd.DataFrame:
    """Add extra column to events_df with the ending timestamp of each event.

    Just copy the next row's start timestamp and for the last row calculate using frame count.

    Args:
        events_df: contains the sorted log files events
        counts_df: contains the total frame counts per video from counts.csv
        fps: frames per second

    Returns:
        events_df: updated dataframe
    """
    events_df["end_ts"] = events_df["ts"].shift(-1)
    last_video_name = str(counts_df.iloc[-1]["filename"])
    last_video_name = last_video_name.split("/")[-1].replace(".h264", "")[:-7]
    last_video_datetime_obj = datetime.strptime(last_video_name, FILE_TIME_FORMAT)
    new_last_video_datetime = last_video_datetime_obj + timedelta(seconds= int(counts_df.iloc[-1]["frames"]/fps))
    new_timestamp = new_last_video_datetime.strftime(LOG_TIME_FORMAT)
    events_df.iloc[-1, events_df.columns.get_loc("end_ts")] = new_timestamp

    return events_df

def _add_video_end_info(file_epoch_map_df: pd.DataFrame, counts_df: pd.DataFrame, fps: int) -> pd.DataFrame:
    """Add extra column to file_epoch_map_df with the ending timestamp of each video.

    Calculate the video end info from framecount.

    Args:
        file_epoch_map_df: contains the filename to epoch time mappings
        counts_df: contains the total frame counts per video from counts.csv
        fps: frames per second

    Returns:
        file_epoch_map_df: updated dataframe
    """

    file_epoch_map_df["length"] = file_epoch_map_df["epoch_ts"].shift(-1) - file_epoch_map_df["epoch_ts"]
    file_epoch_map_df.iloc[-1, file_epoch_map_df.columns.get_loc("length")] = counts_df.iloc[-1]["frames"] / fps
    file_epoch_map_df["end_epoch_ts"] = file_epoch_map_df["epoch_ts"].shift(-1)
    last_video_name = str(counts_df.iloc[-1]["filename"])
    last_video_name = last_video_name.split("/")[-1].replace(".h264", "")[:-7]
    last_video_datetime_obj = datetime.strptime(last_video_name, FILE_TIME_FORMAT)
    new_last_video_datetime = last_video_datetime_obj + timedelta(seconds= int(counts_df.iloc[-1]["frames"]/fps))

    file_epoch_map_df.iloc[-1, file_epoch_map_df.columns.get_loc("end_epoch_ts")] = new_last_video_datetime.timestamp()
    return file_epoch_map_df

def _filter_events(labels_list, fps):
    filtered_list = []
    for idx, label in enumerate(labels_list):
        if label["beginframe"] > label["endframe"]:
            continue
        elif label["beginframe"] > DEFAULT_VIDEO_LENGTH * fps:
            continue
        else:
            filtered_list.append(label)
    return filtered_list

def event_type_to_class_num(event_type):
    """ convert a string event type to a class number
        the event type is gotten from the name of the file
        e.g. logPos.txt is type logPos, and has class number 0    retval = -1

    Args: 
       event_type: the string type of the event 

    Returns:
       an integer event type 
    """
    retval = -1 
    if (event_type == 'logPos'):
        retval = 0 
    elif (event_type == 'logNo'):
        retval = 1
    elif (event_type == 'logNeg'):
        retval = 2

    if (retval == -1):
        print ("Warning, no class for event type: %s found" % (event_type))
        
    return retval

def run_thru_events(events_df: pd.DataFrame, counts_df: pd.DataFrame, file_epoch_map_df: pd.DataFrame, fps: int) -> pd.DataFrame:
    """Iterate through the logged events and generate labels for various classes / video files.

    Args:
        events_df: contains the sorted log files events
        counts_df: contains the total frame counts per video from counts.csv
        file_epoch_map_df: contains the filename to epoch time mappings

    Returns:
        pd.DataFrame: contains all of the labels for the start and stop frames for the videos corresponding to each event.
    """

    file_epoch_map_df = _add_video_end_info(file_epoch_map_df, counts_df, fps)
    events_df = _add_event_end_info(events_df, counts_df, fps)
    labels_list = []
    for index, row in events_df.iterrows():
        label = {
            "filename" : None,
            "class" : None,
            "beginframe" : None,
            "endframe" : None
        }
        starting_video = _find_latest_video(row["ts"], file_epoch_map_df)
        starting_video_info = file_epoch_map_df[file_epoch_map_df["filename"] == starting_video]
        starting_video_ts = int(starting_video_info["epoch_ts"])
        starting_video_length = int(starting_video_info["length"])
        starting_video_end_ts = int(starting_video_info["end_epoch_ts"])
        event_ts = int(datetime.strptime(row["ts"], LOG_TIME_FORMAT).timestamp())
        event_end_ts = int(datetime.strptime(row["end_ts"], LOG_TIME_FORMAT).timestamp())

        label["filename"] = starting_video
        label["class"] = event_type_to_class_num(row["event_type"])
        label["beginframe"] = int(event_ts - starting_video_ts) * fps

        if starting_video_end_ts - event_ts < 0: # means that there are logged events where there are no videos recorded
            print(f"Missing videos after {starting_video}, {event_ts}, {starting_video_end_ts}, {starting_video_end_ts - event_ts}")
            continue

        if (event_end_ts > starting_video_end_ts):
            label["endframe"] = int(counts_df[counts_df["filename"]==starting_video]["frames"].item()) # to buffer for rounding errors (make sure no frame out of bounds)
            leftover_seconds = event_end_ts - starting_video_end_ts
            video_index = starting_video_info.index[0] + 1

            while leftover_seconds > 0:
                overflowing_label = {
                    "filename" : None,
                    "class" : None,
                    "beginframe" : None,
                    "endframe" : None
                }
                if video_index >= len(file_epoch_map_df):
                    break

                overflowing_label["filename"] = file_epoch_map_df.iloc[video_index]["filename"]

                overflowing_label["class"] = event_type_to_class_num(row["event_type"])
                overflowing_label["beginframe"] = min(4, int(leftover_seconds * fps)) # incase leftover is less than the 4 frame buffer
                if leftover_seconds < file_epoch_map_df.iloc[video_index]["length"]: # if leftover event spans many videos
                    overflowing_label["endframe"] = min(int(leftover_seconds * fps), int(counts_df[counts_df["filename"]==overflowing_label["filename"]]["frames"].item()))
                    leftover_seconds = 0
                else:
                    overflowing_label["endframe"] = int(counts_df[counts_df["filename"]==overflowing_label["filename"]]["frames"].item())
                    leftover_seconds -= file_epoch_map_df.iloc[video_index]["length"]

                labels_list.append(overflowing_label)
                video_index+=1
        else:
            label["endframe"] = int(event_end_ts - starting_video_ts) * fps

        labels_list.append(label)
    labels_list = _filter_events(labels_list, fps)
    return pd.DataFrame(labels_list).sort_values(by="filename")




@app.command()
def run(
    fps: int = typer.Argument(
        24, help = "Need to specify fps for video since there's no metadata."
    ),
    files_dir: pathlib.Path = typer.Argument(
        ..., help = "Must contain videos, logNeg.txt, logPos.txt, logNo.txt"
    ),
    output_dir: pathlib.Path = typer.Argument(
        ..., help = "Path to output dataset.csv"
    )
):
    events_df = parse_logs(files_dir)
    counts_df = parse_frame_counts(files_dir)
    file_epoch_map_df = map_file_names_to_epoch(counts_df)
    labels = run_thru_events(events_df, counts_df, file_epoch_map_df, fps)
    labels.to_csv(output_dir / "dataset.csv", index=False)

if __name__ == "__main__":
    app()

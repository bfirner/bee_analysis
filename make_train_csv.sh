#! /bin/bash

# (c) 2021-2023 Bernhard Firner.
# See the LICENSE file for more information

# Create a train.csv file for a video dataset.
# csv columns are:
#     file, label, begin frame, end frame
# The 'begin' and 'end' columns are in frames.

# Usage: make_train_csv.sh <log directory> <frames excluded around transitions> <fps of video>

# The target directory
target=$1
exclusion_frames=$2
fps=$3

if [[ "" = "$exclusion_frames" ]]; then
    exclusion_frames=15
fi

if [[ "" = "$fps" ]]; then
    fps=30
fi

# File in the target directory will be:
#     'logNeg.txt' 'logNo.txt' 'logPos.txt'
# Corresponding to negative, positive, and no field.
# Contents of each file will be time in this format:
#     YYYYMMDD_HHMMSS
# Each time indicates the beginning of that phase.
#
# Video file names are in this format:
#     YYYY-MM-DD HH:MM:SS.<milliseconds please ignore>.h264
# To convert video file names use the date command as such:
#     date -d "2021-07-28 14:02:06" +%Y%M%d_%H%M%S

# Using 'grep .' to get all nonblank lines
readarray -t all_state_switches < <( grep --no-filename . $target/*.txt |sed "s/[^0-9_]//g" | sort -n )

# Print out the csv header
echo "file, class, begin frame, end frame"

# A function to parse the time string from the log files.
function switch_string_to_time () {
    local swstr=$(echo $1 | sed 's/\(....\)\(..\)\(..\)_\(..\)\(..\)\(..\)/\1-\2-\3 \4:\5:\6/g')
    local swseconds=$(date -d "$swstr" +%s)
    echo $swseconds
}

# A function to check the log files to determine the label associate with a timestamp
# Arguments are the timestamp string and the target directory with the logs
function switch_label () {
    local state_file=$(grep $1 $2/*.txt -H)
    state_file=${state_file##$2/}
    state_file=${state_file%%.txt*}
    local label=""
    if [[ $state_file == "logNeg" ]]; then
        label=1
    elif [[ $state_file == "logNo" ]]; then
        label=2
    elif [[ $state_file == "logPos" ]]; then
        label=3
    fi
    echo $label
}

for video in $target/*.h264; do
    # Reformat the video date to the same date format as in the magnetic switching logs
    v=${video##$target/}
    # Get the times in seconds from 1970-01-01 00:00:00 UTC
    v=${v%%.*.h264}
    vdate=$(date -d "$v" +%Y%m%d_%H%M%S)
    vseconds=$(date -d "$v" +%s)
    # Get the end time of the video
    # Be aware that using ffprobe to get the number of frames may be rather slow
    vframes=$(ffprobe "$video" -select_streams v:0  -count_frames -show_entries stream=nb_read_frames | grep nb_read_frames | cut -d"=" -f2)
    vseconds_end=$((vseconds + vframes * fps))

    # Find the time where we need a label
    # Always skip the first second of video to allow exposure to adjust.
    # Also enforce a similar exclusion between any magnet state change
    next_frame=$((fps + 1))
    next_seconds=$((vseconds + next_frame * fps))

    # Now loop through the logs to find the first switch that occurs before the time that needs a
    # label
    prev_label=""
    prev_switch=""
    prev_sw_frame="$next_frame"

    # Advance to the first transition after the start of the video
    cur_switch_index=0
    while [[ $cur_switch_index -lt ${#all_state_switches[@]} ]] && [[ $prev_sw_frame -lt $vframes ]]; do
        switch=${all_state_switches[$cur_switch_index]}
        swseconds=$(switch_string_to_time $switch)
        swframe=$(echo "($swseconds - $vseconds) * $fps" | bc)
        label=$(switch_label $switch $target)
        # Do we have a label for this time segment?
        if [[ $prev_sw_frame -le $next_frame ]] && [[ $swframe -ge $next_frame ]]; then
            # Print out a row as long as there was a previous transition label
            if [[ "" != $prev_switch ]]; then

                # Apply an exclusion period around each transition
                end_frame=$((swframe - 1 - exclusion_frames))
                echo "$video, $prev_label, $next_frame, $end_frame"
                # The next segment to label begins with this switch time plush the exclusion window
                next_frame=$((swframe + exclusion_frames))
            fi
        fi
        prev_label="$label"
        prev_switch="$switch"
        prev_sw_frame="$swframe"

        # Advance the switch index
        cur_switch_index=$((cur_switch_index + 1))
    done

    # Check if we exited the loop because we ran out of transitions. If that's the case, then all
    # the rest of the frames belong to the last label
    if [[ $cur_switch_index -eq ${#all_state_switches[@]} ]]; then
        echo "$video, $prev_label, $next_frame, $((vframes - 1))"
    fi
done

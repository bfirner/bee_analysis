#! /bin/bash

# Create a train.csv file for a video dataset.
# csv columns are:
#     file, label, begin, end
# The 'begin' and 'end' columns are in frames.

# The target directory
target=$1

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

#declare -a all_state_switches
#readarray -t all_state_switches < <($(cat $target/*.txt | sort -n))
#all_state_switches=($(cat $target/*.txt | sort -n))
readarray -t all_state_switches < <( cat $target/*.txt | sort -n )

#declare -p all_state_switches

#echo -e "all switches are:\n${all_state_switches[@]}\n\n"

for video in $target/*.h264; do
    # Reformat the video date to the same date format as in the magnetic switching logs
    v=${video##$target/}
    #echo "Checking video $v"
    v=${v%%.*.h264}
    vdate=$(date -d "$v" +%Y%m%d_%H%M%S)
    vseconds=$(date -d "$v" +%s)
    # Get the times in seconds from 1970-01-01 00:00:00 UTC
    #echo "Checking for $vdate"
    # Now loop through the logs to find which switch occurs before this video
    # It is possible that the video has two different states.
    # This could be something more sophisticated than a bash script, this is rather ugly.
    prev_switch=""
    prev_seconds=""
    next_switch=""
    next_seconds=""
    for switch in ${all_state_switches[@]}; do
        swstr=$(echo $switch | sed 's/\(....\)\(..\)\(..\)_\(..\)\(..\)\(..\)/\1-\2-\3 \4:\5:\6/g')
        swseconds=$(date -d "$swstr" +%s)
        if [[ "$switch" < "$vdate" ]]; then
            prev_switch=$switch
            prev_seconds=$swseconds
        else
            next_switch=$switch
            next_seconds=$swseconds
            break
        fi
    done
    # The bee video files are 3fps.
    # Always skip the first second of video to allow exposure to adjust.
    # Also enforce a one minute exclusion between any magnet state change
    first_frame=4
    if [[ "1" == $(echo "$vseconds < ($prev_seconds + 60)") ]]; then
        first_frame=$(echo "3 * ($prev_seconds + 60 - $vseconds)" | bc)
    fi
    # This is to prevent the video from running into the next transition. The video may not actually
    # have these many frames. Don't worry if this is negative, we'll skip it in the dataprep step.
    if [[ "" == $next_switch ]]; then
        # Use the entire video. This is just an arbitrary large number.
        last_frame=1000000000
    else
        last_frame=$(echo "($next_seconds - $vseconds - 60) * 3" | bc)
    fi
    #echo -e "$prev_switch $next_switch \n \n"
    state_file=$(grep $prev_switch $target/*.txt -H)
    state_file=${state_file##$target/}
    state_file=${state_file%%.txt*}
    label=""
    if [[ $state_file == "logNeg" ]]; then
        label=1
    elif [[ $state_file == "logNo" ]]; then
        label=2
    elif [[ $state_file == "logPos" ]]; then
        label=3
    fi
    echo "$video, $label, $first_frame, $last_frame"
done

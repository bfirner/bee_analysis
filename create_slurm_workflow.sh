#!/usr/bin/bash

# (c) 2023 Bernhard Firner.
# See the LICENSE file for more information

# This bash script will create a workflow using slurm commands, as described in this document:
  # https://hpc.nih.gov/docs/job_dependencies.html

# Arguments are
# create_slurm_workflow.sh <experiment name> <data dir or existing csv> [code path] [folds]
# All outputs will be placed in a folder named "<experiment name>"
# The second argument must be either a directory to run `make_train_csv.sh` or an existing csv file,
# which will be automatically detected
# If the third argument is not provided then it will be assumed that analysis code is in the current
# directory
# If the number of folds is not provided then will use the number 10

# A workflow consists of the following steps:
# 1. Run `make_train_csv.sh` to create a csv file with meta-information for dataset prepartion
#    Alternatively, if a training set is already provided skip this step.
# 2. run `make_validation_training.py` to create files for n-fold cross validation
#   1. n smaller dataset csvs
#   2. scripts to run data preparation using those csv files
#   3. a slurm script to run data preparation using those scripts
#   4. n scripts to run training
#   5. a slurm script to run training
# 3. Launch dataprep (n times)
# 4. Launch training (n times)
# 5. Plot the training results (TODO)

# Prints the given command ($3) into the given file name ($1) and adds that command to the end of
# the $slurmfile with the jobid variable in ($2). Arguments for the sbatch command may be provided
# ($4). If given file dependencies ($5), then they will be checked in the script printed into the
# given file name.
# If given job dependencies ($6), will add those strings as an "afterany" dependency in the slurm
# command in $slurmfile.
emit_sbatch_script () {
    local sname=$1
    local jid=$2
    local command="$3"
    local sb_args="$4"
    local file_depends="$5"
    local jid_depends="$6"

    # Create the script
    echo -e "#!/usr/bin/bash -l\n\n" > $sname
    echo -e "# automatically created by create_slurm_workflow.sh\n\n" >> $sname
    echo -e "#SBATCH --job-name=${wfname}-$sname" >> $sname
    echo -e "#SBATCH --output=${wfname}-$sname.log" >> $sname
    # Put a check for an expected file in the script
    if [[ $# -ge 5 ]]; then
        # Check for the given file before executing the script
        echo "if [[ ! -e $file_depends ]]; then" >> $sname
        echo "    echo \"Missing expected input $file_depends\"" >> $sname
        echo "    touch \"${sname}.failure\"" >> $sname
        echo '    exit 2' >> $sname
        echo -e 'fi\n' >> $sname
    fi

    # Go to the output directory
    echo "cd \"$wfpath\"" >> $sname

    # Run the command
    echo -e "succ=\$($command)\n" >> $sname

    # Test the command success within the script and create a fail or succ file.
    echo "if [[ \$? -eq 0 ]]; then" >> $sname
    echo "    touch \"${sname}.success\"" >> $sname
    echo "    exit 0" >> $sname
    echo "else" >> $sname
    echo "    touch \"${sname}.failure\"" >> $sname
    echo "    exit 1" >> $sname
    echo "fi" >> $sname

    # Print the command, with job dependencies if there are any, into the slurmfile
    if [[ $# -ge 6 ]]; then
        echo -e "$jid=\$($sb --dependency=afterany:\$$jid_depends $sb_args $sname)\n" >> $slurmfile
    else
        echo -e "$jid=\$($sb $sb_args $sname)\n" >> $slurmfile
    fi
    echo -e "echo Launched job \$$jid\n" >> $slurmfile
    chmod 755 $sname
}

# Copies the template script ($3) into the given file name ($1) and adds that command to the end of
# the $slurmfile with the jobid variable in ($2). Arguments for the sbatch command may be provided
# ($4).
# If given job dependencies ($5), will add those strings as an "afterany" dependency in the slurm
# command in $slurmfile.
# The path to the template should be absolute, the script name should not contain a directory.
# All other arguments will be treated as string key, value pairs that will be replaced in the
# template file. E.g. "BIN_PATH /path/to/a/directory" would replace BIN_PATH with the given string.
emit_sbatch_from_sh () {
    local sname=$1
    local jid=$2
    local template="$3"
    local sb_args="$4"
    local jid_depends="$5"

    # Copy the template file into the specified name
    cp "$template" "$sname"

    # Print the command, with job dependencies if there are any, into the slurmfile
    if [[ $# -ge 5 ]] && [[ ! -z $jid_depends ]]; then
        echo -e "$jid=\$($sb --dependency=afterany:\$$jid_depends $sb_args $sname)\n" >> $slurmfile
    else
        echo -e "$jid=\$($sb $sb_args $sname)\n" >> $slurmfile
    fi

    # We could use an array expansion instead of shifts, but shifts are easy to understand and bash
    # array are prone to errors with string expansions when there are spaces.
    shift 5
    while [[ $# -gt 0 ]]; do
        local key=$1
        local value=$2
        shift 2
        # Escape the value strings, who knows what is inside.
        # Note that this sed expression will not work with strings containing ":" characters
        local value="$(printf "%q" "$value")"
        sed -i "s:$key:$value:g" "$sname"
    done
    echo -e "echo Launched job \$$jid\n" >> $slurmfile

    # Make the script executable
    chmod 755 $sname
}


# Print usage and quit.
usage() {
    echo "Usage: $0 -n <experiment name> -d <data dir or existing csv> [-p <code path>] [-f <folds>]"
    exit 2
}

# Defaults
# If no code path was provided then assume the current directory
codepath=$(realpath "$(pwd)")
# Default to 10 folds
folds=10
while getopts "n:d:p:f:?h" opt; do
    case $opt in
        n) wfname="$OPTARG" ;;
        d) datapath=$(realpath "${OPTARG}") ;;
        p) codepath=$(realpath "${OPTARG}") ;;
        f) folds=$OPTARG ;;
        h|?) usage ;;
    esac
done

if [[ -z $wfname ]] || [[ -z $datapath ]]; then
    echo "-n and -d are required arguments."
    usage
fi

# Check the code path
if [[ ! -d ${codepath} ]]; then
    echo "Code path ${codepath} is not an existing directory."
    exit 2
fi

# Wrapped sbatch command
sb="${codepath}/sbatch_wrapper.sh"

# Create an output directory and change into it
mkdir -p "${wfname}"
wfpath=$(realpath "${wfname}")
cd "${wfpath}"
# Remove anything in the directory that could disturb the workflow
# TODO Nothing is using these files yet, but they could be useful in the future.
statfiles=(/*.success)
if [[ -f ${statfiles[0]} ]]; then
    rm *.success
fi
statfiles=(/*.failure)
if [[ -f ${statfiles[0]} ]]; then
    rm *.failure
fi

slurmfile="${wfpath}/run_workflow.sh"
echo -e "#!/usr/bin/bash\n" > $slurmfile
echo "# Run this command as a regular bash shell" >> $slurmfile
echo -e "\n# automatically created by create_slurm_workflow.sh\n\n" >> $slurmfile
chmod 755 $slurmfile

# Check if the data path exists
if [[ ! -e ${datapath} ]]; then
    echo "Data path ${datapath} does not exist."
    exit 2
fi

# If this is a directory then run `make_train_csv.sh` to make datacsv. Otherwise just copy it.
datacsv="${wfpath}/all_data.csv"
if [[ -d ${datapath} ]]; then
    echo "Will run make_train_csv.sh on path ${datapath}"
    # Create a script that runs make_train_csv.sh on ${datapath}
    # Append that script to ${slurmfile}
    emit_sbatch_script \
        "create_initial_csv.sh" \
        "create_csv_jid" \
        "bash \"${codepath}/make_train_csv.sh\" \"${datapath}\" >> \"${datacsv}\""
    csv_jid="create_csv_jid"
elif [[ -f ${datapath} ]]; then
    echo "Using ${datapath} as data csv file."
    cp "${datapath}" "${datacsv}"
    csv_jid=""
fi

# Now run the make_validation_training.py script
mvtcmd="python3 \"${codepath}/make_validation_training.py\" "
mvtcmd+="--datacsv \"${datacsv}\" --k ${folds} --only_split"

if [[ -z $csv_jid ]]; then
    emit_sbatch_script \
        "mvt_scripts.sh" \
        "mvt_scripts_jid" \
        "${mvtcmd}" \
        "" \
        "${datacsv}"
else
    emit_sbatch_script \
        "mvt_scripts.sh" \
        "mvt_scripts_jid" \
        "${mvtcmd}" \
        "" \
        "${datacsv}" \
        "$csv_jid"
fi

# Create the dataprep task
emit_sbatch_from_sh \
    "dataprep.sh" \
    "dataprep_jid" \
    "${codepath}/slurm_templates/dataprep_roaches.sh" \
    "" \
    "mvt_scripts_jid" \
    "BIN_PATH" "${codepath}" \
    "CSV_BASE" "${datacsv%%.csv}" \
    "TAR_BASE" "${datacsv%%.csv}" \
    "MAX_FOLD" "${folds}" \
    "LOG_FILE" "${wfpath}/dataprep.log" \
    "OUT_PATH" "${wfpath}" \
    "WF_NAME" "${wfname}"

# Create the training task
emit_sbatch_from_sh \
    "train.sh" \
    "train_jid" \
    "${codepath}/slurm_templates/train.sh" \
    "-G 1" \
    "dataprep_jid" \
    "BIN_PATH" "${codepath}" \
    "CHECKPOINT" "${wfname}.ckpt" \
    "BASE_DATA" "${datacsv%%.csv}" \
    "MAX_FOLD" "${folds}" \
    "LOG_FILE" "${wfpath}/training.log" \
    "WF_NAME" "${wfname}"

# TODO
# Create the annotation task
# Plot results from the different training runs


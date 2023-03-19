#!/usr/bin/bash

# (c) 2023 Bernhard Firner.
# See the LICENSE file for more information

# This script wraps the regular sbatch command, as suggested in
  # https://hpc.nih.gov/docs/job_dependencies.html

sb_cmd=$(which sbatch)
sbresult="$($sb_cmd "$@")"

if [[ "$sbresult" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    exit 0
else
    echo "Command: 'sbatch $@' failed"
    exit 2
fi

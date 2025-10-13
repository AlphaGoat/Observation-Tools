#!/bin/bash
# Loop through different configurations for dataset generation
# and call the Python script with those configurations.
counter=0
for i in /data/satsim_configs/*.json; do
    satsim run --jobs 3 --device 0 --mode eager --output_dir /data/satsim_output/run_$counter $i
    counter=$((counter + 1))
done
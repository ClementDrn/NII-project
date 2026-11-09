#!/bin/bash
# Run it inside the directory containing job_* directories.

# Enable nullglob to ensure empty directories yield a count of 0 instead of 1
shopt -s nullglob

mkdir -p job_000

for job in job_*; do
    # Skip non-directories and the target job_000 directory
    if [ -d "$job" ] && [ "$job" != "job_000" ]; then
        
        for sce in "$job"/sce_*; do
            if [ -d "$sce" ]; then
                sce_name=$(basename "$sce")
                target_sce="job_000/$sce_name"
                
                mkdir -p "$target_sce"
                
                for run in "$sce"/run_*; do
                    if [ -d "$run" ]; then
                        # Store existing runs in an array to count them
                        existing_runs=("$target_sce"/run_*)
                        next_num=${#existing_runs[@]}
                        
                        # Format the next run name with zero-padding (e.g., run_001)
                        printf -v new_run_name "run_%03d" "$next_num"
                        
                        # Move and rename the directory
                        mv "$run" "$target_sce/$new_run_name"
                    fi
                done
            fi
        done
    fi
done

# Disable nullglob to restore default behavior
shopt -u nullglob

# Clean up empty directories
find job_* -depth -type d -empty -delete

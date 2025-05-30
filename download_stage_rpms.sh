#!/bin/bash
set -euo pipefail

# Example default base dir (optional fallback)
DEFAULT_BASE_DEST_DIR="$(pwd)"

# Define stages: base_dest_dir, volume_name, and package list
# Format: "base_dest_dir:volume_name:package1 package2 package3"
STAGES=(
  "containers/opencv:opencv-container-dnf-packages-build:python3.12 python3.12-devel gcc gcc-c++ cmake pkg-config opencv opencv-devel curl wget unzip libgomp glib2-devel libstdc++-devel atlas-devel blas-devel lapack-devel libstdc++ atlas blas lapack"
)

# Function to run a podman download stage
download_stage() {
  local base_dest_dir="$1"
  local volume_name="$2"
  local packages="$3"
  local dest_dir="$base_dest_dir/$volume_name"
  local log_file="$base_dest_dir/${volume_name}.log"

  echo "=== [START] Stage: $volume_name ===" | tee "$log_file"

  # Create volume if not exists
  if ! podman volume inspect "$volume_name" &>/dev/null; then
    echo "Creating volume: $volume_name" | tee -a "$log_file"
    podman volume create "$volume_name" >>"$log_file" 2>&1
  fi

  echo "Running container to download packages..." | tee -a "$log_file"
  podman run --rm -v "$volume_name":/var/cache/libdnf5 quay.io/fedora/fedora:latest bash -c "\
dnf upgrade --downloadonly -y && \
dnf install --downloadonly -y $packages" >>"$log_file" 2>&1

  local volume_path
  volume_path=$(podman volume inspect "$volume_name" -f '{{.Mountpoint}}')

  mkdir -p "$dest_dir"
  echo "Copying RPMs from $volume_path to $dest_dir" | tee -a "$log_file"
  cp -rv "$volume_path"/* "$dest_dir/" >>"$log_file" 2>&1 || echo "No RPMs found to copy." | tee -a "$log_file"

  echo "=== [DONE] Stage: $volume_name ===" | tee -a "$log_file"
}

# Loop through stages and run them in parallel
pids=()
for stage in "${STAGES[@]}"; do
  # Split stage into three parts:
  # base_dest_dir : volume_name : packages
  IFS=':' read -r base_dest_dir volume_name packages <<< "$stage"

  # Use default if base_dest_dir is empty (optional)
  base_dest_dir="${base_dest_dir:-$DEFAULT_BASE_DEST_DIR}"

  # Kick off each stage in the background
  download_stage "$base_dest_dir" "$volume_name" "$packages" &
  pids+=($!)
done

# Wait for all parallel jobs to complete
echo "Waiting for all stages to complete..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "✅ All parallel stages completed."

#!/bin/bash
set -euo pipefail

# Default base dir fallback
DEFAULT_BASE_DEST_DIR="$(pwd)"

# Stages format:
# "base_dest_dir:volume_name:requirements_path:podman_image"
STAGES=(
  "containers/opencv:pip-venv-cache:containers/opencv/requirements.txt:verification_app_opencv"
  # Add more stages as needed
)

download_stage() {
  local base_dest_dir="$1"
  local volume_name="$2"
  local requirements_path="$3"
  local podman_image="$4"
  local dest_dir="$base_dest_dir/$volume_name"
  local log_file="$base_dest_dir/${volume_name}.log"

  echo "=== [START] Stage: $volume_name ===" | tee "$log_file"

  # Create volume if it doesn't exist
  if ! podman volume inspect "$volume_name" &>/dev/null; then
    echo "Creating volume: $volume_name" | tee -a "$log_file"
    podman volume create "$volume_name" >>"$log_file" 2>&1
  fi

  # Get volume mount point
  local volume_path
  volume_path=$(podman volume inspect "$volume_name" -f '{{.Mountpoint}}')

  # Copy requirements.txt into volume (optional, for consistent context)
  cp "$requirements_path" "$volume_path/requirements.txt"

  echo "Running container to download pip packages..." | tee -a "$log_file"
  podman run --rm -v "$volume_name":/opt/venv -w /opt/venv "$podman_image" bash -c "\
python3.12 -m venv /opt/venv && \
source /opt/venv/bin/activate && \
python --version && \
python -m pip install --upgrade pip setuptools wheel && \
pip install --no-cache-dir -r requirements.txt" >>"$log_file" 2>&1

  mkdir -p "$dest_dir"
  echo "Copying downloaded packages from volume to $dest_dir" | tee -a "$log_file"
  cp -rv "$volume_path"/* "$dest_dir/" >>"$log_file" 2>&1 || echo "No packages found to copy." | tee -a "$log_file"

  echo "=== [DONE] Stage: $volume_name ===" | tee -a "$log_file"
}

pids=()
for stage in "${STAGES[@]}"; do
  IFS=':' read -r base_dest_dir volume_name requirements_path podman_image <<< "$stage"
  base_dest_dir="${base_dest_dir:-$DEFAULT_BASE_DEST_DIR}"

  download_stage "$base_dest_dir" "$volume_name" "$requirements_path" "$podman_image" &
  pids+=($!)
done

echo "Waiting for all stages to complete..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "✅ All parallel stages completed."

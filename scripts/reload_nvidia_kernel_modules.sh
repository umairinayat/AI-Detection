#!/usr/bin/env bash
# Reload NVIDIA kernel modules when userspace driver was upgraded but the
# running kernel still had an older module (nvidia-smi: "Driver/library version mismatch").
# Requires root. Stops briefly; reload any GPU workloads after.

set -euo pipefail
if [[ "${EUID:-0}" -ne 0 ]]; then
  echo "Run with: sudo $0" >&2
  exit 1
fi

modprobe -r nvidia_uvm 2>/dev/null || true
modprobe -r nvidia_drm 2>/dev/null || true
modprobe -r nvidia_modeset 2>/dev/null || true
modprobe -r nvidia 2>/dev/null || true

modprobe nvidia
modprobe nvidia_modeset
modprobe nvidia_drm
modprobe nvidia_uvm

echo "Loaded module version:"
cat /proc/driver/nvidia/version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

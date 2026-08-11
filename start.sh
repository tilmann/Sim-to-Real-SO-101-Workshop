#!/bin/bash
# Launches the teleop/simulation container. Run from anywhere; it cd's to the repo root itself.
set -e
cd "$(dirname "$0")"

# Allow local X11 clients (the container) to connect to this session's display.
xhost +local: >/dev/null

# Resolve the current session's X auth cookie. On XWayland/GNOME this lives under
# /run/user/<uid>/.mutter-Xwaylandauth.<random>, which changes every login, so it
# can't be hardcoded. Falls back to ~/.Xauthority if that's a real file, or skips
# the mount entirely (xhost +local: above is sufficient on its own).
XAUTH_MOUNT=()
XAUTH_FILE=$(ls /run/user/"$(id -u)"/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
if [ -z "$XAUTH_FILE" ] && [ -f "$HOME/.Xauthority" ]; then
  XAUTH_FILE="$HOME/.Xauthority"
fi
if [ -n "$XAUTH_FILE" ]; then
  XAUTH_MOUNT=(-v "$XAUTH_FILE:/root/.Xauthority:ro")
fi

mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache} \
         ~/docker/isaac-sim/logs ~/docker/isaac-sim/data ~/docker/isaac-sim/documents

docker run --name teleop -it --privileged --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v /dev:/dev \
   -v /run/udev:/run/udev:ro \
   "${XAUTH_MOUNT[@]}" \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   -v ~/.cache/huggingface/lerobot/calibration:/root/.cache/huggingface/lerobot/calibration \
   -v ./docker/env:/root/env \
   -v "$(pwd)/source":/workspace/Sim-to-Real-SO-101-Workshop/source \
   -v "$(pwd)/outputs":/workspace/Sim-to-Real-SO-101-Workshop/outputs \
   -v "$(pwd)/datasets":/workspace/Sim-to-Real-SO-101-Workshop/datasets \
   -v "$(pwd)/docker/real/scripts":/workspace/Sim-to-Real-SO-101-Workshop/docker/real/scripts \
   teleop-docker:latest

#!/bin/bash
BASE_IMAGE_TAG=noetic
#-develop

IMAGE_NAME=learning-safety-margin

SERVE_REMOTE=false
REMOTE_SSH_PORT=2280

HELP_MESSAGE="Usage: ./build-server.sh [-b|--branch branch] [-r] [-v] [-s]
Build a Docker container for remote development and/or running unittests.
Options:
  --base-tag               The tag of ros2-control-libraries image.

  -r, --rebuild            Rebuild the image with no cache.

  -v, --verbose            Show all the output of the Docker
                           build process

  -s, --serve              Start the remove development server.

  -h, --help               Show this help message.
"

BUILD_FLAGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-tag) BASE_IMAGE_TAG=$2; shift 2;;
    -r|--rebuild) BUILD_FLAGS+=(--no-cache); shift 1;;
    -v|--verbose) BUILD_FLAGS+=(--progress=plain); shift 1;;
    -s|--serve) SERVE_REMOTE=true ; shift ;;
    -h|--help) echo "${HELP_MESSAGE}"; exit 0;;
    *) echo "Unknown option: $1" >&2; echo "${HELP_MESSAGE}"; exit 1;;
  esac
done

docker pull ghcr.io/aica-technology/ros-control-libraries:"${BASE_IMAGE_TAG}"

BUILD_FLAGS+=(--build-arg BASE_IMAGE_TAG="${BASE_IMAGE_TAG}")
BUILD_FLAGS+=(-t "${IMAGE_NAME}:${BASE_IMAGE_TAG}")

DOCKER_BUILDKIT=1 docker build "${BUILD_FLAGS[@]}" .

if [ "${SERVE_REMOTE}" = true ]; then
  aica-docker server "${IMAGE_NAME}:${BASE_IMAGE_TAG}" -u ros -p "${REMOTE_SSH_PORT}" \
  -v /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data:/home/ros/ros_ws/src/learning_safety_margin/data \
  -p1601:1601 -p1602:1602
fi

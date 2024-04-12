#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "You need to supply CODE_PATH and (optionally) DATA_PATH as specified in the README"
    exit 1
fi

CODE_DIR_PATH=$1

if [ -z "$2" ]
  then
    # run docker image without data path mounting
    docker run --gpus all --detach --shm-size=8gb --name egs-container -t -v $CODE_DIR_PATH/:/exhaustive-grid-search egs-image
    exit 1
fi

DATA_DIR_PATH=$2
docker run --gpus all --detach --shm-size=8gb --name egs-container -t -v $CODE_DIR_PATH/:/exhaustive-grid-search -v $DATA_DIR_PATH/:/data egs-image
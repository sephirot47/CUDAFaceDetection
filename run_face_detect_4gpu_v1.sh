#!/bin/bash

. common.sh
NAME=face_detect_4gpu_v1
make bin/${NAME}.exe
qsub -l cuda job.sh $NAME

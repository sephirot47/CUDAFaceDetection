#!/bin/bash

. common.sh
NAME=face_detect_1gpu
make bin/${NAME}.exe
qsub -l cuda job.sh $NAME

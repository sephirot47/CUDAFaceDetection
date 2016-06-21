#!/bin/bash

. common.sh
NAME=face_detect_seq
make bin/${NAME}.exe
#qsub -l cuda job.sh $NAME
./bin/${NAME}.exe ./images/input.png

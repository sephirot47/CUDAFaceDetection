#!/bin/bash

#scp ./* cuda05@boada.ac.upc.edu:CUDAFaceDetection
echo retrasito
scp src/face_detection.cu Makefile cuda05@boada.ac.upc.edu:CUDAFaceDetection/src

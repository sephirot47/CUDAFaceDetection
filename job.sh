#!/bin/bash
export PATH=/Soft/cuda/7.5.18/bin:$PATH
### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N FaceDetection
# Cambiar el shell
#$ -S /bin/bash

time nvprof --print-summary-per-gpu ./bin/${1}.exe images/input.png

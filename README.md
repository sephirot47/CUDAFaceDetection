# CUDAFaceDetection
CUDA program to locate a given face in a picture.

EJECUCIÓN:

Para ejecutar cada versión X, hay que usar el script run_X.sh, que recibe como parámetro la imagen en la que se quieren detectar caras. 

Los scripts run_X.sh se encargan de detectar si la imagen pasada como argumento existe, la prepararan y hacen submit del job.sh. 

El resultado de cada ejecución queda almacenado en output/result.bmp.

Como ejemplos de uso, en la carpeta images hay varias imágenes para probar la aplicación.

El código de cada versión se encuentra en la carpeta src.

Ejemplo de uso (siempre desde la carpeta "raíz" de la práctica):
    ./run_face_detect_sequential.sh images/img1.png
    ./run_face_detect_4gpu.sh images/img4.png


VER RESULTADO:

Para ver el resultado de la ultima ejecución, usar el siguiente comando:
    convert -scale 30% output/result.bmp output/result_small.bmp && xdg-open output/result_small.bmp	 

# CUDAFaceDetection
CUDA program to locate a given face in a picture.

EJECUCION:

Para ejecutar cada versión X, hay que usar el script run-X.sh, que recibe como parámetro la imagen en la que se quren detectar caras. 

Los script run-X.sh se encargan de detectar si la imágen pasada como argumento existe, prepararla, y hacer submit del job.sh. 

El resultado de cada ejecución queda almacenado en output/result.bmp
Como ejemplos de uso, en la carpeta images hay varias imágenes para probar la aplicación
El código de cada versión se encuentra en la carpeta src.

Ejemplo de uso (siempre desde la carpeta "raiz" de la practica):
	./run_face_detect_sequential.sh images/img1.png
	./run_face_detect_4gpu.sh images/img4.png


VER RESULTADO:

Para ver el resultado de la ultima ejecucion, se ha de usar la siguiente linea de comandos:
	convert -scale 30% output/result.bmp output/result_small.bmp && xdg-open output/result_small.bmp	 

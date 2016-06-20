#!/bin/bash

function getTime()
{
	while [ "$(qstat | grep FaceDe | wc -l)" != "0" ]; do 
		sleep 1
	done
	return "$(cat FaceDetection.e* | grep user | awk '{print $2}' | cut -d"m" -f2)"
}

OUTPUT="times.txt"
echo "__________________________________________" >> $OUTPUT
echo "________________ NEW ROUND _______________" >> $OUTPUT
echo "__________________________________________" >> $OUTPUT

for img in images/*.png
do
	echo "Processing image ${img}..."
	echo "Image: $img" >> $OUTPUT
	
	echo "1gpu..."
	./run_face_detect_1gpu.sh $img
	time=$(getTime)
	echo $time ; echo $time >> $OUTPUT
	
	echo "4gpu..."
	./run_face_detect_4gpu.sh $img
	time=$(getTime)
	echo $time ; echo $time >> $OUTPUT
	
	echo "4gpu pinned..."
	./run_face_detect_4gpu_pin.sh $img
	time=$(getTime)
	echo "___________" >> $OUTPUT
	echo $time ; echo $time >> $OUTPUT
done

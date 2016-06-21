#!/bin/bash

function getTime()
{
	while [ "$(qstat | grep FaceDe | wc -l)" != "0" ]; do 
		sleep 1
	done
	sleep 10
	echo "$(cat FaceDetection.e* | grep user | awk '{print $2}' | cut -d"m" -f2)"
}

OUTPUT="times.txt"
echo "__________________________________________" >> $OUTPUT
echo "________________ NEW ROUND _______________" >> $OUTPUT
echo "__________________________________________" >> $OUTPUT

for i in {1..4}; 
do
for img in images/*.png
do
	echo "Processing image ${img}..."
	echo "Image: $img" >> $OUTPUT
	
	echo "1gpu..." ; echo "1gpu" >> $OUTPUT
	./run_face_detect_1gpu.sh $img
	time="$(getTime)"
	echo $time ; echo $time >> $OUTPUT
	echo "______________" >> $OUTPUT
	
	echo "4gpu v1..." ; echo "4gpu v1" >> $OUTPUT
	./run_face_detect_4gpu_v1.sh $img
	time=$(getTime)
	echo $time ; echo $time >> $OUTPUT
	echo "______________" >> $OUTPUT
	
	echo "4gpu v2..." ; echo "4gpu v2" >> $OUTPUT
	./run_face_detect_4gpu_v2.sh $img
	time=$(getTime)
	echo $time ; echo $time >> $OUTPUT
	echo "______________" >> $OUTPUT
	
	echo "4gpu v2 pinned..." ; echo "4gpu v2 pinned..." >> $OUTPUT
	./run_face_detect_4gpu_pin_v2.sh $img
	time=$(getTime)
	echo $time ; echo $time >> $OUTPUT
	echo "______________" >> $OUTPUT
	echo "****" >> $OUTPUT
done
	echo "***************************" >> $OUTPUT
done

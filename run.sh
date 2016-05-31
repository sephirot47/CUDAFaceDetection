#!/bin/bash

make ultraclean && make 
if [ "$?" != "0" ] ; then
	echo "COMPILATION ERROR!"
	exit 1
fi

INPUT="images/img1.png"
WIDTH=$( file $INPUT | cut -d"," -f2 | cut -d"," -f1 | cut -d"x" -f1 | cut -d" " -f2)
HEIGHT=$( file $INPUT | cut -d"," -f2 | cut -d"," -f1 | cut -d"x" -f2 | cut -d" " -f2)
if [ $HEIGHT -lt 1024 ] && [ $HEIGHT -lt $WIDTH ]; then
	convert -resize x1024 $INPUT images/input.png  # Resize to have at least size of 1024 in its smallest dimension
elif [ $WIDTH -lt 1024 ] && [ $WIDTH -lt $HEIGHT ] ; then
	convert -resize 1024x $INPUT images/input.png  # Resize to have at least size of 1024 in its smallest dimension
else
	cp $INPUT images/input.png
fi

qsub -l cuda job.sh 
while [ "$(qstat | grep cuda05 | wc -l)" != "1" ]; do 
	sleep 1
done
#convert -scale 30% output/result.bmp output/result_small.bmp
display output/result.bmp

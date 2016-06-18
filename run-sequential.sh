#!/bin/bash

make 
if [ "$?" != "0" ] ; then
	echo "COMPILATION ERROR!"
	exit 1
fi

INPUT=$1
if ! [ -f $INPUT ] || [ -z $INPUT ]
then
	echo "ERROR: The image does not exist."
	exit 1
fi 

WIDTH=$( file $INPUT | cut -d"," -f2 | cut -d"," -f1 | cut -d"x" -f1 | cut -d" " -f2)
HEIGHT=$( file $INPUT | cut -d"," -f2 | cut -d"," -f1 | cut -d"x" -f2 | cut -d" " -f2)
if [ $HEIGHT -lt 1024 ] && [ $HEIGHT -lt $WIDTH ]; then
	convert -resize x1024 $INPUT images/input.png  # Resize to have at least size of 1024 in its smallest dimension
elif [ $WIDTH -lt 1024 ] && [ $WIDTH -lt $HEIGHT ] ; then
	convert -resize 1024x $INPUT images/input.png  # Resize to have at least size of 1024 in its smallest dimension
else
	cp $INPUT images/input.png
fi

qsub -l cuda job-sequential.sh 

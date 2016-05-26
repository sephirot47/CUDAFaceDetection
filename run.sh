#!/bin/bash

make ultraclean && make 
if [ "$?" != "0" ] ; then
	echo "COMPILATION ERROR!"
	exit 1
fi

for i in {0..2} 
do
	echo "Executing job $i"
	qsub -l cuda job.sh 
	while [ "$(qstat | grep cuda05 | wc -l)" != "1" ]; do 
		sleep 1; 
	done 
	sleep 2
	convert -scale 30% output/result.bmp output/result_small_${i}.bmp
done

diff output/result_small_0.bmp output/result_small_1.bmp
diff output/result_small_0.bmp output/result_small_2.bmp
diff output/result_small_1.bmp output/result_small_2.bmp

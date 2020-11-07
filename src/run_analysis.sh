#!/bin/bash

#for i in {50..99}
for i in {50..53}
do
	qsub -v "EVENT_NUM=$i" a_sub.sh
done

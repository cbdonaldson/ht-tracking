#!/bin/bash

for i in 9
do
	qsub -q medium -v "EVENT_NUM=$i" sub.sh
done



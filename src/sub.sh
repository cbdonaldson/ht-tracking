#!/bin/bash

export DATA_DIR=/home/cpdonaldson/ht-tracking/storage/
export ITERATIONS=5
export SIZE=80
export T1=4
export T2=0.5
export T_MATCH=7.1
export NN=1
#export EVENT_NUM=$1

cd ~
source ~/venv/bin/activate
#cd ht-tracking/outputs/
#mkdir event_"$EVENT_NUM"
cd ht-tracking/src
mkdir /home/cpdonaldson/ht-tracking/outputs/events/t_match_$T_MATCH
mkdir /home/cpdonaldson/ht-tracking/outputs/events/t_match_$T_MATCH/event_$EVENT_NUM

python3 process_events.py $DATA_DIR $EVENT_NUM $ITERATIONS $SIZE $T1 $T2 $T_MATCH $NN

mv *.p ../outputs/events/t_match_$T_MATCH


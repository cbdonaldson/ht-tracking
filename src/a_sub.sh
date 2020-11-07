#!/bin/bash

export DATA_DIR=/home/cpdonaldson/ht-tracking/outputs/event_$EVENT_NUM/
export T_MATCH=7.1

cd ~
mkdir ht-tracking/efficiencies/event_$EVENT_NUM
source ~/venv/bin/activate
cd ht-tracking/src

python3 analyse_results.py $DATA_DIR $T_MATCH

mv *.p ~/ht-tracking/efficiencies/event_$EVENT_NUM


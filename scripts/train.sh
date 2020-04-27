#!/bin/usr/bash/env bash

idx=3

for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		cd code
		echo "s = $s, t = $t"
		python train.py --data_path ../data/s$s/t$t --epochs 40 --epochs_per_val 5 --save_id s${s}t${t} --device_id $idx --bidirectional
		cd ../
	done
done

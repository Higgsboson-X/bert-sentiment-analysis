#!/bin/usr/bash/env bash

idx=3

mkdir -p logs

echo "+++++++ eval +++++++" > logs/eval

for s in 1 2
do
	for t in 1 2 3 4 5 6 7
	do
		cd code
		echo "s = $s, t = $t"
		python eval.py --data_path ../data/s$s/t$t --save_id s${s}t${t} --device_id $idx >> ../logs/eval
		cd ../
	done
done

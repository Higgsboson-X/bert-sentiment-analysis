#!/bin/usr/bash/env bash

idx=-1

for model in DRG
do
	mkdir -p logs
	echo "model = $model"
	echo "+++++++ eval (non-pretrain) +++++++" > logs/$model
	for s in 1 2
	do
		for t in 1 2 3 4 5 6 7
		do
			cd code
			echo "s = $s, t = $t"
			python eval.py --data_path ../data/outputs/$model/s$s/t$t --save_id s${s}t${t} --device_id $idx >> ../logs/$model
			cd ../
		done
	done
done

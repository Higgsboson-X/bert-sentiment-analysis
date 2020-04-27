#!/bin/usr/bash/env bash

idx=-1

for model in template cross_align DRG dualRL vae maml_cross_align maml_vae
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

for model in cross_align_pretrain vae_pretrain maml_cross_align_pretrain maml_vae_pretrain
do
	echo "model = $model"
	echo "+++++++ eval (pretrain) +++++++" > logs/$model
	for s in 2
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

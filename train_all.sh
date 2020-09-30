#!/bin/bash
for i in {1..22}; do
	python3 cli.py --train --save_dir '.save' --num_epochs 100 --batch_size 32 --lr 0.01 --task $i --memory_size 50 --num_hops 3 --max_clip 40.0 --tenk >> logs/task$i.txt;
done

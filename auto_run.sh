#!/bin/bash

ps_num=1
worker_num=2


ps_num -= 1
worker_num -= 1

for i in `eval echo {0..$ps_num}`
do
  CUDA_VISIBLE_DEVICES='' python ddpg.py --job_name="ps" --workers_num $worker_num --task_index=$i &
done

for i in `eval echo {0..$worker_num}`
do
  CUDA_VISIBLE_DEVICES='' python ddpg.py --job_name="worker" --workers_num $worker_num --task_index=$i &
done


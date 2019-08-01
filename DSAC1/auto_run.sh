#!/bin/bash


env_name="LunarLanderContinuous-v2"
ps_num=1
worker_num=2


for i in `eval echo {0..$((ps_num-1))}`
do
  CUDA_VISIBLE_DEVICES='' python dsac1.py --env_name=$env_name --job_name="ps" --workers_num $worker_num --task_index=$i &
done

for i in `eval echo {0..$(($worker_num-1))}`
do
  CUDA_VISIBLE_DEVICES='' python dsac1.py --env_name=$env_name --job_name="worker" --workers_num $worker_num --task_index=$i &
done


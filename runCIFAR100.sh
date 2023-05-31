#!/bin/bash

python3 ./badge/run.py --model vgg_simple --data CIFAR100 --alg PoincareKmeans > ./badge/output/latest_runs/CIFAR100_vgg_simple_PoincareKmeans_2500.txt
python3 ./badge/run.py --model vgg_simple --data CIFAR100 --alg rand > ./badge/output/latest_runs/CIFAR100_vgg_simple_rand_2500.txt
python3 ./badge/run.py --model vgg_simple --data CIFAR100 --alg PoincareKmeansUncertainty > ./badge/output/latest_runs/CIFAR100_vgg_simple_PoincareKmeansUncertainty_2500.txt
python3 ./badge/run.py --model vgg_simple --data CIFAR100 --alg meal > ./badge/output/latest_runs/CIFAR100_vgg_simple_meal_2500.txt
python3 ./badge/run.py --model vgg_simple --data CIFAR100 --alg badge > ./badge/output/latest_runs/CIFAR100_vgg_simple_badge_2500.txt
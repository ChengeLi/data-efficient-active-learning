#!/bin/bash
python3 ./badge/run.py --model vgg_simple --data CalTech256 --alg rand > ./badge/output/latest_runs/CalTech256_vgg_simple_rand_1192.txt
python3 ./badge/run.py --model vgg_simple --data CalTech256 --alg PoincareKmeans > ./badge/output/latest_runs/CalTech256_vgg_simple_PoincareKmeans_1192.txt
python3 ./badge/run.py --model vgg_simple --data CalTech256 --alg PoincareKmeansUncertainty > ./badge/output/latest_runs/CalTech256_vgg_simple_PoincareKmeansUncertainty_1192.txt
python3 ./badge/run.py --model vgg_simple --data CalTech256 --alg badge > ./badge/output/latest_runs/CalTech256_vgg_simple_badge_1192.txt
python3 ./badge/run.py --model vgg_simple --data CalTech256 --alg meal > ./badge/output/latest_runs/CalTech256_vgg_simple_meal_1192.txt

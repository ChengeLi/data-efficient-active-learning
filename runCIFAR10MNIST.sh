#!/bin/bash
python3 ./badge/run.py --model net00 --data MNIST --alg rand > ./badge/output/latest_runs/MNIST_net00_rand_100.txt
python3 ./badge/run.py --model net00 --data MNIST --alg badge > ./badge/output/latest_runs/MNIST_net00_badge_100.txt
python3 ./badge/run.py --model net00 --data MNIST --alg PoincareKmeans > ./badge/output/latest_runs/MNIST_net00_PoincareKmeans_100.txt
python3 ./badge/run.py --model net00 --data MNIST --alg PoincareKmeansUncertainty > ./badge/output/latest_runs/MNIST_net00_PoincareKmeansUncertainty_100.txt
python3 ./badge/run.py --model net00 --data MNIST --alg meal > ./badge/output/latest_runs/MNIST_net00_meal_100.txt
python3 ./badge/run.py --model net00 --data CIFAR10 --alg rand > ./badge/output/latest_runs/CIFAR10_net00_rand_100.txt
python3 ./badge/run.py --model net00 --data CIFAR10 --alg badge > ./badge/output/latest_runs/CIFAR10_net00_badge_100.txt
python3 ./badge/run.py --model net00 --data CIFAR10 --alg PoincareKmeans > ./badge/output/latest_runs/CIFAR10_net00_PoincareKmeans_100.txt
python3 ./badge/run.py --model net00 --data CIFAR10 --alg PoincareKmeansUncertainty > ./badge/output/latest_runs/CIFAR10_net00_PoincareKmeansUncertainty_100.txt
python3 ./badge/run.py --model net00 --data CIFAR10 --alg meal > ./badge/output/latest_runs/CIFAR10_net00_meal_100.txt

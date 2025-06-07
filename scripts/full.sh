#!/bin/bash

python main.py -b 128 -e 100 -l 5e-4 -t 1000 -d 2048 -a none -m train

python main.py -b 128 -e 100 -l 5e-4 -t 1000 -d 2048 -a triplet -m train

python main.py -b 128 -e 100 -l 5e-4 -t 1000 -d 2048 -a none -m sample

python main.py -b 128 -e 100 -l 5e-4 -t 1000 -d 2048 -a triplet -m sample

python main.py -b 128 -e 100 -l 5e-4 -t 1000 -d 2048 -a none -m stats

python main.py -b 128 -e 100 -l 5e-4 -t 1000 -d 2048 -a triplet -m stats

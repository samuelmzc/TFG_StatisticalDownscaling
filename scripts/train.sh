#!/bin/bash

python main.py --island gc -b 128 -e 50 -l 5e-4 -t 1000 -d 1024 -a none -m train

python main.py --island gc -b 128 -e 50 -l 5e-4 -t 1000 -d 1024 -a triplet -m train

python main.py --island lp -b 128 -e 50 -l 5e-4 -t 1000 -d 1024 -a none -m train

python main.py --island lp -b 128 -e 50 -l 5e-4 -t 1000 -d 1024 -a triplet -m train
# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

This script is used to sample data from the raw dataset

python sample.py s_path t_path prob

"""

import argparse
import sys
import random

s_path = sys.argv[1]
t_path = sys.argv[2]
prob = float(sys.argv[3])

with open(t_path,'wb') as f:
    for line in open(s_path,'rb'):
        if random.random() < prob:
            f.write(line)

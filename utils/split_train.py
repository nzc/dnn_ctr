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
tr_path = sys.argv[2]
te_path = sys.argv[3]
prob = float(sys.argv[4])

with open(tr_path,'wb') as fr:
    with open(te_path,'wb') as fe:
        for line in open(s_path,'rb'):
            if random.random() < prob:
                fr.write(line)
            else:
                fe.write(line)

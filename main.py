# -*- coding:utf-8 -*-

from utils import data_preprocess
from model import DeepFM
import torch

result_dict = data_preprocess.read_criteo_data('./data/tiny_train_input.csv', './data/category_emb.csv')
test_dict = data_preprocess.read_criteo_data('./data/tiny_test_input.csv', './data/category_emb.csv')

with torch.cuda.device(2):
    deepfm = DeepFM.DeepFM(39,result_dict['feature_sizes'],verbose=True,use_cuda=True, weight_decay=0.0001,use_fm=True,use_ffm=False,use_deep=True).cuda()
    deepfm.fit(result_dict['index'], result_dict['value'], result_dict['label'],
               test_dict['index'], test_dict['value'], test_dict['label'],ealry_stopping=True,refit=True)

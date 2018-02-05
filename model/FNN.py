# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of FNN

Reference:
[1] Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction

Weinan Zhang, Tianming Du, Jun Wang

"""
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn

"""
    网络结构部分
"""

class FNN(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    deep_layers_activation: relu or sigmoid etc
    n_epochs: epochs
    batch_size: batch_size
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    verbose: verbose
    pre_weight_decay: pretrain 's weight decay (L2 penalty)
    weight_decay: weight decay (L2 penalty)
    random_seed: random_seed=950104 someone's birthday, my lukcy number
    use_fm: bool
    use_ffm: bool
    loss_type: "logloss", only
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1
    greater_is_better: bool. Is the greater eval better?


    Attention: only support logsitcs regression
    """

    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth=2, deep_layers=[32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='tanh', n_epochs=64, batch_size=256, learning_rate=0.003,
                 optimizer_type='adam', is_batch_norm=False, verbose=False, random_seed=950104, pre_weight_decay= 0.0,weight_decay=0.0,
                 use_fm=True, use_ffm=False, loss_type='logloss', eval_metric=roc_auc_score,
                 use_cuda=True, n_class=1, greater_is_better=True
                 ):
        super(FNN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.pre_weight_decay = pre_weight_decay
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better
        self.pretrain = False

        torch.manual_seed(self.random_seed)

        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm:
            print("The model is FNN(fm+nn layers)")
        elif self.use_ffm:
            print("The model is FFNN(ffm+nn layers)")
        else:
            print("You have to choose more than one of (fm, ffm, deep) models to use")
            exit(1)

        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True) #w0
            self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes]) #wi
            self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]) #vi
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)
            self.ffm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size,1) for feature_size in self.feature_sizes])
            self.ffm_second_order_embeddings = nn.ModuleList([nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for feature_size in self.feature_sizes])
            print("Init ffm part succeed")

        print("Init nn part")
        if self.is_deep_dropout:
            self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
        if not use_ffm:
            self.linear_1 = nn.Linear(1 + self.field_size + self.field_size  * self.embedding_size, deep_layers[0])
        else:
            self.linear_1 = nn.Linear(1 + self.field_size + self.field_size * self.field_size * self.embedding_size, deep_layers[0])

        if self.is_batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])

        if self.is_deep_dropout:
            self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            if self.is_batch_norm:
                setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))
        self.deep_last_layer = nn.Linear(self.deep_layers[-1], self.n_class)
        print("Init nn part succeed")

        print "Init succeed"

    def forward(self, Xi, Xv):
        """
        :param Xi: index input tensor, batch_size * k * 1
        :param Xv: value input tensor, batch_size * k * 1
        :param is_pretrain: the para to decide fm pretrain or not
        :return: the last output
        """
        if self.pretrain and self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_first_order_sum = torch.sum(sum(fm_first_order_emb_arr),1)
            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
            fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
            fm_sum_second_order_emb_square = fm_sum_second_order_emb*fm_sum_second_order_emb # (x+y)^2
            fm_second_order_emb_square = [item*item for item in fm_second_order_emb_arr]
            fm_second_order_emb_square_sum = sum(fm_second_order_emb_square) #x^2+y^2
            fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
            fm_second_order_sum = torch.sum(fm_second_order,1)
            return self.fm_bias+fm_first_order_sum+fm_second_order_sum
        elif self.pretrain and self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            sum_ = torch.sum(sum(ffm_first_order_emb_arr),1)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:,i,:]), 1).t() * Xv[:,i]).t() for emb in  f_embs] for i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            for i in range(self.field_size):
                for j in range(i+1, self.field_size):
                    sum_ += torch.sum((ffm_second_order_emb_arr[i][j]*ffm_second_order_emb_arr[j][i]),1)
            return self.ffm_bias + sum_
        elif not self.pretrain and self.use_fm:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
            fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
            fm_first_order = torch.cat(fm_first_order_emb_arr,1)
            fm_second_order = torch.cat(fm_second_order_emb_arr,1)
            if self.use_cuda:
                fm_bias = self.fm_bias * Variable(torch.ones(Xi.data.shape[0],1)).cuda()
            else:
                fm_bias = self.fm_bias * Variable(torch.ones(Xi.data.shape[0], 1))
            deep_emb = torch.cat([fm_bias,fm_first_order,fm_second_order],1)
            if self.deep_layers_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = F.tanh
            else:
                activation = F.relu
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
            x_deep = self.deep_last_layer(x_deep)
            return torch.sum(x_deep,1)
        else:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.ffm_first_order_embeddings)]
            ffm_second_order_emb_arr = [torch.cat([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs],1) for
                                        i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr,1)
            ffm_second_order = torch.cat(ffm_second_order_emb_arr,1)
            if self.use_cuda:
                ffm_bias = self.ffm_bias * Variable(torch.ones(Xi.data.shape[0], 1)).cuda()
            else:
                ffm_bias = self.ffm_bias * Variable(torch.ones(Xi.data.shape[0], 1))
            deep_emb = torch.cat([ffm_bias, ffm_first_order, ffm_second_order], 1)
            if self.deep_layers_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = F.tanh
            else:
                activation = F.relu
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
            x_deep = self.deep_last_layer(x_deep)
            return torch.sum(x_deep,1)

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None,
                y_valid = None, is_pretrain = False, ealry_stopping=False, refit=False, save_path = None):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                        indi_j is the feature index of feature field j of sample i in the training set
            :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                        vali_j is the feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param is_pretrain: pretrain or not ?
        :param ealry_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :param save_path: the path to save the model
        :return:
        """
        """
        pre_process
        """
        if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            print("Save path is not existed!")
            return

        if is_pretrain:
            print("The model is pre_training now. You must change the mode in the next fitting")

        if self.verbose:
            print("pre_process data ing...")
        self.pretrain = is_pretrain
        is_valid = False
        Xi_train = np.array(Xi_train).reshape((-1,self.field_size,1))
        Xv_train = np.array(Xv_train)
        y_train = np.array(y_train)
        x_size = Xi_train.shape[0]
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1,self.field_size,1))
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True
        if self.verbose:
            print("pre_process data finished")

        """
            train model
        """
        model = self.train()
        if self.pretrain:
            weight_decay = self.pre_weight_decay
        else:
            weight_decay = self.weight_decay
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)

        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = x_size // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter+1):
                offset = i*self.batch_size
                end = min(x_size, offset+self.batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data[0]
                if self.verbose:
                    if i % 100 == 99:  # print every 100 mini-batches
                        eval = self.evaluate(batch_xi, batch_xv, batch_y)
                        print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                              (epoch + 1, i + 1, total_loss/100, eval, time()-batch_begin_time))
                        total_loss = 0.0
                        batch_begin_time = time()

            train_loss, train_eval = self.eval_by_batch(Xi_train,Xv_train,y_train,x_size)
            train_result.append(train_eval)
            print('*'*50)
            print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                  (epoch + 1, train_loss, train_eval, time()-epoch_begin_time))
            print('*'*50)

            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)
                print('*' * 50)
                print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval,time()-epoch_begin_time))
                print('*' * 50)
            if save_path:
                torch.save(self.state_dict(),save_path)
            if is_valid and ealry_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch+1))
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if is_valid and refit:
            if self.verbose:
                print("refitting the model")
            if self.greater_is_better:
                best_epoch = np.argmax(valid_result)
            else:
                best_epoch = np.argmin(valid_result)
            best_train_score = train_result[best_epoch]
            Xi_train = np.concatenate((Xi_train,Xi_valid))
            Xv_train = np.concatenate((Xv_train,Xv_valid))
            y_train = np.concatenate((y_train,y_valid))
            x_size = x_size + x_valid_size
            self.shuffle_in_unison_scary(Xi_train,Xv_train,y_train)
            for epoch in range(64):
                batch_iter = x_size // self.batch_size
                for i in range(batch_iter + 1):
                    offset = i * self.batch_size
                    end = min(x_size, offset + self.batch_size)
                    if offset == end:
                        break
                    batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                    batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                    batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                    if self.use_cuda:
                        batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                    optimizer.zero_grad()
                    outputs = model(batch_xi, batch_xv)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                train_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size)
                if save_path:
                    torch.save(self.state_dict(), save_path)
                if abs(best_train_score-train_eval) < 0.001 or \
                        (self.greater_is_better and train_eval > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break
            if self.verbose:
                print("refit finished")

    def eval_by_batch(self,Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        if self.use_ffm:
            batch_size = 16384*2
        else:
            batch_size = 16384
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter+1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
            outputs = model(batch_xi, batch_xv)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data[0]*(end-offset)
        total_metric = self.eval_metric(y,y_pred)
        return total_loss/x_size, total_metric

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: the same as fit function
        :param Xv: the same as fit function
        :return: output, ont-dim array
        """
        Xi = np.array(Xi).reshape((-1,self.field_size,1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def inner_predict(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """
        y_pred = self.inner_predict_proba(Xi, Xv)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)

"""
    test part
"""
import sys
sys.path.append('../')
from utils import data_preprocess

result_dict = data_preprocess.read_criteo_data('../data/train.csv', '../data/category_emb.csv')
test_dict = data_preprocess.read_criteo_data('../data/test.csv', '../data/category_emb.csv')
with torch.cuda.device(2):
    fnn = FNN(39, result_dict['feature_sizes'], batch_size=128 * 64, verbose=True, use_cuda=True,
                      pre_weight_decay= 0.0001 ,weight_decay=0.00001, use_fm=False, use_ffm=True).cuda()
    fnn.load_state_dict(torch.load('../data/model/ffnn.pkl'))
    # fnn.fit(result_dict['index'], result_dict['value'], result_dict['label'],
    #            test_dict['index'], test_dict['value'], test_dict['label'],ealry_stopping=True,refit=False,is_pretrain=True,save_path='../data/model/ffnn.pkl')
    fnn.fit(result_dict['index'], result_dict['value'], result_dict['label'],
               test_dict['index'], test_dict['value'], test_dict['label'],ealry_stopping=True,refit=False,is_pretrain=False,save_path='../data/model/ffnn.pkl')

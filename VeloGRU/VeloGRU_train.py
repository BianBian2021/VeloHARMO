# encoding=utf-8
import torch
from torch import nn
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from time import time
from sklearn.linear_model import LinearRegression
import os
import random
import logging
from VeloGRU_data import load_data,AllData,MyDataSet
from torch.utils.data import DataLoader
logging.basicConfig(level=logging.INFO)
from models import *


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def grad_clipping(net, theta):  # 检查梯度，范数大于θ的改为θ
    if isinstance(net, nn.Module):
        params = [param for param in net.parameters() if param.requires_grad == True]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((param.grad ** 2)) for param in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sequence_mask(X, valid_len, value=0):  # 给X,valid_len,会使X在valid_len后的值全为0
    # X/label:(B,L,1),valid_len:(B,1) 和d2l不同，valid_len本来就是(B,1)
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)  # ,按照dim=1,=L
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len  # (1,L)<(B,1)--->(B,L) 各自在为1的维度上广播
    X[~mask] = value  # X:(B,L,1); mask(B,L)--->?当成是选中吧(B,L,V) #例如X:(B,L,1) X[0]=1会使 第一维的 L*V个元素都为1
    return X


class MaskedHuberLoss(nn.HuberLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedHuberLoss, self).forward(pred, label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class MaskedMSELoss(nn.MSELoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedMSELoss, self).forward(pred, label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class MaskedCosLoss(nn.Module):
    def forward(self,pred,label,valid_len):  #
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        maxlen = pred.size(1)  # ,按照dim=1,=L
        mask = torch.arange((maxlen), dtype=torch.float32, device=pred.device)[None, :] < valid_len
        #(1,L)<(B,1)-->(B,L)
        mask=mask.unsqueeze(-1) #广播是从尾部看的
        pred = pred.masked_select(mask)  #(B*L)
        label= label.masked_select(mask) #(B*L)
        return -cos(pred - pred.mean(), label - label.mean()) #取负值以最大化
        # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # pearson=0
        # for i in range(len(pred)):
        #     pred_i=pred[i][:valid_len[i]]  #(L,1)
        #     label_i=label[i][:valid_len]
        #     pearson += cos(pred_i - pred_i.mean(), label_i - label_i.mean())
        # return pearson

# def data_iter(X, Y, length, batch_size):
#     num_examples = len(X)
#     indices = list(range(num_examples))
#     # 这些样本是随机读取的，没有特定的顺序
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(
#             indices[i: min(i + batch_size, num_examples)])
#         yield X[batch_indices], Y[batch_indices], length[batch_indices]


def plot_correlation(y1, y2, op="relationship"):  # 绘制y1, y2的线性关系
    y1 = np.array(y1)
    y2 = np.array(y2)
    model = LinearRegression()
    x_train = np.array(y1.reshape(-1, 1))
    model.fit(x_train, y2)

    b = model.intercept_  # 截距
    a = model.coef_[0]
    y_train_pred = model.predict(x_train)
    plt.plot(x_train, y_train_pred, color='red', linewidth=3)
    score = model.score(x_train, y2)  # r2 score
    a, b, score = ('%.4f' % a), ('%.4f' % b), ('%.4f' % score)
    # pearson = np.corrcoef(y1, y2)[0][1] #pearson相关系数 线性回归=r
    ax = plt.gca()
    plt.text(0.1, 0.9, ("${r^2}$=%s" % score), transform=ax.transAxes, fontsize="20")
    # plt.text(0.2, 0.75, ("pearson=%s" % pearson))
    plt.scatter(y1,  # x轴数据为汽车速度
                y2,  # y轴数据为汽车的刹车距离
                s=3,  # 设置点的大小
                c='k',  # 设置点的颜色
                marker='o',  # 设置点的形状
                alpha=0.9,  # 设置点的透明度
                linewidths=0.3,  # 设置散点边界的粗细
                )
    # 添加轴标签和标题
    # plt.title('fraction relations')
    plt.xlabel('Observed Velocity', fontsize="20")
    plt.ylabel('Predicted Velocity', fontsize="20")
    plt.tick_params(labelsize=15)
    max_velocity = max(max(y1), max(y2))
    plt.xlim(0, max_velocity)
    plt.ylim(0, max_velocity)
    plt.savefig('{}_relationship_result.jpg'.format(op), bbox_inches='tight', pad_inches=0.2)

    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴。
    plt.close()  # 关窗口
    return float(score)

def plot_loss(t, v, op='0'):  # 绘制训练过程中loss改变的趋势
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(len(t)), t, label="train_loss")
    ax.plot(range(len(v)), v, label="validation_loss")
    plt.legend()
    plt.savefig('{}_loss_of_training.jpg'.format(op), bbox_inches='tight', pad_inches=0.2, dpi=400)
    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴。
    plt.close()  # 关窗口



def getloss(data_iter, net, loss, device):  # 给定模型和样本，计算样本的平均loss
    net.eval()
    l = 0
    valid_codon = 0
    for X0, Y0, len0 in data_iter:
        max_len = X0.shape[1]
        X = X0
        y = Y0  # (B,L,1)
        X, y = X.to(device), y.to(device)
        y_hat = net(X, len0)  # y_hat的size:[B,L,1]
        with torch.no_grad():
            l += loss(y, y_hat, len0).sum() * max_len  # (B,1)--->标量
            valid_codon += len0.sum()
    return (l / valid_codon).cpu().numpy()  # l/valid_codon is a tensor


def plot_test(net, test_iter, device=try_gpu(),exclude_zeros=False,op="0",showfig=True):  # 绘制测试集上的预测结果
    net.eval()
    net.to(device)
    trues = []
    preds = []
    per_gene_cor = []
    num = 0
    for X0, Y0, len0 in test_iter:  # (B,L,V) (B,L,1),(B,1)
        batch_size = X0.shape[0]
        assert batch_size == 1 and len(len0) == 1, "当前测试数据应逐个计算并画图"
        X = X0
        y = Y0.permute(1, 0, 2).reshape(-1)  # (B,L,1)---->(L,B,1)--->(L*B,1)
        X, y = X.to(device), y.to(device)
        y_hat = net(X, len0)
        y_hat = y_hat.permute(1, 0, 2).reshape(-1)  # (B,L,1)---->(L,B,1)--->(L*B,1)

        y, y_hat = y[:len0[0]], y_hat[:len0[0]]  # 注意，这里要求len0只有一个元素
        if exclude_zeros:
            mask=y>0
            y,y_hat=torch.masked_select(y,mask),torch.masked_select(y_hat,mask)
        with torch.no_grad():
            y, y_hat = y.cpu().numpy(), y_hat.cpu().numpy()
        if showfig:
            plt.plot(y_hat, label='pred')
            plt.plot(y, label='true')
            plt.legend()
            plt.savefig('{}_Gene_velocity_result_{}.jpg'.format(op, num), bbox_inches='tight', pad_inches=0.2, dpi=400)
            plt.clf()  # 清图。
            plt.cla()  # 清坐标轴。
            plt.close()  # 关窗口
        num += 1
        trues.extend(y)
        preds.extend(y_hat)
        score, p = stats.pearsonr(y, y_hat)
        per_gene_cor.append(score)
    mean = np.mean(per_gene_cor)
    std = np.std(per_gene_cor)
    logging.info("average pearson of {}:{}±{}".format(op, mean, std))
    score= plot_correlation(trues, preds, op)
    return (mean,std,score)

def train_epoch(net, train_iter, loss, updater, device):  # 训练一个epoch
    state, epoch_start = None, time()
    metric = Accumulator(2)

    for X, Y, valid_len in train_iter:
        batch_size = X.shape[0]  # X:(B,L,V)
        max_len = X.shape[1]
        y = Y  # (B,L,1)
        X, y,valid_len = X.to(device), y.to(device),valid_len.to(device)
        y_hat = net(X, valid_len)  # y_hat:[B,L,1]
        l = loss(y_hat, y, valid_len).mean()  # loss:(B,1)
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
        metric.add(l * batch_size * max_len, valid_len.sum())
        # l*batchsize*max_len是本batch所有loss,在loss计算时按max_len求了一次平均，l又对loss按batch求了一次平均
        # valid_len求和是本batch中所有有效codon的个数
    return metric[0] / metric[1], metric[1] / (time() - epoch_start)

def train_and_validate(net, data, num_epochs, device, loss, updater, scheduler=None, test_ratio=0.1,validation_ratio=0.1 ,batch_size=32, loga=False,
          save_best=True,shuffle=False,seed=None,name="current"):
    data_X, data_Y, valid_len = data
    valid_len = valid_len.to(device)
    # 是否将数据log化处理
    if loga: data_Y = torch.log(data_Y + 1)
    # 划分测试
    data = AllData(data_X, data_Y, valid_len, validation_ratio, test_ratio, batch_size, shuffle=shuffle, seed=seed)
    val_iter = data.get_val()
    train_iter = data.get_train()

    train_loss_recorder = []
    val_loss_recorder = []
    best_loss = float("inf")
    for epoch in range(num_epochs):  # 每个epoch用函数train_epoch训练
        net.train()  # 训练模式
        train_loss, rate = train_epoch(net, train_iter, loss, updater, device)
        val_loss = getloss(val_iter, net, loss, device)
        if scheduler: scheduler.step()  # 动态调整学习率
        if save_best and val_loss < best_loss:
            print("At epoch {},current validation loss:{} smaller than previous best:{},update best".format(epoch + 1,
                                                                                                            val_loss,
                                                                                                            best_loss))
            best_loss = val_loss
            # torch.save(net.module.state_dict(), os.path.join(os.getcwd(), 'best_network.pth'))
            torch.save(net, os.path.join(os.getcwd(), '{}_best_network.pth'.format(name)))  # 保存整个模型
        if (epoch + 1) % 10 == 0:  # 每十个epoch记录一下loss
            logging.info("At epoch {},train loss:{},validation loss:{}".format(epoch + 1, train_loss, val_loss))
            # logging.info("current learning rate:{}".format(updater.param_groups[0]["lr"]))
            train_loss_recorder.append(train_loss)
            val_loss_recorder.append(val_loss)
    # torch.save(net.module.state_dict(), os.path.join(os.getcwd(), 'final_network.pth'))
    torch.save(net, os.path.join(os.getcwd(), '{}_final_network.pth'.format(name)))  # 保存整个模型
    plot_loss(train_loss_recorder, val_loss_recorder,op=name)
    # print(f'loss {train_loss:.4f}, {rate:.1f} codon/秒 {str(device)}')

def test(model,data_path,nt=True,aa=True,protein=True,ss=True,exclude_zeros=False,gene_filter="0.6",op="0",showfig=False):
    # 因为数据已经经过筛选最低是0.6，0.6相当于没有筛选
    logging.info("feature option nt:{},aa:{},protein:{},ss:{}".format(nt, aa, protein, ss))
    data_X, data_Y, valid_len=load_data(data_path,nt,aa,protein,ss)
    num_inputs = len(data_X[0][0])
    logging.info("feature dimension:{}".format(num_inputs))
    dataset=MyDataSet(( data_X, data_Y, valid_len))
    model = torch.load(model)
    filtered_dataset=dataset.filter_non_zero(float(gene_filter))
    test_loader = DataLoader(filtered_dataset, batch_size=1, shuffle=True)
    torch.cuda.empty_cache()
    return plot_test(model, test_loader,exclude_zeros=exclude_zeros ,op="{}_{}_test".format(op,gene_filter),showfig=showfig)

def train(datafile,nt=True,aa=True,protein=True,ss=True,name="current"):
    logging.info("feature option nt:{},aa:{},protein:{},ss:{}".format(nt,aa,protein,ss))
    data_X, data_Y, valid_len = load_data(datafile,nt,aa,protein,ss)
    num_inputs = len(data_X[0][0])
    logging.info("feature dimension:{}".format(num_inputs))
    # 模型参数
    num_hiddens = 128
    num_layers = 2
    dropout = 0.3

    # 建立模型
    num_outputs, device = 1, try_gpu()
    net = BiGRU(num_inputs, num_hiddens, num_layers=num_layers, output_size=num_outputs)
    logging.info("Use {} GPUs for training!".format(torch.cuda.device_count()))
    net = net.to(device)

    # updater参数
    lr = 1e-3
    weight_decay = 1e-6

    updater = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(updater, 5, gamma=0.95)

    # 训练参数
    num_epoches = 100
    test_ratio = 0.1  # 0~1之间的float表示用于测试的比例
    validation_ratio = 0.1  # 测试之外的数据再进一步划分
    batch_size = 16
    loga = False  # 是否log化速率值

    # 训练

    start_time = time()
    train_and_validate(net, (data_X, data_Y, valid_len),
                       num_epochs=num_epoches,
                       device=device,
                       loss=MaskedHuberLoss(),
                       updater=updater,
                       test_ratio=test_ratio,
                       validation_ratio=validation_ratio,
                       batch_size=batch_size,
                       scheduler=scheduler,
                       loga=loga,
                       name=name)
    print("time elapse:", time() - start_time)
#[1,2,3] [TTF] __>[1,2]
if __name__ == '__main__':

    data_X, data_Y, valid_len = load_data(r"E:\course\log\Doc\Velocity_test\P.aeruginosa.summary.add.pro.SS.txt")
    num_inputs=len(data_X[0][0])
    logging.info("feature dimension:{}".format(num_inputs))
    # 模型参数
    num_hiddens = 128
    num_layers = 2
    dropout = 0.3

    # 建立模型
    num_outputs, device = 1, try_gpu()
    net = BiGRU(num_inputs, num_hiddens, num_layers=num_layers, output_size=num_outputs)
    logging.info("Use {} GPUs for training!".format(torch.cuda.device_count()))
    net = net.to(device)

    # updater参数
    lr = 1e-3
    weight_decay = 1e-6

    updater = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(updater, 5, gamma=0.95)

    # 训练参数
    num_epoches = 500
    test_ratio = 0.1  # 0~1之间的float表示用于测试的比例
    validation_ratio=0.1  # 测试之外的数据再进一步划分
    batch_size = 16
    loga = False  # 是否log化速率值

    # 训练

    start_time = time()
    train_and_validate(net, (data_X, data_Y, valid_len),
          num_epochs=num_epoches,
          device=device,
          loss=MaskedHuberLoss(),
          updater=updater,
          test_ratio=test_ratio,
          validation_ratio=validation_ratio,
          batch_size=batch_size,
          scheduler=scheduler,
          loga=loga)
    torch.cuda.empty_cache()
    print("time elapse:", time() - start_time)

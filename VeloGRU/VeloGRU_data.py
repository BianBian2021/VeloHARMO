#encoding=utf-8
from sklearn.model_selection import KFold
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torch
import utils
import time

#split data 将txt数据分为不同的train和test并保存为txt
def split_data(data_file,num_fold=10):
    kf=KFold(n_splits=num_fold,shuffle=True)
    data_file_name=os.path.splitext(data_file)[0]
    print(data_file_name)
    with open(data_file, "r") as f:
        data = f.readlines()
        assert len(data) % 4 == 0
        n=len(data)//4
        fold_number=0
        for train,test in kf.split(list(range(n))):
            with open("{}_{}_train.txt".format(data_file_name,fold_number),"w") as target_file:
                for i in train:
                    target_file.write(data[4*i])
                    target_file.write(data[4 * i+1])
                    target_file.write(data[4 * i+2])
                    target_file.write(data[4 * i+3])
            with open("{}_{}_test.txt".format(data_file_name,fold_number),"w") as target_file:
                for i in test:
                    target_file.write(data[4 * i])
                    target_file.write(data[4 * i + 1])
                    target_file.write(data[4 * i + 2])
                    target_file.write(data[4 * i + 3])
            fold_number += 1
class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[0][item], self.data[1][item], self.data[2][item]

    def __len__(self):
        return len(self.data[0])

    def filter_non_zero(self, non_zero_ratio):
        x, y, valid_len = self.data
        index = ((torch.sum((y > 0), dim=1) / valid_len) > non_zero_ratio).squeeze(-1)
        x = x[index]
        y = y[index]
        valid_len = valid_len[index]
        return MyDataSet((x, y, valid_len))  # 返回一个新的MyData对象
class AllData:
    def __init__(self, x, y, valid_len, validation_ratio=0.0, test_ratio=0.1, batch_size=16, shuffle=False,seed=None):  #

        self.batch_size = batch_size

        if shuffle:  # 如果需要shuffle,使用指定的seed进行shuffle
            np.random.seed(seed)
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x, y, valid_len = x[indices], y[indices], valid_len[indices]
        assert 0 <= validation_ratio <= 1  # 划分训练和验证集
        validation_num = int(round(validation_ratio * len(x)))
        self.val_data = MyDataSet((x[:validation_num], y[:validation_num], valid_len[:validation_num]))
        self.train_data = MyDataSet((x[validation_num:], y[validation_num:], valid_len[validation_num:]))

    def get_train(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def get_val(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)





def truncate_pad(maxlength, X, Y):
    assert len(X) == len(Y),"特征长度：{}，速率长度：{}".format(len(X),len(Y))
    X = X.astype(float)
    Y=Y.astype(float)
    # 将字符串数组转换为Tensor对象
    X = torch.tensor(X,dtype=torch.float)
    Y=torch.tensor(Y,dtype=torch.float)
    valid_len = len(X)
    if valid_len < maxlength:
        new_shape_X = torch.tensor(X.shape)
        new_shape_X[0] = maxlength - valid_len

        new_shape_Y = torch.tensor(Y.shape)
        new_shape_Y[0] = maxlength - valid_len

        X = torch.cat((X, torch.zeros(tuple(new_shape_X))))  # 默认按dim=0 concat
        Y = torch.cat((Y, torch.zeros(tuple(new_shape_Y))))
        return X, Y, valid_len
    else:
        return X[:maxlength], Y[:maxlength], maxlength



#load data 将数据从txt文件中读取出来,和ribomimo一致,应该返回x,y,valid_len
def load_data(path,nt=True,aa=True,protein=True,add_ss=True):

    #加载转换所需的词典 对应load_onehot
    aa_table = utils.aa_table
    codon2aa=utils.codon2aa
    SS = "-STIGEBH"
    nts = "ATCG"
    codon2idx = {}
    for nt1 in nts:
        for nt2 in nts:
            for nt3 in nts:
                codon2idx[nt1 + nt2 + nt3] = len(codon2idx)
    onehot_nt = {nts[i]: np.eye(4)[i] for i in range(len(nts))}
    onehot_codon = {codon: np.eye(64)[codon2idx[codon]] for codon in codon2idx}
    onehot_aa = {aa_table[i, 2]: np.eye(21)[i] for i in range(len(aa_table))}
    onehot_ss={SS[i]: np.eye(len(SS))[i] for i in range(len(SS))}

    #读取文件 #load_data
    with open(path, "r") as f:
        data = f.readlines()
    assert len(data) % 4 == 0
    list_name = []
    list_seq = []
    list_pro_feature = []  # add protein feature1
    list_density = []
    list_ss=[]
    list_avg = []
    for i in range(len(data) // 4):
        name = data[4 * i + 0].split('>')[1].split()[0]
        seq = data[4 * i + 1].split()
        count = [float(e) for e in data[4 * i + 2].split()]
        pro_feature = [list(map(float, e.split(",")[:-1])) for e in data[4 * i + 3].split()]# add protein feature
        ss=[e.split(",")[-1] for e in data[4 * i + 3].split()]
        avg = np.mean(np.array(count)[np.array(count) > 0.5])
        density = (np.array(count) > 0.5) * np.array(count) / avg
        # criteria containing ribosome density AND coverage percentage
        list_name.append(name)
        list_seq.append(seq)
        list_density.append(density)
        list_pro_feature.append(pro_feature)  # add protein feature3 #1d=per gene, 2d=each codon, 3d=each feature
        list_ss.append(ss)
        list_avg.append(avg)

    list_name = np.array(list_name,dtype=object)
    list_pro_feature = np.array(list_pro_feature,dtype=object)  # add protein feature4
    list_ss=np.array(list_ss,dtype=object)
    list_seq = np.array(list_seq,dtype=object)
    list_density = np.array(list_density,dtype=object)
    list_avg = np.array(list_avg,dtype=object)
    list_gene_all = list_name

    dict_seq = {}
    dict_seq_nt = {}
    dict_seq_aa = {}
    dict_seq_codon = {}
    dict_density = {}
    dict_ss={}
    dict_pro_feature = {}  # add protein feature5
    dict_avg = {}
    index = []
    for i in range(len(list_gene_all)):
        codons = list_seq[i]
        nts = "".join(codons)
        if "N" in nts:
            continue
        aas = [codon2aa[codon] for codon in codons]
        index.append(i)
        dict_seq[list_gene_all[i]] = nts
        dict_seq_nt[list_gene_all[i]] = np.array([onehot_nt[nt] for nt in nts],dtype=object)
        dict_seq_codon[list_gene_all[i]] = np.array([onehot_codon[codon] for codon in codons],dtype=object)
        dict_seq_aa[list_gene_all[i]] = np.array([onehot_aa[aa] for aa in aas],dtype=object)
        dict_density[list_gene_all[i]] = list_density[i]
        dict_ss[list_gene_all[i]] = np.array([onehot_ss[structure] for structure in list_ss[i]],dtype=object)
        dict_pro_feature[list_gene_all[i]] = list_pro_feature[i]  # add protein feature6
        dict_avg[list_gene_all[i]] = list_avg[i]

    #将读取的数据变为x,y,valid_len的形式 #get_data_pack
    n=len(dict_seq_codon)
    x_input = [dict_seq_codon[gene] for gene in dict_seq_codon]
    length = [ torch.tensor([len(dict_seq_codon[gene])])for gene in dict_seq_codon]
    if nt:
        x_nt = [dict_seq_nt[gene] for gene in dict_seq_nt]
        x_input = [np.concatenate([x_input[i], x_nt[i].reshape((len(x_nt[i]) // 3, -1))], axis=1) for i in
                   range(n)]
    if aa:
        x_aa = [dict_seq_aa[gene] for gene in dict_seq_aa]
        x_input = [np.concatenate([x_input[i], x_aa[i]], axis=1) for i in range(n)]
    if protein:  # add protein feature7
        x_pro_feature = [dict_pro_feature[gene] for gene in dict_pro_feature]
        x_input = [np.concatenate([x_input[i], x_pro_feature[i]], axis=1) for i in range(n)]
    if add_ss:
        x_ss = [dict_ss[gene] for gene in dict_ss]
        x_input = [np.concatenate([x_input[i], x_ss[i]], axis=1) for i in range(n)]
    density = [dict_density[gene] for gene in dict_density]
    max_len=max(length)
    for i in range(len(x_input)):
        x_input[i],density[i],_=truncate_pad(max_len,x_input[i],density[i])

    x_input=torch.stack(x_input)#warning提示x_input中的元素是ndarray,实际上在pad中已经改成tensor了
    density=torch.stack(density).unsqueeze(-1)
    length=torch.stack(length)
    return x_input,density,length




# # 训练时通过指定不同的txt文件来实现不同的train test划分。 在训练时对train随机选择validation set
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # define environment
#     parser.add_argument("--gpu", default="0", help="which GPU to use", type=str)
# x,y,valid_len=load_data(r"E:\course\log\Doc\Velocity_test\P.aeruginosa.summary.add.pro.SS.txt")
# print(x[0],type(x[0]))
# for i in range(len(x)):
#     print(len(x[i]))

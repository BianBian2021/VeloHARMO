import torch
from torch import nn
import numpy as np
from torch.nn import functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class BiGRU(nn.Module):
    def __init__(self,num_inputs=64, num_hiddens=128,num_layers=2,output_size=1,dropout=0.3, **kwargs):
        '''
        num_input:输入的特征维度
        num_hiddens:RNN隐藏层的维度
        num_layers:RNN隐藏层的个数
        output_size:默认是1
        '''
        super(BiGRU, self).__init__(**kwargs)
        self.num_directions = 2
        self.num_inputs = num_inputs #特征维度：64/70
        self.output_size=output_size #输出维度：1

        self.rnn =  nn.GRU(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers,batch_first=True,dropout=dropout)
        self.num_hiddens = self.rnn.hidden_size #128
        self.relu=nn.ReLU()
        self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

    def forward(self, X, length):
        self.rnn.flatten_parameters()  # 使用多个GPU时,权重会被分别存储,将权重压成一块，减少内存使用

        length = length.cpu().squeeze(1)  # pack_padded要求数据在cpu上,length原先是（B,1）
        total_length = X.shape[1]
        X = pack_padded_sequence(X, length, batch_first=True, enforce_sorted=False)
        X, _ = self.rnn(X)  # [B,L,H]#默认使用全0的隐状态
        Y, _ = pad_packed_sequence(X, batch_first=True, padding_value=0.0,
                                   total_length=total_length)  # 要恢复到整体最长的L，而不是该部分最长的L

        Y = self.relu(Y)
        output = self.linear(Y)  # [B,L,1]
        output = self.relu(output)  # 预测值应该都大于零
        return output
# class avg_BiGRU(nn.Module):  # 没什么用，怎么分析取平均对整体的影响？
#     def __init__(self, num_inputs=64, num_hiddens=128, num_layers=2, output_size=1, dropout=0.3, **kwargs):
#         '''
#         num_input:输入的特征维度
#         num_hiddens:RNN隐藏层的维度
#         num_layers:RNN隐藏层的个数
#         output_size:默认是1
#         '''
#         super(avg_BiGRU, self).__init__(**kwargs)
#         self.num_directions = 2
#         self.num_inputs = num_inputs  # 特征维度：64/70
#         self.output_size = output_size  # 输出维度：1
#
#         self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers, batch_first=True,
#                           dropout=dropout)
#         self.num_hiddens = self.rnn.hidden_size  # 128
#         self.relu = nn.ReLU()
#         self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)
#     def forward(self, X,length):
#         self.rnn.flatten_parameters()   #使用多个GPU时,权重会被分别存储,将权重压成一块，减少内存使用
#
#         length=length.cpu().squeeze(1) #pack_padded要求数据在cpu上,length原先是（B,1）
#         total_length=X.shape[1]
#         X = pack_padded_sequence(X, length, batch_first=True,enforce_sorted=False)
#         X, _ = self.rnn(X) #[B,L,H]#默认使用全0的隐状态
#         Y, _ = pad_packed_sequence(X, batch_first=True, padding_value=0.0,total_length=total_length) #要恢复到整体最长的L，而不是该部分最长的L
#
#         Y=self.relu(Y)
#         output= self.linear(Y)          # [B,L,1]
#         output=self.relu(output) #预测值应该都大于零
#
#         length=length.unsqueeze(-1).to(output.device)  # (B,1)
#         mask = torch.arange((total_length), dtype=torch.float32, device=output.device)[None, :] < length
#         #(1,L)<(B,1)-->(B,L)
#         output = output * mask.unsqueeze(-1)#广播是从尾部看的
#         output_mean=torch.sum(output,dim=1,keepdim=True)/length.unsqueeze(-1)
#         return output/output_mean


class DuelFeatureGRU(nn.Module):  #尝试如何拼接protein feature
    def __init__(self,num_inputs=70, num_hiddens=128,num_layers=2,output_size=1,dropout=0.3, **kwargs):
        '''
        num_input:输入的特征维度
        num_hiddens:RNN隐藏层的维度
        num_layers:RNN隐藏层的个数
        output_size:默认是1
        '''
        assert num_inputs==70,"a test for codon+protein feature"
        super(DuelFeatureGRU, self).__init__(**kwargs)
        self.num_directions = 2
        self.num_inputs = num_inputs #特征维度：64/70
        self.output_size=output_size #输出维度：1

        self.rnn =  nn.GRU(num_inputs, num_hiddens, bidirectional=True, num_layers=num_layers,batch_first=True,dropout=dropout)
        self.linear1=nn.Linear(6,num_hiddens*2)
        self.num_hiddens = self.rnn.hidden_size #128
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(self.num_hiddens * 4, self.output_size)

    def forward(self, X,length):
        self.rnn.flatten_parameters()   #使用多个GPU时,权重会被分别存储,将权重压成一块，减少内存使用
        length=length.cpu().squeeze(1) #pack_padded要求数据在cpu上,length原先是（B,1）
        total_length=X.shape[1]
        Y = pack_padded_sequence(X, length, batch_first=True,enforce_sorted=False)
        Y, _ = self.rnn(Y) #[B,L,H]#默认使用全0的隐状态
        Y, _ = pad_packed_sequence(Y, batch_first=True, padding_value=0.0,total_length=total_length) #要恢复到整体最长的L，而不是该部分最长的L
        Y=torch.cat((self.linear1(X[:, :, -6:]),Y),dim=-1)  #(B,L,6)--->(B,L,H)
        Y=self.relu(Y)
        output= self.linear2(Y)          # [B,L,1]
        output=self.relu(output) #预测值应该都大于零
        return output




class Fused_model(nn.Module):
    def __init__(self,slow_net,fast_net,frac_net,freeze=False):
        super(Fused_model,self).__init__()
        self.net1=slow_net
        self.net2=fast_net
        self.net3=frac_net
        self.linear1=nn.Linear(1,1)#frac的输出值始终大于零，设法返回到[-无穷，+无穷]区间里
        self.linear2=nn.Linear(1,1)#tanh到[-1,1]后再线性映射一下试图到正负无穷
        if freeze==True:
            for param in self.net1.parameters():
                param.requires_grad = False
            for param in self.net2.parameters():
                param.requires_grad = False
    def forward(self,X,length):
        pred1=self.net1(X,length) #（B,L,1）
        pred2=self.net2(X,length)
        frac=torch.sigmoid(self.linear2(torch.tanh(self.linear1(self.net3(X,length)))))
        output=torch.add(torch.mul(pred1,frac),torch.mul(torch.exp(pred2)-1,1-frac)) #仍然是(B,L,1)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len = 4028):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len,d_model) #
        position = torch.arange(0 , max_len , dtype= torch.float).unsqueeze(1) #(1,L)
        div_term = torch.exp(torch.arange(0 , d_model ,2).float() * (-math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:,:x.size(1),:]
        return x
def length_to_mask(lengths,max_len):
    mask = torch.arange(max_len,device=lengths.device).expand(lengths.shape[0], max_len) < lengths
    return mask
class Transformer(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_size=1,dim_feedforward = 512,num_head = 4,num_layers = 4,dropout=0.1,max_len = 4028,activation:str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.linear = nn.Linear(input_dim, hidden_dim,bias=False)
        self.position_embedding = PositionalEncoding(hidden_dim,max_len)
        # 编码层
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim,num_head,dim_feedforward,dropout,activation,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers)
        # 输出层
        self.output = nn.Linear(hidden_dim,output_size)
        self.relu=nn.ReLU()
    def forward(self,inputs,lengths):
        hidden_states=self.linear(inputs)# (B,L,H)
        hidden_states = self.position_embedding(hidden_states) #(B,L,H)--->(B,L,H)

        #max_len=torch.tensor(inputs.shape[1]) #L
        # attention_mask = length_to_mask(lengths,max_len) == False
        # hidden_states = self.transformer(hidden_states,src_key_padding_mask = attention_mask)
        hidden_states = self.transformer(hidden_states)
        output = self.relu(self.output(hidden_states))

        return output
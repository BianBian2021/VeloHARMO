from torch.utils.data import Dataset
from transformers import BertTokenizer
from enum import Enum
import os
from . import utils
import numpy as np
import torch

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
    #print(codon2idx)
    onehot_nt = {nts[i]: np.eye(4)[i] for i in range(len(nts))}
    onehot_codon = {codon: np.eye(64)[codon2idx[codon]] for codon in codon2idx} # 4^3=64种密码子，one-hot编码
    onehot_aa = {aa_table[i, 2]: np.eye(21)[i] for i in range(len(aa_table))}
    onehot_ss={SS[i]: np.eye(len(SS))[i] for i in range(len(SS))}

    #读取文件 #load_data
    with open(path, "r") as f:
        data = f.readlines()
    assert len(data) % 4 == 0
    list_name = []
    list_seq = []
    list_speed = []
    list_pro_feature = []  # add protein feature1
    list_density = []
    list_ss=[]
    list_avg = []
    for i in range(len(data) // 4):
        name = data[4 * i + 0].split('>')[1].split()[0]
        seq = data[4 * i + 1].split()   # coden sequence
        count = [float(e) for e in data[4 * i + 2].split()] # velocity
        pro_feature = [list(map(float, e.split(",")[:-1])) for e in data[4 * i + 3].split()]# add protein feature
        ss=[e.split(",")[-1] for e in data[4 * i + 3].split()]  # structure
        avg = np.mean(np.array(count)[np.array(count) > 0.5])
        density = ((np.array(count) > 0.5) * np.array(count) / avg).tolist()

        # criteria containing ribosome density AND coverage percentage
        list_name.append(name)
        list_seq.append(seq)
        list_speed.append(count)
        list_density.append(density)
        list_pro_feature.append(pro_feature)  # add protein feature3 #1d=per gene, 2d=each codon, 3d=each feature
        list_ss.append(ss)
        list_avg.append(avg)
    
    list_name = np.array(list_name,dtype=object)
    list_pro_feature = np.array(list_pro_feature,dtype=object)  # add protein feature4
    list_ss=np.array(list_ss,dtype=object)
    list_seq = np.array(list_seq,dtype=object)
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
    
    #print(nts) # ATCG
    
    for i in range(len(list_gene_all)):
        codons = list_seq[i]
        nts = "".join(codons)
        #print(nts)
        if "N" in nts:
            continue
        aas = [codon2aa[codon] for codon in codons]
        index.append(i)
        dict_seq[list_gene_all[i]] = nts
        dict_seq_nt[list_gene_all[i]] = np.array([onehot_nt[nt] for nt in nts]) # 核苷酸one-hot编码
        dict_seq_codon[list_gene_all[i]] = np.array([onehot_codon[codon] for codon in codons])
        dict_seq_aa[list_gene_all[i]] = np.array([onehot_aa[aa] for aa in aas])
        dict_density[list_gene_all[i]] = list_density[i]
        dict_ss[list_gene_all[i]] = np.array([onehot_ss[structure] for structure in list_ss[i]])
        dict_pro_feature[list_gene_all[i]] = list_pro_feature[i]  # add protein feature6
        dict_avg[list_gene_all[i]] = list_avg[i]
    

    # print(f'codons: {len(codons)}') # 642
    # print(f"dict_seq:{len(dict_seq)}") # below are all 1170
    # print(f"dict_seq_nt:{len(dict_seq_nt)}")
    # print(f"dict_seq_codon:{len(dict_seq_codon)}")
    # print(f"dict_seq_aa:{len(dict_seq_aa)}")
    # print(f"dict_density:{len(dict_density)}")
    # print(f"dict_ss:{len(dict_ss)}")
    # print(f"dict_pro_feature:{len(dict_pro_feature)}")
    # print(f"dict_avg:{len(dict_avg)}")

    # for k, v in dict_seq_codon.items():  # 每个codon的维度=64
    #     print(v.shape)
    #将读取的数据变为x,y,valid_len的形式 #get_data_pack
    # n=len(dict_seq_codon)
    # x_input = [dict_seq_codon[gene] for gene in dict_seq_codon] # list of ndarray # (N, 64)
    # length = [ torch.tensor([len(dict_seq_codon[gene])])for gene in dict_seq_codon] # 所有基因序列的长度
    # cnt = 0
    # for leng in length:
    #     if leng > 510:
    #         cnt += 1
    # print(f"{cnt} seqs > 510")
    # for gene in dict_seq_nt:
    #     print(len(dict_seq_nt[gene]))
    # if nt:  # 12d
    #     for gene in dict_seq_nt:
    #         print(type(dict_seq_nt[gene]))
    #     x_nt = torch.tensor([dict_seq_nt[gene] for gene in dict_seq_nt])
    #     # print('x_size: ', x_input[0].shape)
    # if aa:  # 21d
    #     x_aa = torch.tensor([dict_seq_aa[gene] for gene in dict_seq_aa])
    #     #x_input = [np.concatenate([x_input[i], x_aa[i]], axis=1) for i in range(n)] # (N, 97)
    #     # print('x_size: ', x_input[0].shape)
    # if protein:  # add protein feature7　6d
    #     x_pro_feature = torch.tensor([dict_pro_feature[gene] for gene in dict_pro_feature])
    #     #x_input = [np.concatenate([x_input[i], x_pro_feature[i]], axis=1) for i in range(n)]    # (N, 103)
    #     # print('x_size: ', x_input[0].shape)
    # if add_ss:  # 8d
    #     x_ss = torch.tensor([dict_ss[gene] for gene in dict_ss])
    #     #x_input = [np.concatenate([x_input[i], x_ss[i]], axis=1) for i in range(n)] # (N, 111)
    #     # print('x_size: ', x_input[0].shape)
    # for k, v in dict_pro_feature.items():
    #     print(type(v))
    return list_seq, list_density, list_gene_all, dict_seq_nt, dict_seq_aa, dict_pro_feature, dict_ss


class Split(Enum):
    train = "train"
    dev = "val"
    test = "test"




class GeneSequenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        # if isinstance(mode, Split):
        #     mode = mode.value
        # file_path = os.path.join(data_dir, f"{mode}.txt")
        sequences, speed, list_gene_all, dict_seq_nt, dict_seq_aa, dict_pro_feature, dict_ss= load_data(file_path)
        self.features = convert_examples_to_features(sequences, speed, list_gene_all, dict_seq_nt, dict_seq_aa, 
                                                     dict_pro_feature, dict_ss, tokenizer, max_seq_length)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        return features


def convert_examples_to_features(
    sequences, 
    speed, 
    list_gene_all,
    dict_seq_nt, 
    dict_seq_aa, 
    dict_pro_feature, 
    dict_ss,
    tokenizer, 
    max_seq_length,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True
    ):
    features = []
    
    for i, (tokens, labels) in enumerate(zip(sequences, speed)):
        #print(f"gene: {list_gene_all[i]} \n tokens: {tokens}")
        # print(tokens) list codon
        special_tokens_count = 2
        # Truncate or pad the tokenized sequence and speeds list
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            labels = labels[:(max_seq_length - special_tokens_count)]
        
        # if dict_seq_nt is not None:
        #     x_nt = dict_seq_nt[list_gene_all[i]]
        #     print('x_nt: ', len(x_nt), ' token: ', len(tokens))
        
        
        if dict_seq_aa is not None:
            x_aa = dict_seq_aa[list_gene_all[i]][:(max_seq_length - special_tokens_count)].tolist()
            # print('x_aa: ', type(x_aa), ' token: ', len(tokens))
          
        if dict_pro_feature is not None:
            x_pro_feature = dict_pro_feature[list_gene_all[i]][:(max_seq_length - special_tokens_count)]
            # print('x_pro_feature: ', type(x_pro_feature), ' token: ', len(tokens))
            
        if dict_ss is not None:
            x_ss = dict_ss[list_gene_all[i]][:(max_seq_length - special_tokens_count)].tolist()
            # print('x_ss: ',type(x_ss), ' token: ', len(tokens))
            
        tokens += [sep_token]
        labels += [pad_token_label_id]
        
        aa_dimension = len(x_aa[0])
        pro_dimension = len(x_pro_feature[0])
        ss_dimension = len(x_ss[0])
        
        x_aa += [[pad_token_label_id] * aa_dimension]
        x_pro_feature += [[pad_token_label_id] * pro_dimension]
        x_ss += [[pad_token_label_id] * ss_dimension]
        
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # print("len label: ", len(labels))
        if cls_token_at_end:
            tokens += [cls_token]
            labels += [pad_token_label_id]
            
            x_aa += [[pad_token_label_id] * aa_dimension]
            x_pro_feature += [[pad_token_label_id]  * pro_dimension]
            x_ss += [[pad_token_label_id] * ss_dimension]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            labels = [pad_token_label_id] + labels
            x_aa = [[pad_token_label_id] * aa_dimension] + x_aa
            x_pro_feature = [[pad_token_label_id] * pro_dimension] + x_pro_feature
            x_ss = [[pad_token_label_id] * ss_dimension] + x_ss
            segment_ids = [cls_token_segment_id] + segment_ids

        # print("tokens: ", tokens) 
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #print("input_ids: ", input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            labels = ([pad_token_label_id] * padding_length) + labels
            x_aa = ([pad_token_label_id] * aa_dimension) * padding_length + x_aa
            x_pro_feature = ([pad_token_label_id] * pro_dimension) * padding_length + x_pro_feature
            x_ss = ([pad_token_label_id] * ss_dimension) * padding_length + x_ss
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            labels += [pad_token_label_id] * padding_length
            x_aa += [[pad_token_label_id] * aa_dimension] * padding_length
            x_pro_feature += [[pad_token_label_id] * pro_dimension] * padding_length
            x_ss += [[pad_token_label_id] * ss_dimension] * padding_length


        valid_sequence = (torch.tensor(labels) != -100).float()
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length
        assert len(x_aa) == max_seq_length
        assert len(x_pro_feature) == max_seq_length
        assert len(x_ss) == max_seq_length
        
        # print('x_aa: ', x_aa)
        # print('x_pro_feature: ', x_pro_feature)
        # print('x_ss: ', x_ss)
        # print('labels:', labels)
        features.append({
            'input_ids': torch.tensor(input_ids), 
            'attention_mask': torch.tensor(input_mask), 
            'token_type_ids': torch.tensor(segment_ids), 
            'density': torch.tensor(labels), 
            'valid_sequence': valid_sequence.clone().detach().requires_grad_(False),
            'aa': torch.tensor(x_aa),
            'pro_feature': torch.tensor(x_pro_feature),
            'ss': torch.tensor(x_ss),
        })
        # print('input_ids: ', type(input_ids))
        # print('attention_mask: ', type(input_mask))
        # print('token_type_ids: ', type(segment_ids))
        # print('density: ', type(labels))
        # print('valid_sequence:', valid_sequence)
    return features


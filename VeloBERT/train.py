import argparse
import glob
import json
import logging
import os
import re
import shutil
import random
from matplotlib import pyplot as plt
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy
import csv
import numpy as np
from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import utils
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    BertConfig,
    BertTokenizer,
    BertModel,
)
from dataset.dataset import GeneSequenceDataset, Split

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class MaskedHuberLoss(nn.HuberLoss):
    def forward(self, pred, label, valid_sequence):
       # print('pred ', pred.size(), ' label', label.size(), ' valid: ', valid_sequence.size())
        weights = torch.ones_like(label)
        weights = weights.float() * valid_sequence
        self.reduction = 'none'
        unweighted_loss = super(MaskedHuberLoss, self).forward(pred, label)
        weighted_loss = (unweighted_loss * weights).mean()
        
        return weighted_loss


class CodenBert(nn.Module):
    def __init__(self, bert_model, num_labels, additional_feature_dim, args):
        super(CodenBert, self).__init__()
        self.bert = bert_model
        self.args = args
        self.hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Linear(self.hidden_size, num_labels)
        self.additional_feature_layer = nn.Linear(additional_feature_dim, self.hidden_size)
        self.down_project = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids, additional_features):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        sequence_output = outputs[0]     # (B, N, 768)
        
        if self.args.use_additional_feature:
            additional_features_output = self.additional_feature_layer(additional_features) # (B, N, 768)
            fused_output = torch.concatenate([sequence_output, additional_features_output], dim=-1)
            fused_output = self.down_project(fused_output)
            sequence_output = self.relu(fused_output)
        
        prediction_scores = self.regressor(sequence_output)
        prediction_scores = self.relu(prediction_scores)
        return prediction_scores


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def regression(y_pred, y_true):
    # print(y_pred.shape)
    # print(y_true.shape)

    linear_model = LinearRegression()
    x_train = np.array(y_pred.reshape(-1, 1))
    y_true = np.array(y_true.reshape(-1, 1))
    # print(x_train.shape)
    linear_model.fit(x_train, y_true)
    y_train_pred = linear_model.predict(x_train)
    score = linear_model.score(x_train, y_true)  # r2 score
    return score


def plot_correlation(y1, y2, fold, score):
    y1 = np.array(y1)
    y2 = np.array(y2)
    model = LinearRegression()
    x_train = np.array(y1.reshape(-1, 1))
    model.fit(x_train, y2)

    b = model.intercept_  # 截距
    a = model.coef_[0]
    y_train_pred = model.predict(x_train)
    plt.plot(x_train, y_train_pred, color='red', linewidth=3)
    #score = model.score(x_train, y2)  # r2 score
    #print('score: ', score)
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
    plt.savefig(f'{fold}_relationship_result.jpg', bbox_inches='tight', pad_inches=0.2)

    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴。
    plt.close()  # 关窗口


def train_epoch(model, train_loader, optimizer, loss, args):
    model.train()
    device = args.device
    pbar = tqdm(total=len(train_loader), leave=False, desc='train')

    loss_list = []
    r2_list = []
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        density = batch['density'].to(device)
        valid_sequence = batch['valid_sequence'].to(device)
        aa = batch['aa'].to(device) # (B, N, 21)
        pro_feature = batch['pro_feature'].to(device) # (B, N, 6)
        ss = batch['ss'].to(device) # (B, N, 8)
        
        if not args.use_additional_feature:
            additional_features = None
        else:
            features = []
            if args.aa:
                features.append(aa)
            if args.protein:
                features.append(pro_feature)
            if args.ss:
                features.append(ss)

            additional_features = torch.cat(features, dim=-1)  # 在最后一个维度上拼接
        #print('density in train loop: ', density)
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                        additional_features=additional_features)


        outputs = outputs.squeeze(-1)

        tr_loss = loss(outputs, density, valid_sequence)

        selected_y_pred = outputs[valid_sequence == 1]
        selected_y_true = density[valid_sequence == 1]
        y_pred, y_true = selected_y_pred.detach().cpu().numpy(), selected_y_true.detach().cpu().numpy()

        r2 = regression(y_pred, y_true)
        r2_list.append(r2)
        # Backward pass
        tr_loss.backward() 
        # Optimize
        optimizer.step()
        # Append the batch loss to the loss list
        loss_list.append(tr_loss.item())

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    avg_loss = sum(loss_list) / len(loss_list)
    r2_score = sum(r2_list) / len(r2_list)
    return avg_loss, r2_score


def evaluate(model, eval_loader, loss, args, max_val_v, fold):
    model.eval()
    device = args.device
    pbar = tqdm(total=len(eval_loader), leave=False, desc='eval')
    loss_list = []
    r2_list = []
    trues = []
    preds = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            density = batch['density'].to(device)
            valid_sequence = batch['valid_sequence'].to(device)
            
            aa = batch['aa'].to(device) # (B, N, 21)
            pro_feature = batch['pro_feature'].to(device) # (B, N, 6)
            ss = batch['ss'].to(device) # (B, N, 8)
            
            if not args.use_additional_feature:
                additional_features = None
            else:
                features = []
                if args.aa:
                    features.append(aa)
                if args.protein:
                    features.append(pro_feature)
                if args.ss:
                    features.append(ss)

                additional_features = torch.cat(features, dim=-1)  # 在最后一个维度上拼接
            

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           additional_features=additional_features)
            outputs = outputs.squeeze(-1)

            eval_loss = loss(outputs, density, valid_sequence)

            selected_y_pred = outputs[valid_sequence == 1]
            selected_y_true = density[valid_sequence == 1]
            y_pred, y_true = selected_y_pred.detach().cpu().numpy(), selected_y_true.detach().cpu().numpy()

            trues.extend(y_true)
            preds.extend(y_pred)
            r2 = regression(y_pred, y_true)
            
            r2_list.append(r2)
            
            loss_list.append(eval_loss.item())

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    avg_loss = sum(loss_list) / len(loss_list)
    r2_score = sum(r2_list) / len(r2_list)
    if r2_score > max_val_v:
        plot_correlation(trues, preds, fold, r2_score)
    return avg_loss, r2_score
    

def train_and_validate(model, tokenizer, loss, optimizer, num_epoch, batch_size, lr_scheduler, train_loader, eval_loader, save_path, args, fold):
    epoch_max = num_epoch
    epoch_val = 1
    max_val_v = -1e8
    best_loss = 1e8
    best_eval_loss = 1e8
    # best_loss = 1.1492
    # max_val_v = 4.8238
    timer = utils.Timer()
    epoch_start = 1
    print(f'num_epoch: {num_epoch}')

    train_log_path = f'./save/train_log_{epoch_max}'
    eval_log_path = f'./save/eval_log_{epoch_max}'

    if args.aa:
        train_log_path += '_aa'
        eval_log_path += '_aa'
    if args.protein:
        train_log_path += '_protein'
        eval_log_path += '_protein'
    if args.ss:
        train_log_path += '_ss'
        eval_log_path += '_ss'
    
    train_log_path += '.csv'
    eval_log_path += '.csv'

    for epoch in range(1, num_epoch + 1):
        print(f"epoch: {epoch}")
        t_epoch_start = timer.t()
        
        train_loss, r2 = train_epoch(model, train_loader, optimizer, loss, args)
        print(f'train epoch: {epoch}: loss= {train_loss}, r2= {r2}')
        lr_scheduler.step()

       
        with open(train_log_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, train_loss, r2])
            f.close()
            
        logger.info('epoch {}/{}'.format(epoch, epoch_max))
        logger.info('train G: loss={:.4f}'.format(train_loss))

        save(model, tokenizer, save_path, 'last')
        
        if train_loss < best_loss:
            best_loss = train_loss
            save(model, tokenizer, save_path, 'best_loss')
        
        if epoch_val is not None and epoch % epoch_val == 0:
            torch.cuda.empty_cache()
            
            eval_loss, r2 = evaluate(model, eval_loader, loss, args, max_val_v, fold)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save(model, tokenizer, save_path, 'best_eval_loss')
                
            print(f'eval epoch: {epoch}, loss= {eval_loss}, r2= {r2}')
            if r2 > max_val_v:
                max_val_v = r2
                save(model, tokenizer, save_path, 'best')
            
            with open(eval_log_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch, eval_loss, r2])
                f.close()
            
            logger.info('epoch {}/{}'.format(epoch, epoch_max))
            logger.info('eval: loss={:.4f}'.format(eval_loss))

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            logger.info('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
    
    return max_val_v

def save(model, tokenizer, save_path, name):
    save_path = save_path + '/' + name
    os.makedirs(save_path, exist_ok=True)
    torch.save({'model_state': model.state_dict()}, os.path.join(save_path, "pytorch_model.pth"))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
  
    parser.add_argument(
        "--n_process",
        default=2,
        type=int,
        help="number of processes used for data process",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list:",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    
    # Other parameters
    parser.add_argument(
        "--visualize_data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        help="The directory where the dna690 and mouse will save results.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--predict_dir",
        default=None,
        type=str,
        help="The output directory of predicted result. (when do_predict)",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_visualize", action="store_true", help="Whether to calculate attention score.")
    parser.add_argument("--visualize_train", action="store_true", help="Whether to visualize train.tsv or dev.tsv.")
    parser.add_argument("--do_ensemble_pred", action="store_true", help="Whether to do ensemble prediction with kmer 3456.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--per_gpu_pred_batch_size", default=8, type=int, help="Batch size per GPU/CPU for prediction.",
    )
    parser.add_argument(
        "--early_stop", default=0, type=int, help="set this to a positive integet if you want to perfrom early stop. The model will stop \
                                                    if the auc keep decreasing early_stop times",
    )
    parser.add_argument(
        "--predict_scan_size",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    # parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float, help="Dropout rate of attention.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn_dropout", default=0.0, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn", default="lstm", type=str, help="What kind of RNN to use")
    parser.add_argument("--num_rnn_layer", default=2, type=int, help="Number of rnn layers in dnalong model.")
    parser.add_argument("--rnn_hidden", default=768, type=int, help="Number of hidden unit in a rnn layer.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")
    parser.add_argument("--batch_size", type=int, default=16, help="training batch size")


    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--visualize_models", type=int, default=None, help="The model used to do visualization. If None, use 3456.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    
    parser.add_argument("--aa", action="store_true", help="Whether to use amino acid feature.")
    parser.add_argument("--protein", action="store_true", help="Whether to use protein feature")
    parser.add_argument("--ss", action="store_true", help="Whether to use protein structure feature.")

    args = parser.parse_args()
    
    if args.aa or args.protein or args.ss:
        args.use_additional_feature = True
    else:
        args.use_additional_feature = False


    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    log_file_path = './save/train.log'

    # 检查文件是否存在，如果不存在则创建
    if not os.path.exists(log_file_path):
        # 创建目录（如果目录不存在）
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # 创建文件
        open(log_file_path, 'w').close()

    # Setup logging
    logging.basicConfig(
        filename=log_file_path,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    max_seq_length = args.max_seq_length


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    logger.info("Training/evaluation parameters %s", args)
    
    batch_size = args.batch_size
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # eval_loader = DataLoader(eval_dataset, batch_size=batch_size)


    # # layers = list(bert_model.modules())
    # # print(layers)
    # lr = 5e-5
    # lr_min = 1.0e-7
    # # 设置AdamW优化器
    # optimizer = AdamW(model.parameters(), lr=lr)

    # num_epoch = args.num_train_epochs
    # # 设置余弦退火学习率调度器
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    # loss = MaskedHuberLoss()
    # # 保存路径
    # save_path = './save'

    # train_and_validate(model, tokenizer, loss, optimizer, num_epoch, batch_size, lr_scheduler, train_loader, eval_loader, save_path, device)

    # TODO : data_dir
    base_path = '/root/autodl-tmp/CodenBert/codenbert/dataset/ten_fold'
    scores = []
    best_result_path = './save/best_results'

    additional_feature_dim = 0
        
    if args.aa:
        additional_feature_dim += 21
        best_result_path += '_aa'
    if args.protein:
        additional_feature_dim += 6
        best_result_path += '_protein'
    if args.ss:
        additional_feature_dim += 8
        best_result_path += '_ss'

    best_result_path += '.csv'
    
    for i in range(10):
        # 读取训练和测试数据
        print(f"Fold: {i+1}")

        max_seq_length = args.max_seq_length
        # TODO : model_path

        model_path = '/root/autodl-tmp/CodenBert/DNABERT/3-new-12w-0'
        tokenizer_path = '/root/autodl-tmp/CodenBert/DNABERT/3-new-12w-0'
        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=1,
            cache_dir=args.cache_dir,
            output_hidden_states=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=args.cache_dir,
        )
        bert_model = BertModel.from_pretrained(
            model_path,
            config=config,
            cache_dir=args.cache_dir,
        )
        
        model = CodenBert(bert_model, num_labels=1, additional_feature_dim=additional_feature_dim, args=args).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        logger.info('finish loading model')
        train_file = os.path.join(base_path, f"E.coli.summary.add.pro.SS_{i}_train.txt")
        test_file = os.path.join(base_path, f"E.coli.summary.add.pro.SS_{i}_test.txt")
        
        train_dataset = ( 
            GeneSequenceDataset(
                train_file,
                tokenizer,
                max_seq_length,
                )
            if args.do_train
            else None
        )

        test_dataset = ( 
            GeneSequenceDataset(
                test_file,
                tokenizer,
                max_seq_length,
                )
            if args.do_eval
            else None
        )
        
        print("train dataset: ", len(train_dataset))
        print("test dataset: ", len(test_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)

        lr = 5e-5
        lr_min = 1.0e-7
        # 设置AdamW优化器
        optimizer = AdamW(model.parameters(), lr=lr)

        num_epoch = args.num_train_epochs
        # 设置余弦退火学习率调度器
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

        loss = MaskedHuberLoss()

        save_path = f'./save/_{i+1}'
        score = train_and_validate(model, tokenizer, loss, optimizer, num_epoch, batch_size, lr_scheduler, train_loader, test_loader, save_path, args, fold=i+1)
        
        
        with open(best_result_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([i+1, score])
            f.close()
        # 保存每折的分数
        scores.append(score)
    
    print("num of folds: ", len(scores))
    # 计算平均分数
    average_score = sum(scores) / len(scores)
    print(f"Average score across 10 folds: {average_score}")
    logger.info(f"Average score across 10 folds: {average_score}")


if __name__ == "__main__":
    main()
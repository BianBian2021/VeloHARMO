import argparse
import VeloGRU_data
import VeloGRU_train
import os
import logging
logging.basicConfig(level=logging.INFO)
def main(args):
    if args.command == 'split':
        VeloGRU_data.split_data(args.datafile, int(args.num_fold))
    elif args.command == 'train':
        VeloGRU_train.train(args.datafile,args.nt,args.aa,args.protein,args.ss)
    elif args.command == 'test':
        VeloGRU_train.test(args.model, args.data,args.nt,args.aa,args.protein,args.ss,args.exclude_zeros,args.gene_filter,args.name)
    elif args.command == 'cross_test':
        datafile=args.datafile
        data_file_name = os.path.splitext(datafile)[0]
        num_fold=int(args.num_fold)
        VeloGRU_data.split_data(args.datafile, num_fold)
        pearsons=[]
        r2_scores=[]
        stds=[]
        for i in range(num_fold):
            fold_train="{}_{}_train.txt".format(data_file_name,i)
            fold_test="{}_{}_test.txt".format(data_file_name,i)
            name="fold_{}".format(i)
            model_name=name+"_best_network.pth"
            VeloGRU_train.train(fold_train,args.nt,args.aa,args.protein,args.ss,name=name)
            pearson,std,score=VeloGRU_train.test(model_name, fold_test, args.nt, args.aa, args.protein, args.ss, args.exclude_zeros,
                            args.gene_filter, name)
            pearsons.append(pearson)
            r2_scores.append(score)
            stds.append(std)
        for i in range(num_fold):
            logging.info("In fold {}, pearson:{}±{},r2 score:{}".format(i, pearsons[i],stds[i],r2_scores[i]))
        logging.info("Across {} folds, average pearson:{} r2 score:{}".format(num_fold,sum(pearsons)/num_fold,sum(r2_scores)/num_fold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers.required = True
    subparsers.dest = 'command'

    parser_a = subparsers.add_parser('train', help='train help')
    parser_a.add_argument('datafile', help='fname1')
    parser_a.add_argument('--nt', action='store_true')
    parser_a.add_argument('--aa', action='store_true')
    parser_a.add_argument('--protein',action='store_true')
    parser_a.add_argument('--ss', action='store_true')

    parser_b = subparsers.add_parser('test', help='test help')
    parser_b.add_argument('model', help='torch model path')
    parser_b.add_argument('data', help='txt file for test')
    parser_b.add_argument('--nt', action='store_true')
    parser_b.add_argument('--aa', action='store_true')
    parser_b.add_argument('--protein', action='store_true')
    parser_b.add_argument('--ss', action='store_true')
    parser_b.add_argument('--exclude_zeros', action='store_true') #测试时是否评估真实值为零的位点
    parser_b.add_argument('--gene_filter', default="0.6") #测试时只挑选非零位点比例不低于该值的基因
    parser_b.add_argument('--name', default="") #指定测试中画图的名字“test_name_thres”


    parser_c = subparsers.add_parser('split', help='split help')
    parser_c.add_argument('datafile', help='full datafile including train and test')
    parser_c.add_argument('num_fold', help='k-fold splits')

    parser_d = subparsers.add_parser('cross_test', help='use one file for k-fold cross test')
    parser_d.add_argument('datafile', help='full datafile including train and test')
    parser_d.add_argument('num_fold', help='k-fold splits')
    parser_d.add_argument('--nt', action='store_true')
    parser_d.add_argument('--aa', action='store_true')
    parser_d.add_argument('--protein', action='store_true')
    parser_d.add_argument('--ss', action='store_true')
    parser_d.add_argument('--exclude_zeros', action='store_true')  # 测试时是否评估真实值为零的位点
    parser_d.add_argument('--showfig', action='store_true')  # 是否对
    parser_d.add_argument('--gene_filter', default="0.6")  # 测试时只挑选非零位点比例不低于该值的基因
    #python run.py cross_test "filename" "k" --nt --aa --ss --protein --exclude_zerospython run.py cross_test "filename" "k" --nt --aa --ss --protein --exclude_zeros

    args = parser.parse_args()
    main(args)




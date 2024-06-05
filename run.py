import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import sys


class Logger(object):
    def __init__(self, filename='./log/default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--flag', type=str, default='5',
                        help='2: 将离散数据合并为新特征 3：新特征+新hum+新windspeed+删winddir 4：新特征+新Hum+新WS')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='GrainTemp_48_12', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer,DLinear, ETSformer, FEDformer, Informer, LightTS, Pyraformer, Reformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='GrainTemp', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/GrainTemp/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Temp_1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',  # 单变量或者多变量预测
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='3h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # ！！！！！！！！！！freq
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task 预测任务
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')  # 输入序列长度
    parser.add_argument('--label_len', type=int, default=24, help='start token length')  # 标签序列长度
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')  # 输出序列长度
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task 输入任务
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task 异常检测任务
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model 512')  # 输入模型数据大小
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn 2048')  # 全连接层大小
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')  # 移动平均线的窗口大小
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',  # 是否在编码器中使用蒸馏，使用该参数表示不使用蒸馏
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding,为timeF时timeenc=1，else=0, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')  # 原默认为10 epoch！！！！ 实验过程设为100
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')  # 提前终止
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate 0.000')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')  # 原默认为test
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params 去静态投影仪参数
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # PatchTST
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser_0 = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Wheat': {'data': 'Wheat_enc.csv', 'T': 'AveragePrice', 'M': [49, 49, 49], 'S': [1, 1, 1], 'MS': [49, 49, 1]},
        'beijing': {'data': 'beijing.csv', 'T': 'TEMP', 'M': [5, 5, 5], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
        'RYF': {'data': 'RYF.csv', 'T': 'Solar', 'M': [9, 9, 9], 'S': [1, 1, 1], 'MS': [9, 9, 1]},
        # 'GrainTemp': {'data': 'Temp_1.csv', 'T': 'Temp', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1]},
        'GrainTemp': {'data': 'Temp_2.csv', 'T': 'Temp', 'M': [9, 9, 9], 'S': [1, 1, 1], 'MS': [9, 9, 1]},
        # 'GrainTemp': {'data': 'Temp_3.csv', 'T': 'Temp', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [8, 8, 1]},
        'm4': {'data': 'm4', 'T': 'OT', 'M': [1, 1, 1], 'S': [1, 1, 1], 'MS': [1, 1, 1]},
        # 如果有自己的数据就在这里面做
        # [7,7,1] 编码层 解码层 输出层 的特征数
        # 数据名 T标签列名 M, S, MS M：多变量预测多变量，S：单变量预测单变量，MS：多变量预测单变量',预测未来12个点
    }
    if args.data in data_parser_0.keys():
        data_info = data_parser_0[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    # 将控制台输出保存到log中
    log_name = './log/'+'{}_{}_{}_epoch{}_bs{}_ft{}_dm{}_nh{}_el{}_dl{}_dff{}_lr{}_fc{}_eb{}_dt{}_{}'.format(
        args.flag,
        args.model_id,
        args.model,
        args.train_epochs,
        args.batch_size,
        args.features,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.learning_rate,
        args.factor,
        args.embed,
        args.distil,
        args.des)
    sys.stdout = Logger(log_name+'.log', sys.stdout)

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.flag,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.flag,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='arguments for daisy')
    # tuner settings
    parser.add_argument('--optimization_metric', 
                        type=str, 
                        default='ndcg', 
                        help='the metric to be optimized for hyper-parameter tuning via HyperOpt')
    parser.add_argument('--hyperopt_trail', 
                        type=int, 
                        default=30, 
                        help='the number of trails of HyperOpt')
    parser.add_argument('--hyperopt_pack', 
                        type=str, 
                        default='{}', 
                        help='record the searching space of hyper-parameters for HyperOpt')
    # common settings
    parser.add_argument('--algo_name', 
                        type=str, 
                        help='algorithm to choose')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='10filter', 
                        help='dataset preprocess op.: origin/Nfilter/Ncore')
    parser.add_argument('--topk', 
                        type=int, 
                        default=50, 
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='tsbr', 
                        help='method for split test,options: tsbr/rsbr/tloo/rloo')
    parser.add_argument('--val_method', 
                        type=str, 
                        default='tsbr', 
                        help='validation method, options: tsbr/rsbr/tloo/rloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        default=.2, 
                        help='split ratio for test set')
    parser.add_argument('--val_size', 
                        type=float, 
                        default=.1, help='split ratio for validation set')
    parser.add_argument('--fold_num', 
                        type=int, 
                        default=5, 
                        help='No. of folds for cross-validation')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, 
                        help='the number of candidate items used for ranking')
    parser.add_argument('--sample_method', 
                        type=str, 
                        default='uniform', 
                        help='negative sampling method mixed with uniform, options: high-pop, low-pop')
    parser.add_argument('--sample_ratio', 
                        type=float, 
                        default=0, 
                        help='control the ratio of popularity sampling for the hybrid sampling strategy in the range of (0,1)')
    parser.add_argument('--init_method', 
                        type=str, 
                        default='default', 
                        help='weight initialization method: normal, uniform, xavier_normal, xavier_uniform')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=0, 
                        help='negative sampling number')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='CL', 
                        help='loss function type')
    parser.add_argument('--optimizer', 
                        type=str, 
                        default='default', 
                        help='optimize method')
    # algo settings
    parser.add_argument('--factors', 
                        type=int, 
                        default=32, 
                        help='latent factors numbers in the model')
    parser.add_argument('--reg_1', 
                        type=float, 
                        default=0., 
                        help='L1 regularization')
    parser.add_argument('--reg_2', 
                        type=float, 
                        default=0.001, 
                        help='L2 regularization')
    parser.add_argument('--kl_reg', 
                        type=float, 
                        default=0.5, 
                        help='VAE KL regularization')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.02, 
                        help='constant to multiply the penalty terms for SLIM')
    parser.add_argument('--elastic', 
                        type=float, 
                        default=0.5, 
                        help='the ElasticNet mixing parameter for SLIM in the range of (0,1)')
    parser.add_argument('--maxk', 
                        type=int, 
                        default=40, 
                        help='The (max) number of neighbors to take into account')
    parser.add_argument('--node_dropout_flag', 
                        type=int, 
                        default=1,
                        help='NGCF: 0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout',
                        type=float,  
                        nargs='?', 
                        default=0.1,
                        help='NGCF: Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', 
                        type=float, 
                        nargs='?', 
                        default=0.1,
                        help='NGCF: Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='batch size for training')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=2, 
                        help='number of layers in MLP model')
    parser.add_argument('--act_func', 
                        type=str, 
                        default='relu', 
                        help='activation method in interio layers')
    parser.add_argument('--out_func', 
                        type=str, 
                        default='sigmoid', 
                        help='activation method in output layers')
    parser.add_argument('--no_batch_norm', 
                        action='store_false', 
                        default=True, 
                        help='whether do batch normalization in interior layers')
    parser.add_argument("--tune_testset",
                        action="store_true",
                        default=False,
                        help="whether to directly tune on testset")
    parser.add_argument("--early_stop",
                        action="store_false",
                        default=True,
                        help="whether to activate early stop mechanism")
    args = parser.parse_args()

    return args

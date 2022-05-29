import argparse


def baseargs():
    parser = argparse.ArgumentParser()
    
    # args for path
    parser.add_argument('--data_dir', default='./data',help='train/dev/test data dir')
    parser.add_argument('--save_model_dir', default='./save_model',help='save model')
    parser.add_argument('--test_res_dir', default='./res',help='the test result output')
    
    # train setting
    parser.add_argument('--train_epochs', default=10, type=int, help='Max training epoch')
    
    parser.add_argument('--max_len', type=int, default=35)
    
    parser.add_argument('--class_num', type=int)
    
    parser.add_argument('--train_batch_size', type=int, default=60)
    parser.add_argument('--dev_batch_size', type=int, default=60)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    return args
    
# if __name__ == "__main__":
    
#     args = baseargs()
#     print(args.lr)

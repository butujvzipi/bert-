import torch
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel
from tqdm import tqdm
from data_process import read_data
from dataset_utils import BertDataset
from options import baseargs


def test(args):
    
    for root, dirs, files in os.walk("/home/user/tmp1/save_model"):
        model_list = files   # ['model_100.pt', 'model_340.pt', 'model_200.pt']

    for mpt in model_list:    # 遍历每个保存的模型
        
        model = torch.load(os.path.join("save_model",mpt))
        

        test_texts, test_labels = read_data(args.data_dir,"test",300)
        
        tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")

        test_dataset = BertDataset(test_texts, test_labels, 35,tokenizer)
        
        test_dataloader = DataLoader(test_dataset, 3, shuffle=False, collate_fn=test_dataset.pro_batch_data)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # model = model.to(device)
        result = []
        for text,label,batch_len in tqdm(test_dataloader):
            text = text.to(device)
            pre = model(text,batch_len)
            result.extend(pre)
        with open(os.path.join(args.test_res_dir,mpt+"_"+"test_result.txt"),"w",encoding="utf-8") as f:
            f.write("\n".join([str(i) for i in result]))
        test_acc = sum([i == int(j) for i,j in zip(result,test_labels)]) / len(test_labels)
        print(f"test acc = {test_acc * 100:.2f} % ")
        print("test over")
    
if __name__ == "__main__":
    args = baseargs()
    test(args)
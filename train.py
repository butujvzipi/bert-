import torch
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel
from tqdm import tqdm
from data_process import read_data
from dataset_utils import BertDataset
from model_utils import BertClassifier
from options import baseargs

def train(args):
    
    train_texts, train_labels = read_data(args.data_dir,"train",2000)
    
    assert len(train_texts) == len(train_labels)
    
    tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.class_num = len(set(train_labels))
    
    train_dataset = BertDataset(train_texts, train_labels, args.max_len, tokenizer)
    
    model = BertClassifier(args.class_num).to(device)
    
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, collate_fn=train_dataset.pro_batch_data)
    
    optim = torch.optim.AdamW(model.parameters(), lr = args.lr)
    
    global_step = 0
    early_stop = 0
    best_result = 0
    
    for e in range(args.train_epochs):
        print(f"epoch:{e}")
        model.train()
        for step,(batch_x,batch_label,batch_len) in tqdm(enumerate(train_dataloader)):
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            loss =  model.forward(batch_x,batch_len,batch_label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step % 30 == 0:
                 print('Train Epoch[{}] Step[{} / {}] - loss: {:.6f}  '.format(e+1, step+1, len(train_dataloader), loss))
        # result = evaluate(dev_dataset)
        # print(f"dev_acc:{result:.3f}% ")
            global_step += 1
            # 100个批次验证一次 or 达到训练轮次就验证
            if (global_step % 100 == 0) or (e==int(args.train_epochs)-1 and step == len(train_dataloader)-1): 
                early_stop += 1
                result = evaluate(args,model)
                print("best acc: %.2f, current acc: %.2f" % (best_result, result))
                if result >= best_result:
                    best_result = result
                    print("Saving model")
                    torch.save(model,os.path.join(args.save_model_dir,"model"+ "_" + str(global_step) + ".pt"))
                    early_stop = 0
                print()
                
            if early_stop >=10:  # 验证了10次还没有取得更好的结果就提前停止
                return 
                       

def evaluate(args,model):
    
    dev_texts, dev_labels = read_data(args.data_dir,"dev",400)
    assert len(dev_texts) == len(dev_labels)
    tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dev_dataset = BertDataset(dev_texts, dev_labels, args.max_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, args.dev_batch_size, shuffle=False, collate_fn=dev_dataset.pro_batch_data)
     
    model.eval()
    right = 0
    for step, (t_batch_x, t_batch_label, t_batch_len) in enumerate(dev_dataloader):
        t_batch_x = t_batch_x.to(device)
        t_batch_label = t_batch_label.to(device)
        pre = model(t_batch_x, t_batch_len)
        right += int(torch.sum(pre == t_batch_label))
    
    dev_acc = right / len(dev_texts) * 100
        
    return dev_acc
    
    # print(f"dev_acc:{right / len(dev_texts) * 100:.3f}% ")


if __name__ == "__main__":
    
    args = baseargs()
    
    train(args)
    
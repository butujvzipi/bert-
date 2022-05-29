import torch
import torch.nn as nn 
from transformers import BertTokenizer,BertModel

class BertClassifier(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert_base_chinese")
        self.classifier = nn.Linear(768,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

        s = 0
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            s += param.nelement() * param.element_size() / 1024 / 1024
        print(s)


    def forward(self,batch_x,batch_len,label=None):
        bert_out = self.bert.forward(batch_x,attention_mask=(batch_x>0))
        pre = self.classifier(bert_out[1])

        if label is not None:
            loss = self.loss_fun(pre,label)
            return loss 
        else :
            return torch.argmax(pre, dim=-1)
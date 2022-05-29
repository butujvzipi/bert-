from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn 

class BertDataset(Dataset):
    def __init__(self,texts,labels,max_len,tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.texts[index][:self.max_len]
        label= int(self.labels[index])
        t_len = len(text)

        return text,label,t_len

    def __len__(self):
        return len(self.labels)

    def pro_batch_data(self,batch_data):
        batch_text, batch_label, batch_len = zip(*batch_data)
        batch_embedding = []
        for text,label,len_ in zip(batch_text, batch_label, batch_len):

            text_embedding = self.tokenizer.encode(text,add_special_tokens=True,truncation=True,padding='max_length',max_length=self.max_len+2,return_tensors='pt')
            batch_embedding.append(text_embedding)
        batch_embedding = torch.cat(batch_embedding,dim=0)
        return batch_embedding,torch.tensor(batch_label),batch_len
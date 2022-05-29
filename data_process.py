import os

def read_data(data_dir,train_or_test,num=None):
    with open(os.path.join(data_dir,train_or_test + ".txt"),encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t,l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num == None:
        return texts,labels
    else:
        return texts[:num],labels[:num]
    
    
if __name__ == "__main__":
    a,b = read_data("test",200)
    print(len(a))
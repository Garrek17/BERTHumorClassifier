from torch.utils.data import Dataset
import torch
import csv

class TrainingData(Dataset):
    def __init__(self,
                 datatset,
                 embed,):
        self.data = csv.reader(open(datatset))
        #skip the header
        next(self.data)

        print("Initializing dataset...")

        self.funnyScore = [0,0]
        for row in self.data:
            if row[1] == "True":
                self.funnyScore.append(1)
            else:
                self.funnyScore.append(0)
        #Skip header embedding
        self.title_embed = torch.load(embed,
                                      map_location=torch.device('cpu'))
        print("Finished initializing dataset...")

    def __getitem__(self, index):
        X = self.title_embed[index]
        y = self.funnyScore[index]
        return X,y

    def __len__(self):
        return self.title_embed.size()[0]

dataset = TrainingData("Data/dataset.csv",
                       "Data/BERT_Embed_Dataset2.pth")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size,test_size],
                                                            generator=torch.Generator().manual_seed(30))
import numpy
from dataloader import dataset, train_dataset, test_dataset
from model import Feedforward
import torch
from DataProcessing.embedBERT import BERTEmbed

FF = torch.load("Model/feedforward.pth",  map_location=torch.device('cpu'))

#Testing on the test dataset
# loss = []
# print("Inference Starting...")
# for pair in test_dataset:
#     X, y = pair
#     pred = FF(X)
#     cur_loss = (pred[0].detach().numpy() - y) ** 2
#     loss.append(cur_loss)
#
# avg_ep_error = sum(loss) / len(loss)
# print(avg_ep_error)
#Test on a user generated value
text =  "[CLS] " + "I tried to take a photo of a wheat field. It turned out grainy." + " ["  \
                                                             "SEP]"#expect
# not funny
embed = BERTEmbed(text)
embed = dataset.getPCA(embed)
pred = FF(embed)
print("Funny Score: ", pred.detach().numpy()[0])


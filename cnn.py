import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        super(CNN, self).__init__()
        weights = w2vmodel.wv 
        EMBEDDING_SIZE = 500 
        NUM_FILTERS = 10      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding =  nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.vocab['pad'].index)
        self.convs =  nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0)) for window_size in window_sizes])
        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)


    def forward(self, x):
        embeddings = self.forward_embed(x)
        convs = self.forward_convs(embeddings)
        x = convs.view(convs.size(0), -1)
        logits = self.fc(x)
        return(logits)

    def forward_embed(self, x):
        embeddings = self.embedding(x)
        return(embeddings)

    def forward_convs(self, embed):
        x = torch.unsqueeze(embed, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)
        return(x)

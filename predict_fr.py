from gensim.utils import simple_preprocess
import torch
from gensim.models import Word2Vec
import pandas as pd
import sys



n = len(sys.argv)
if n == 3:
    print("Total arguments passed:", n-1)
if n != 3: 
    print("Requires two arguments, the name of the input file and output file as a string")

input_df = pd.read_csv(sys.argv[1])
inputs = list(input_df["Title"])
model = torch.load("cnn_model_function_role.pth")
w2v_model = Word2Vec.load("word2vec.model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_fr = {0: 'Engineering', 1:'IT-Development', 2:'IT-General', 3:'IT-Information Security', 4:'IT-Networking', 5:'Non-ICP', 6:'Procurement', 7:'Risk/Legal/Compliance'}

def make_word2vec_vector(sentence, input_list, pre_loaded_word2vec):
        #helper function to create a vector representation of a sentence within the training data set 
        max_sen_len = max(map(len, input_list))
        padding_idx = pre_loaded_word2vec.wv.vocab['pad'].index
        padded_X = [padding_idx for m in range(max_sen_len)]
        i = 0
        for word in sentence:
            if word not in pre_loaded_word2vec.wv.vocab:
                padded_X[i] = 0
          
            else:
                padded_X[i] = pre_loaded_word2vec.wv.vocab[word].index
            i += 1
        return torch.tensor(padded_X, dtype=torch.long, device=device).view(1, -1)

def make_predictions():
    cnn_predictions = []
    model.eval()
    preprocessed_inputs = [simple_preprocess(text, deacc=True) for text in inputs]
    with torch.no_grad():
        for index in range(len(preprocessed_inputs)):
            bow_vec = make_word2vec_vector(preprocessed_inputs[index], inputs, w2v_model)
            probs = model(bow_vec)
            _, predicted = torch.max(probs.data, 1)
            cnn_predictions.append(predicted.cpu().numpy()[0])
    predict_label = [label_fr[int(k)] for k in cnn_predictions]
    input_df["predicted_fr"] = predict_label
    input_df.to_csv(sys.argv[2])
    return None


make_predictions()
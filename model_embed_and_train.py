from sklearn.model_selection import train_test_split
from gensim.utils import simple_preprocess
import pandas as pd
from gensim.models import Word2Vec
import torch.nn as nn
from torch import optim
import torch
from cnn import CNN
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


class model_preprocess(object):
    def __init__(self, preprocessed_df, class_prob):
        self.label_level = {0: 'C-Level', 1: 'Contributor', 2: 'Director', 3: 'Executive', 4: 'Manager'}
        self.label_mapping_level = {'C-Level': 0, 'Contributor': 1, 'Director': 2, 'Executive':3, 'Manager': 4}
        self.label_fr = {0: 'Engineering', 1:'IT-Development', 2:'IT-General', 3:'IT-Information Security', 4:'IT-Networking', 5:'Non-ICP', 6:'Procurement', 7:'Risk/Legal/Compliance'}
        self.label_mapping_fr = {'Engineering': 0, 'IT-Development': 1, 'IT-General': 2, 'IT-Information Security': 3, 'IT-Networking': 4, 'Non-ICP': 5, 'Procurement': 6, 'Risk/Legal/Compliance': 7}
        self.data = preprocessed_df
        self.size = 500
        self.window = 3
        self.min_count = 1
        self.workers = 3
        self.sg = 1
        self.num_classes = None
        self.num_epochs = 7
        self.classification_type = class_prob  #(either "fr" for function-role or "level")
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_test_split(self):
        self.data = self.data.dropna()
        train, test = train_test_split(self.data, test_size=0.3, random_state=49)
        self.test_df = test

        x_train = list(train['Title'])
        x_test = list(test['Title'])
        self.x_train_preprocessed = [simple_preprocess(text, deacc=True) for text in x_train]
        self.x_test_preprocessed = [simple_preprocess(text, deacc=True) for text in x_test]

        y_train_level_list = list(train['levelRemapped'])
        y_test_level_list =  list(test['levelRemapped'])
        y_train_fr_list = list(train['jobFunctionRoleRemapped'])
        y_test_fr_list = list(test['jobFunctionRoleRemapped']) 

        self.y_train_level = [self.label_mapping_level[k] for k in y_train_level_list]
        self. y_test_level = [self.label_mapping_level[k] for k in y_test_level_list]
        self.y_train_fr = [self.label_mapping_fr[k] for k in y_train_fr_list]
        self.y_test_fr = [self.label_mapping_fr[k] for k in y_test_fr_list]

        

    def create_word_2_vec_model(self):
        #helper function to create a word2vec model
        self.x_train_preprocessed.append(['pad'])
        W2v_model = Word2Vec(self.x_train_preprocessed, min_count = self.min_count, size = self.size, workers = self.workers, window = self.window, sg = self.sg)
        W2v_model.save("word2vec.model")
        self.w2v_model = W2v_model
    
    def make_word2vec_vector_train(self, sentence):
        #helper function to create a vector representation of a sentence within the training data set 
        max_sen_len = max(map(len, self.x_train_preprocessed))
        padding_idx = self.w2v_model.wv.vocab['pad'].index
        padded_X = [padding_idx for m in range(max_sen_len)]
        i = 0
        for word in sentence:
            if word not in self.w2v_model.wv.vocab:
                padded_X[i] = 0
          
            else:
                padded_X[i] = self.w2v_model.wv.vocab[word].index
            i += 1
        return torch.tensor(padded_X, dtype=torch.long, device=self.device).view(1, -1)
    
    def make_word2vec_vector_test(self, sentence):
        #helper function to create a vector representation of a sentence within the testing data set
        max_sen_len = max(map(len, self.x_test_preprocessed))
        padding_idx = self.w2v_model.wv.vocab['pad'].index
        padded_X = [padding_idx for m in range(max_sen_len)]
        i = 0
        for word in sentence:
            if word not in self.w2v_model.wv.vocab:
                padded_X[i] = 0
          
            else:
                padded_X[i] = self.w2v_model.wv.vocab[word].index
            i += 1
        return torch.tensor(padded_X, dtype=torch.long, device=self.device).view(1, -1)
    
    def create_cnn_model(self):
        #create a cnn model that takes in a pre-trained word2vec model as an input
        if self.classification_type == "fr":
            self.num_classes = 8
        if self.classification_type == "level":
            self.num_classes = 5
        self.cnn_model = CNN(self.w2v_model, num_classes=self.num_classes)

    def train_cnn(self): 
        #train the model using gradient descent. Each epoch uses the whole training set (1 batch).
        self.cnn_model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)
        self.cnn_model.train() 
        if self.classification_type == "fr":
            y_train = self.y_train_fr
        if self.classification_type == "level":
            y_train = self.y_train_level
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            shuffled_i = list(range(0,len(y_train)))
            random.shuffle(shuffled_i)

            for index in range(len(shuffled_i)):
                self.cnn_model.zero_grad()
                bow_vec = self.make_word2vec_vector_train(self.x_train_preprocessed[index])
                outputs = self.cnn_model(bow_vec)
                y = torch.tensor([y_train[index]], dtype=torch.long, device=self.device)

                loss = criterion(outputs, y)
                total_loss += loss.item()

                loss.backward()
                optimizer.step() 

            print("loss on epoch %i: %f" % (epoch+1, total_loss))

    def testing_and_predictions(self):
        #apply the cnn model to the testing set and assess the model performance in terms of accuracy, precision and recall.
        #additionally, return the testing as a pandas data set written into a csv with the inclusion of the predicted labels 
        if self.classification_type == "fr":
            y_test = self.y_test_fr
            label = self.label_fr
            col_name = "predictedLabelFunctionRole"
            csv_path = "preds_function_role.csv"
        if self.classification_type == "level":
            y_test = self.y_test_level
            label = self.label_level
            col_name = "predictedLabelLevel"
            csv_path = "preds_level.csv"

        cnn_predictions = []
        original_lables_cnn = []
        self.cnn_model.eval()

        with torch.no_grad():
            for index in range(len(y_test)):
                bow_vec = self.make_word2vec_vector_test(self.x_test_preprocessed[index])
                probs = self.cnn_model(bow_vec)
                _, predicted = torch.max(probs.data, 1)
                cnn_predictions.append(predicted.cpu().numpy()[0])
                target = torch.tensor([y_test[index]], dtype=torch.long, device=self.device)
                original_lables_cnn.append(target.cpu().numpy()[0])
        predict_label = [label[int(k)] for k in cnn_predictions]
        self.test_df[col_name] = predict_label
        self.accuracy = accuracy_score(original_lables_cnn, cnn_predictions)
        self.precision = precision_score(original_lables_cnn, cnn_predictions, average='weighted')
        self.recall = recall_score(original_lables_cnn, cnn_predictions, average='weighted')
        print("Test accuracy for %s using CNN: %s percent" % (self.classification_type, str(round(self.accuracy*100, 2))))
        print("Test precision for %s using CNN: %s percent" % (self.classification_type, str(round(self.precision*100, 2))))
        print("Test recall for %s using CNN: %s percent" % (self.classification_type, str(round(self.recall*100, 2))))

        report = classification_report(original_lables_cnn, cnn_predictions)
        print(report)
        self.test_df.to_csv(csv_path)
        
    
    def save_model(self):
        if self.classification_type == "fr":
            self.model_path = "cnn_model_function_role.pth"
        if self.classification_type == "level":
            self.model_path = "cnn_model_level.pth"
        torch.save(self.cnn_model, self.model_path)


    
# NetskopePracticum

Classifying job titles into their respective Job Function, Job Level, and Job Role by implementing a word2vec model as an embedding method and feeding representation into a cnn model.  

## Files: 

### environment/environment_setup.md 

Contains details on how to set up a conda environment compatible with the required libraries.

### Historical_Lead_Records_with_id.csv.zip

The most recent version of historical records.

Please unzip before running "netskope_cnn_execute.py"

### predict_level.py/predict_fr.py

These python files load the pretrained word2vec and cnn models and return the predicted classes. 

These files require two arguments, the name of the input csv and the desired name for the output csv file.

For example: $python predict_level.py "inputs.csv" "outputs.csv" 

The main assumption is that the input csv will have a column labelled "Title".

The output csv file maintain the same format as the input csv file with the addition of the predicted values.

### netskope_cnn_execute.py 

The main python file to be executed. This script calls the modules/functions that perform preprocessing, implementation of the models, training, testing etc. 

When called, it takes in 2 additional arguments, the name of the input data which I have named "Historical_Lead_Records_with_id.csv" and the classification type, either "level" for classifying job levels or "fr" for classifying Job Function/Role. 

For example a call in terminal can look like this: $python netskope_cnn_execute.py "Historical_Lead_Records_with_id.csv" "level" or $python netskope_cnn_execute.py "Historical_Lead_Records_with_id.csv" "fr"

### data_preprocessing.py

This file contains the functions and methods for preprocessing the historical data so that it is fit for training/testing purposes.

### model_embed_and_train.py 

This file contains the functions and methods needed to split the data into training/testing, create a word2vec model, create a cnn model, train the cnn model, and evaluate performance on the testing set. In the constructor, you can change parameters such as number of epochs used in training, the learning rate, vector size etc. 

### cnn.py 

This file contains the constructor, and functions/methods necessary for performing cnn (convolutional nueral networks).

### func_role_unique.csv

This csv contains the unique combinations of Job Function and Job Role as well as their respective mapping. It is called in data_preprocessing.py and is needed to remap the historical data to the new representations.

### job_level_unique.csv

This csv contains the mapping for the Job Levels. It is called in data_preprocessing.py and is needed to remap the historical data to the new representations.

### preds_function_role.csv

This csv file is the testing data frame with the inclusion of the predicted Function-Role written to a csv.

### preds_level.csv

This csv file is the testing data frame with the inclusion of the predicted Job Level written to a csv. 

### cnn_model_function_role.pth/cnn_model_level.pth

These are the saved models for the Function/Role and Level classification problem. 

These models are called in predict_level.py or predict_fr.py.

Please unzip before running.

### word2vec.model

Saved word2vec model called in predict_level.py and predict_fr.py.

Please unzip before running.





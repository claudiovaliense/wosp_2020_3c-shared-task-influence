import timeit  # Measure time
import sys # Import other directory
import claudio_funcoes as cv  # Functions utils author
from sklearn.feature_extraction.text import TfidfVectorizer # representation tfidf
from sklearn.datasets import dump_svmlight_file # save format svmlight
import os
import nltk
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import numpy as np

ini = timeit.default_timer() # Time process
name_dataset = sys.argv[1]
ids=sys.argv[2]
datas=sys.argv[3]
labels=sys.argv[4]
index = int(sys.argv[5])

try:
    os.mkdir("dataset/representations/"+name_dataset) # Create directory
except OSError:
    print('directory exist')

x_train, y_train, x_test, y_test = cv.docs_kaggle(name_dataset) #kaggle


y_test = [1 for x in range(len(x_test))]   #kaggle
x_train = [cv.preprocessor(x) for x in x_train]
x_test = [cv.preprocessor(x) for x in x_test]

x_train_pre = x_train
x_test_pre = x_test

y_train = [float(y) for y in y_train] # float para permitir utilizar no classificador
y_test = [float(y) for y in y_test]   

union = Pipeline([                         
                       ('features',   FeatureUnion(transformer_list=[
                            
                           ('tfdif_features', Pipeline([                                
                                ('word', TfidfVectorizer(ngram_range=(1,2)) )#,                      
                           ]))
                        ]
                        
                        ))
])

x_train = union.fit_transform(x_train)
print('fold ' +str(index) +', x_train.shape: ', x_train.shape)    
dump_svmlight_file(x_train, y_train, "dataset/representations/" + name_dataset +'/train'+str(index))

x_test = union.transform(x_test)
print('fold ' +str(index) +', x_test.shape: ', x_test.shape)
dump_svmlight_file(x_test, y_test, "dataset/representations/" + name_dataset +'/test'+str(index))
print("Time End: %f" % (timeit.default_timer() - ini))

# Implicit Collaborative Filtering with TensorFlow 

This code implements a collaborative filtering recommendation model using the Matrix factorisation algorithm provided by (http://yifanhu.net/PUB/cf.pdf) and (https://www.kaggle.com/wikhung/implicit-cf-tensorflow-implementation).

To read more about model deployment in google cloud with tensorflow:
```
https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction
https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview
```

This code works in python 2 and tensorflow 1.9  
When training/deploying a new model makes sure the google cloud is using the same tensorflow version. In google cloud the tensorflow could be changed by:
```
'--runtime-version'
```


### Data set 'u.data' is from Movielens:
```
'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
```

* To train the model locally:
```
$ ./mltrain.sh local data/u.data
```


* To train the model on ML Engine in google cloud:
```
$ ./mltrain.sh train gs://mybucket/data/u.data
```

'gs://mybucket/data/u.data' is the path where the data is saved so it can be trained in cloud


* To test the deployed model:
```
$ gcloud ml-engine predict --model recom_top_k --version recom_top_k --json-instances ./data/test.json
```
or by:
```
$ ./mltrain.sh test ./data/test.json
```








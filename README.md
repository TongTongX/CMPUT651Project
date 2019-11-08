# CMPUT651Project
CMPUT 651 Project at University of Alberta in Fall 2019

## Setup Procudure
1. Ensure `torch` and `sklearn` are installed.
2. Set up the project.
```
$ pip install -e .
```
## Baseline Accuracy
### 1. SVM - linear kernel: 
#### * Trial:
  * humour_int: 41.3%
  * sarcasm_int: 36.5%
  * offensive_int: 37.5%
  * motivational_int: 57.7%
  * overall_sentiment_int: 39.4%  
### 2. SVM - Gaussian kernel:
#### * Trial:
  * humour_int: 31.7%
  * sarcasm_int: 53.8%
  * offensive_int: 42.3%
  * motivational_int: 65.4%
  * overall_sentiment_int: 46.15%
### 3. SVM - Poly kernel:
  * overall_sentiment_int: 48.07%

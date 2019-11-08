# CMPUT651Project
CMPUT 651 Project at University of Alberta in Fall 2019

## Setup Procudure
1. Ensure `torch` and `sklearn` are installed.
2. Set up the project.
```
$ pip install -e .
```
## Baseline Accuracy
### Random
#### * Trial
| Target label                  | Accuracy  | F1 Score           |
| ----------------------------- |:---------:| ------------------:|
| humour binary                 | 0.488     | 0.4444444444444444 |
| sarcasm binary                | 0.496     | 0.4698732323848552 |
| offensive binary              | 0.508     | 0.5036139327094205 |
| motivational binary           | 0.497     | 0.48901587301587296|
| overall_sentiment ternary     | 0.357     | 0.31397327074064973|
| humour multiclass             | 0.243     | 0.23348355215333016|
| sarcasm multiclass            | 0.258     | 0.23064146320843554|
| offensive multiclass          | 0.276     | 0.24770465653472937|
| overall_sentiment multiclass  | 0.198     | 0.16606166537824602|
#### * Train
| Target label                  | Accuracy              | F1 Score           |
| ----------------------------- |:---------------------:| ------------------:|
| humour binary                 | 0.5054285714285714    | 0.4692325725885891 |
| sarcasm binary                | 0.49414285714285716   | 0.4540475643136132 |
| offensive binary              | 0.49985714285714283   | 0.49274269349841937|
| motivational binary           | 0.49328571428571427   | 0.4828366396774837 |
| overall_sentiment ternary     | 0.322                 | 0.29110762093553005|
| humour multiclass             | 0.25014285714285717   | 0.2381639125273231 |
| sarcasm multiclass            | 0.243                 | 0.21696825783403043|
| offensive multiclass          | 0.25342857142857145   | 0.22805139198882443|
| overall_sentiment multiclass  | 0.19157142857142856   | 0.1613884308632046 |
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
  * overall_sentiment_int: 49.04%
### 3. SVM - Poly kernel, 16 degree:
  * overall_sentiment_int: 48.07%

import numpy as np
import sklearn.linear_model
from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.multiclass
from sklearn.datasets import make_multilabel_classification



class LogisticRegression_MultiClass:
    def __init__(self, X, y):
        # Assume X dim: (num_examples, num_features)
        self.X_input = X
        # Assume y dim: (num_examples, )
        self.y_output = y

        self.model = sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

    def fit(self):
        self.model.fit(self.X_input, self.y_output)

    def predict(self, input_X):
        return self.model.predict(input_X)

    def cal_accuracy(self, input_X, output_y):
        # Mean accuracy
        return self.model.score(input_X, output_y)



class LogisticRegression_MultiLabel:
    def __init__(self, X, y):
        # Assume X dim: (num_examples, num_features)
        self.X_input = X
        # Assume y dim: (num_examples, num_total_classes), for each y, one-hot encoding, 1 if this sample is in this class, 0 if not
        self.y_output = y

        self.model = sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression(solver='lbfgs'))

    def fit(self):
        self.model.fit(self.X_input, self.y_output)

    def predict(self, input_X):
        return self.model.predict(input_X)

    def cal_accuracy(self, input_X, output_y):
        # Mean accuracy
        return self.model.score(input_X, output_y)



# # Test for multi-class classifier
# iris = datasets.load_iris()
# X = iris.data[:, :2]
# Y = iris.target
#
# lr_classifier = LogisticRegression_MultiClass(X, Y)
#
# lr_classifier.fit()
#
# print('Training mean accuracy: ', lr_classifier.cal_accuracy(X, Y))
#
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# h = .02  # step size in the mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# Z = lr_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()



# # Test for multi-label classifier
# X, Y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1)
#
# lr_multilabel_classifier = LogisticRegression_MultiLabel(X, Y)
#
# lr_multilabel_classifier.fit()
#
# Z = lr_multilabel_classifier.predict(X)
#
# print('Training mean accuracy: ', lr_multilabel_classifier.cal_accuracy(X, Y))

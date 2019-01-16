import numpy as np
import sklearn.metrics


def f1_score(actual, predicted, average='macro'):
  actual = np.array(actual)
  predicted = np.array(predicted)
  return sklearn.metrics.f1_score(actual, predicted, average=average)

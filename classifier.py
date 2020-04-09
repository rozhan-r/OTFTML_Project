from mnist import MNIST
import sklearn.metrics as metrics
import sklearn.preprocessing as pp
import numpy as np

NUM_CLASSES = 10
PREF_DIGIT = 7
NUM_WEAK_CLASSIFIERS = 5

def load_dataset():
    mndata = MNIST('mnist_benchmark/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def weak_train(X_train, y_train, weighting, reg=0.9):
    ''' Build a model from X_train -> y_train '''
    x_vectors = X_train
    y_vectors = y_train

    a = np.dot(np.transpose(x_vectors), np.transpose(list(map(lambda el: np.multiply(weighting, el), x_vectors.T))))
    b = np.dot(np.transpose(x_vectors), np.multiply(weighting, list(y_vectors)))
    a += (reg*np.identity(x_vectors.shape[1]))

    return np.dot(np.linalg.inv(a), b)

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    matrix = np.zeros((X_train.shape[0], NUM_CLASSES))
    i = 0; 
    for label in labels_train:
        matrix[i][label] = 1
        i = i + 1;
    return matrix

def signarize(labels_train):
    return map(lambda el: 1.0 if el == PREF_DIGIT else -1.0,labels_train)

def dot_product(model, X):
    return np.dot(model, X)

def weak_predict(model, X):
    ''' From model and data points, output prediction vectors '''
    W = np.transpose(model)
    results = np.zeros(X.shape[0])
    i = 0

    for data in X:
        pred = dot_product(W, data)
        if pred > 0:
            results[i] = 1.0
        else:
            results[i] = -1.0
        i = i + 1

    return results

def strong_train(X_train, y_train):
    #should return matrix of weight vectors and alphas
    alphas = np.zeros(NUM_WEAK_CLASSIFIERS)
    weights = np.zeros((NUM_WEAK_CLASSIFIERS, X_train.shape[1]))
    #the 0th is extra - discard before returning

    step_size = int(X_train.shape[0] / (NUM_WEAK_CLASSIFIERS + 1))

    prev = X_train[0:step_size] # initial sub-dataset
    prev_y = y_train[0:step_size]

    i = 0

    while (i / step_size) < NUM_WEAK_CLASSIFIERS:
        ith = int(i / step_size)

        weighting = np.ones(step_size)
        for j in range(step_size):
            weighting[j] = 1.0 / (1.0 + np.exp(prev_y[j] * strong_eval(prev[j], weights, alphas, iters=(ith))))

        norm = np.linalg.norm(weighting)
        if norm != 0.0: 
            weighting = weighting/norm

        weights[ith] = weak_train(prev, prev_y, weighting)

        #we increment here b/c from here on out we use a new slice of data
        i = i + step_size

        prev = X_train[i : i + step_size]
        prev_y = y_train[i : i + step_size]

        epsilon = 0.0
        pred = np.zeros(step_size)
        for j in range(step_size):
            pred[j] = dot_product(weights[ith], prev[j])
        pred = np.sign(pred)
        for j in range(step_size):
            epsilon = epsilon + (pred[j] == 1.0)
        epsilon = epsilon / step_size

        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        alphas[ith] = alpha

    return weights, alphas

def strong_eval(x, weights, alphas, iters=NUM_WEAK_CLASSIFIERS):
    if(iters == 0):
        return 0.0
    result = 0.0
    for i in range(iters):
        result = result + alphas[i] * dot_product(weights[i], x)
    return result

def strong_predict(X_train, weights, alphas):
    results = np.zeros(X_train.shape[0])
    i = 0

    for data in X_train:
        pred = strong_eval(data, weights, alphas)
        if pred > 0:
            results[i] = 1.0
        else:
            results[i] = -1.0
        i = i + 1

    return results

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    print(X_train, labels_test)

    labels_train, labels_test = list(signarize(labels_train)), list(signarize(labels_test))

    #========== Weak boosted classifier ==========

    model = weak_train(X_train, labels_train, np.ones(X_train.shape[0]), reg = 0.9)

    weak_pred_labels_train = weak_predict(model, X_train)
    weak_pred_labels_test = weak_predict(model, X_test)

    print("Weak train accuracy: {0}".format(metrics.accuracy_score(labels_train, weak_pred_labels_train)))
    print("Weak validation accuracy: {0}".format(metrics.accuracy_score(labels_test, weak_pred_labels_test)))

    #========== Strong boosted classifier ==========

    weights, alphas = strong_train(X_train, labels_train)

    strong_pred_labels_train = strong_predict(X_train, weights, alphas)
    strong_pred_labels_test = strong_predict(X_test, weights, alphas)

    print("Strong train accuracy: {0}".format(metrics.accuracy_score(labels_train, strong_pred_labels_train)))
    print("Strong validation accuracy: {0}".format(metrics.accuracy_score(labels_test, strong_pred_labels_test)))

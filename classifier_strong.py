from mnist import MNIST
import sklearn.metrics as metrics
from pylab import *
import sklearn.preprocessing as pp
import numpy as np
from labels_to_current import otft_classifier_real
import random
import cv2
import scipy.stats
from labels_to_current import otft_classifier_real
import matplotlib.pyplot as plt
from save_csv import results_to_csv
from sklearn.ensemble import AdaBoostClassifier

NUM_CLASSES = 10
PREF_DIGIT = 9
NUM_WEAK_CLASSIFIERS = 4

def load_dataset(N_new):
    # digit 7
    # random.seed(10)
    # digit 0
    # random.seed(3)
    # digit 4
    # random.seed(10)
    random.seed(5)
    mndata = MNIST('mnist_benchmark/')
    X_train, labels_train = map(np.array, mndata.load_training())
    N = 2000
    #-----
    N_pref = int(N*0.4)
    index_pref = list(np.nonzero(labels_train == PREF_DIGIT)[0])
    index_pref = random.sample(index_pref, N_pref)
    index_not_pref = list(np.nonzero(labels_train != PREF_DIGIT)[0])
    index_not_pref = random.sample(index_not_pref, N-N_pref)
    index = []
    index.extend(index_pref)
    index.extend(index_not_pref)
    random.shuffle(index)
    X_train = X_train[index]
    labels_train = labels_train[index]
    #------
    X_train = X_train[0:N]
    labels_train = labels_train[0:N]
    X_test, labels_test = map(np.array, mndata.load_testing())
    #------
    Ntest = 2000
    N_pref = int(Ntest * 0.4)
    index_pref = list(np.nonzero(labels_test == PREF_DIGIT)[0])
    index_pref = random.sample(index_pref, N_pref)
    index_not_pref = list(np.nonzero(labels_test != PREF_DIGIT)[0])
    index_not_pref = random.sample(index_not_pref, Ntest - N_pref)
    index = []
    index.extend(index_pref)
    index.extend(index_not_pref)
    random.shuffle(index)
    X_test = X_test[index]
    labels_test = labels_test[index]
    #------

    X_train = np.reshape(X_train, (N, 28, 28)).astype('float32')
    X_test = np.reshape(X_test, ( X_test.shape[0], 28, 28)).astype('float32')
    X_train_n = []
    X_test_n = []
    for img in X_train:
        X_train_n.append(cv2.resize(img, (N_new, N_new)))
    for img in X_test:
        X_test_n.append(cv2.resize(img, (N_new, N_new)))
    X_train_n  = np.array(X_train_n)
    X_test_n = np.array(X_test_n)
    X_train_n = X_train_n/255.0
    X_test_n = X_test_n/255.0
    N2 = N_new*N_new
    X_train = np.reshape(X_train_n, (N, N2))
    X_test = np.reshape(X_test_n, (X_test.shape[0], N2))
    return (X_train, labels_train), (X_test, labels_test)


def weak_train(X_train, y_train, weighting, reg=0.9):
    ''' Build a model from X_train -> y_train '''
    x_vectors = X_train
    y_vectors = y_train

    a = np.dot(np.transpose(x_vectors), np.transpose(list(map(lambda el: np.multiply(weighting, el), x_vectors.T))))
    b = np.dot(np.transpose(x_vectors), np.multiply(weighting, list(y_vectors)))
    a += (reg*np.identity(x_vectors.shape[1]))

    return np.dot(np.linalg.inv(a), b)



def weak_train_otft_perceptron(X_train, y_train,  weighting, cost_interval=1, decr_learning_rate=False,
            iterations=100, learning_rate_fn=None, stochastic=False,
            verbose=False, reg=0, eps=0.0000005):

    cost = []
    X_train = np.transpose(list(map(lambda el: np.multiply(weighting, el), X_train.T)))
    y_train = np.array(y_train)
    #w = np.zeros(X_train.shape[1])
    w = y_train[15]*X_train[15]
    if learning_rate_fn is None and decr_learning_rate:
        learning_rate_fn = lambda eps, t: eps / (t + 1)
    elif not decr_learning_rate:
        learning_rate_fn = lambda eps, t: eps
    for t in range(iterations):
        index = random.randint(0, X_train.shape[0] - 1)
        X_t = X_train[index] if stochastic else X_train
        #s_t = 4.98117692e-10*2.0e4*np.matmul(X_t, w)
        #s_t, XT_t, R = otft_classifier_real(w, X_t)
        s_t = np.matmul(X_t, w)
        y_t = y_train[index] if stochastic else y_train
        mult_ys_t = np.multiply(y_t, s_t)
        index_cost_t = np.nonzero(mult_ys_t < 0)
        XT_t = X_t
        if t % cost_interval == 0:
            s = s_t
            mult_ys = np.multiply(y_train, s)
            index_cost = np.nonzero(mult_ys < 0)
            train_cost = -np.sum(mult_ys[index_cost])
            reg_cost = reg * np.linalg.norm(w) ** 2
            loss = train_cost + reg_cost
            cost.append(loss)
            if verbose:
                print("Iteration {}, Cost {}".format(t, loss))
        X_n = XT_t[index_cost_t]
        y_n = y_t[index_cost_t]
        #grad = 4.98117692e-10*2.0e4*np.dot(X_n.T, y_n)
        grad = np.dot(X_n.T, y_n)
        train_grad = -np.sum(grad)
        reg_grad = 2 * reg * w
        eps = learning_rate_fn(eps, t)
        w = w - eps * (train_grad + reg_grad)
    return w

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

def weak_predict(model, X,out):
    ''' From model and data points, output prediction vectors '''
    W = np.transpose(model)
    results = np.zeros(X.shape[0])
    i = 0
    #pred = np.matmul(X, W)
    #pred = 0.2 / 3 * np.matmul(X, W) + 0.8 / 3 * np.sum(X, axis=1) - 0.1 * np.sum(W)
    pred, _, _ = otft_classifier_real(W, X)
    for item in pred:
        if item > out:
            results[i] = 1.0
        else:
            results[i] = -1.0
        i = i + 1

    print("Max : ", np.max(pred), "\nMin:", np.min(pred))

    return results

def strong_train(X_train, y_train,eps,NUMM):
    NUM_WEAK_CLASSIFIERS = NUMM
    #should return matrix of weight vectors and alphas
    alphas = np.zeros(NUM_WEAK_CLASSIFIERS)
    weights = np.zeros((NUM_WEAK_CLASSIFIERS, X_train.shape[1]))
    #the 0th is extra - discard before returning

    step_size = int(X_train.shape[0] / (NUM_WEAK_CLASSIFIERS + 1))
    step_size = X_train.shape[0]

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

        #weighting = np.ones(step_size)
        weights[ith] = weak_train(prev, prev_y, weighting)
        #weights[ith] = weak_train_otft_perceptron(prev, prev_y, weighting = weighting, stochastic=False, verbose=True, eps=eps,iterations=70)

        #we increment here b/c from here on out we use a new slice of data
        i = i + step_size


        # prev = X_train[i : i + step_size]
        # prev_y = y_train[i : i + step_size]
        prev = X_train[0:step_size]  # initial sub-dataset
        prev_y = y_train[0:step_size]

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

def strong_eval_otft(x, weights, alphas, iters=NUM_WEAK_CLASSIFIERS, single=False):
    if(iters == 0):
        return 0.0
    result = 0.0
    for i in range(iters):
        pred,_,_ = otft_classifier_real(weights[i], x, single)
        result = result + alphas[i] * pred
    return result

def strong_predict(X_train, weights, alphas, out=0, NUMM=NUM_WEAK_CLASSIFIERS):
    results = np.zeros(X_train.shape[0])
    i = 0

    for data in X_train:
        pred = strong_eval(data, weights, alphas, NUMM)
        if pred > out:
            results[i] = 1.0
        else:
            results[i] = -1.0
        i = i + 1

    return results

def strong_predict_otft(X_train, weights, alphas, out, NUMM=NUM_WEAK_CLASSIFIERS):
    results = np.zeros(X_train.shape[0])
    i = 0
    pred = strong_eval_otft(X_train, weights, alphas, iters=NUMM)
    for item in pred:
        if item > out:
            results[i] = 1.0
        else:
            results[i] = -1.0
        i = i + 1
    print("Max : ", np.max(pred), "\nMin:", np.min(pred))
    return results


def AdaBoost_scratch(X, y, M=10, learning_rate=1, out=0):
    # Initialization of utility variables
    N = len(y)
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [], [], [], []

    # Initialize the sample weights
    sample_weight = np.ones(N) / N
    sample_weight_list.append(sample_weight.copy())

    # For m = 1 to M
    for m in range(M):
        # Fit a classifier
        model = weak_train_otft_perceptron(X, y, np.ones(X.shape[0]), verbose=False,
                                           eps=10 ** (-2.8), iterations=70)
        y_predict = weak_predict(model, X,out = out)

        # Misclassifications
        incorrect = (y_predict != y)

        # Estimator error
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Boost estimator weights
        estimator_weight = learning_rate * np.log((1. - estimator_error) / estimator_error)

        # Boost sample weights
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        # Save iteration values
        estimator_list.append(model)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Convert to np array for convenience
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # Predictions
    preds = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()+out) for point in range(N)]))
    print('Accuracy = ', (preds == y).sum() / N)
    accuracy = (preds == y).sum() / N

    return accuracy, estimator_list, estimator_weight_list, sample_weight_list


# def AdaBoost_scratch(X, y, M=10, learning_rate=1, out=0):


if __name__ == "__main__":
    N_new = 28
    (X_train, labels_train), (X_test, labels_test) = load_dataset(N_new)
    X_train = X_train*3+0.8
    X_test = X_test*3+0.8
    print(X_test.shape, labels_test)

    ones_X_train = np.ones((X_train.shape[0], 1))
    X_train = np.append(X_train, ones_X_train, axis=1)

    ones_X_test = np.ones((X_test.shape[0], 1))
    X_test = np.append(X_test, ones_X_test, axis=1)

    labels_train, labels_test = list(signarize(labels_train)), list(signarize(labels_test))

    #========== Weak boosted classifier ==========
    # accuracy_train = []
    # accuracy_test = []
    #
    # a = -0.04
    # b = 0.01
    # numm = 20
    # #model = weak_train(X_train, labels_train, np.ones(X_train.shape[0]), reg = 0.9)
    # for i in linspace(a,b,numm):
    #     model = weak_train_otft_perceptron(X_train, labels_train, np.ones(X_train.shape[0]), verbose=False,eps=10**(-7), iterations = 70)
    #     #model = weak_train(X_train, labels_train, np.ones(X_train.shape[0]), reg = 0.9)
    #
    #
    #     weak_pred_labels_train = weak_predict(model, X_train,out=i)
    #     weak_pred_labels_test = weak_predict(model, X_test,out=i)
    #
    #
    #     print("\nWeak train accuracy: {0}".format(metrics.accuracy_score(labels_train, weak_pred_labels_train)))
    #     print("Weak validation accuracy: {0}".format(metrics.accuracy_score(labels_test, weak_pred_labels_test)))
    #     accuracy_train.append(metrics.accuracy_score(labels_train, weak_pred_labels_train))
    #     accuracy_test.append((metrics.accuracy_score(labels_test, weak_pred_labels_test)))

    # plt.plot(linspace(a,b,numm),accuracy_train,label='Training Data')
    # plt.plot(linspace(a,b,numm), accuracy_test,label='Test Data')
    # plt.xlabel('threshold voltage')
    # plt.title("Accuracy of the Weak Classifier vs. threshold voltage")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(model)
    # print(max(model),min(model))
    # plt.show()

    # #========== Strong boosted classifier ==========
    a = 1
    b = 6
    # a = -0.002
    # b = 0.008
    numm = 6
    accuracy_train = []
    accuracy_test = []
    #model = weak_train(X_train, labels_train, np.ones(X_train.shape[0]), reg = 0.9)
    for i in linspace(a,b,numm):

        # #Strong vs Vth
        # weights, alphas = strong_train(X_train, labels_train, eps=10 ** (-7.2), NUMM=4)
        # strong_pred_labels_train = strong_predict_otft(X_train, weights, alphas, out=i, NUMM=4)
        # strong_pred_labels_test = strong_predict_otft(X_test, weights, alphas, out=i, NUMM=4)

        # Strong vs. num of weak classifiers
        weights, alphas = strong_train(X_train, labels_train,eps=10**(-7.2), NUMM=int(i))
        strong_pred_labels_train = strong_predict_otft(X_train, weights, alphas, out=0.006, NUMM=int(i))
        strong_pred_labels_test = strong_predict_otft(X_test, weights, alphas, out=0.006, NUMM=int(i))


        #model = AdaBoostClassifier(n_estimators=5, random_state=10)

        print("Strong train accuracy: {0}".format(metrics.accuracy_score(labels_train, strong_pred_labels_train)))
        print("Strong validation accuracy: {0}".format(metrics.accuracy_score(labels_test, strong_pred_labels_test)))


        accuracy_train.append(metrics.accuracy_score(labels_train, strong_pred_labels_train))
        accuracy_test.append((metrics.accuracy_score(labels_test, strong_pred_labels_test)))

        # accuracy,_,_,_ = AdaBoost_scratch(X_train, labels_train, M=10, learning_rate=1,out=i)
        # accuracy_train.append(accuracy)

    Vth = linspace(a,b,numm)
    results_to_csv(accuracy_train,'accuracy_train')
    results_to_csv(accuracy_test, 'accuracy_test')
    results_to_csv(Vth, 'Vth')
    plt.plot(Vth,accuracy_train, label='Training Data')
    plt.plot(Vth, accuracy_test, label='Test Data')
    plt.xlabel('Number of Weak Classifiers')
    #plt.xlabel('Vth(mV)')
    plt.title("Accuracy")
    plt.legend()
    plt.show()
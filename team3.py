import argparse
import struct
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from math import ceil
from sklearn.metrics import accuracy_score


random_state = 1126


class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.1, max_iter=50, C=0.1,
                 shuffle=True, random_state=1, batch_size=32,
                 validation_data=None):
        self.eta = eta
        self.max_iter = max_iter
        self.C = C
        self.lambda_ = 1.0 / C
        self.shuffle = shuffle
        self.random_state = random_state
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.history_ = {'train_acc': [], 'valid_acc': []}

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        num_class, p = len(self.classes_), X.shape[1]
        self._initialize_weights(num_class, p)

        r = np.arange(X.shape[0])

        for k in range(self.max_iter):
            if self.shuffle:
                self.rgen.shuffle(r)

            for i in range(ceil(X.shape[0] / self.batch_size)):
                batch_r = r[self.batch_size * i: self.batch_size * (i + 1)]
                sum_w = np.zeros((num_class, p))
                sum_b = np.zeros(num_class)

                for idx in batch_r:
                    xi = X[idx]
                    yi = -1 * np.ones(num_class)
                    yi[y[idx]] = 1

                    conf = yi * (np.dot(self.w_, xi) + self.b_)
                    conf_idx = np.where(conf < 1)

                    yt = yi.reshape(yi.shape[0], -1)
                    xt = xi.reshape(-1, xi.shape[0])

                    sum_w[conf_idx] -= np.dot(yt, xt)[conf_idx]
                    sum_b[conf_idx] -= yi[conf_idx]

                # Update
                self.w_ = self.w_ - self.eta * \
                          (sum_w / len(batch_r) + self.lambda_ * self.w_)
                self.b_ = self.b_ - self.eta * sum_b / len(batch_r)

            if k % 10 == 0:
                print(f"Iteration {k + 1} / {self.max_iter} \t", end='')
                print(f"train_accuracy {accuracy_score(self.predict(X), y)}")

        return self

    def _score(self, X, y):
        pred = self.predict(X)
        score = accuracy_score(y, pred)
        return score

    def _initialize_weights(self, n_class, p):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=(n_class, p))
        self.b_ = np.zeros(n_class)

    def predict(self, X):
        dist = np.dot(X, self.w_.T) + self.b_
        pred = np.argmax(dist, axis=1)

        return self.classes_[pred]


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_image', type=str)
    parser.add_argument('training_label', type=str)
    parser.add_argument('test_image', type=str)

    args = parser.parse_args()

    training_image = args.training_image
    training_label = args.training_label
    test_image = args.test_image

    return (training_image, training_label, test_image)


def load_data(training_image, training_label, test_image):
    with open(training_label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        y_train = np.fromfile(flbl, dtype=np.int8)

    with open(training_image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        X_train = np.fromfile(fimg, dtype=np.uint8).reshape(-1, 784)

    with open(test_image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        X_test = np.fromfile(fimg, dtype=np.uint8).reshape(-1, 784)

    return X_train, y_train, X_test


if __name__ == '__main__':
    training_image, training_label, test_image = arg()
    X_train, y_train, X_test = load_data(training_image, training_label, test_image)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    clf = SVC(max_iter=500, eta=0.001, C=1000, random_state=random_state, batch_size=256)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    with open('prediction.txt', 'w') as f:
        for i in y_pred:
            f.write(f'{i}\n')

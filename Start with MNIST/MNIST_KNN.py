import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# Use KNN to classify images
def KNN(test_data_vector, train_data, train_label, k):
    size = train_data.shape[0]
    difference = np.tile(test_data_vector, (size, 1)) - train_data
    Euclid_distance = (difference ** 2).sum(axis=1)
    sorted_index = Euclid_distance.argsort()

    classes = {}
    max_count = 0
    best_class = 0
    for i in range(k):
        current_class = train_label[sorted_index[i]]
        current_count = classes.get(current_class, 0) + 1
        classes[current_class] = current_count
        if current_count > max_count:
            max_count = current_count
            best_class = current_class
    return best_class


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    train_data = mnist.train.images
    train_label = mnist.train.labels
    test_data = mnist.test.images
    test_label = mnist.test.labels
    correct_predictions = 0
    # Calculation takes a lot of time so only 100 (randomly picked) test data are used here
    test_samples = 100
    for i in range(test_samples):
        index = np.random.randint(low=0, high=10000)
        if test_label[index] == KNN(test_data[index], train_data, train_label, 5):
            correct_predictions += 1
    print('Accuracy %g' % (correct_predictions / test_samples))

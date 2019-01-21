from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Hyper parameters
BATCH_SIZE = 128
ITERATIONS = 500
LEARNING_RATE = 1e-3
MODEL_FILEPATH = './MNIST_dense.ckpt'
CONTINUE_TRAINING = False

# Placeholders
X_image = tf.placeholder(tf.float32, shape=[None, 784], name='X_image')
Y_label = tf.placeholder(tf.float32, shape=[None, 10], name='Y_label')
keep_prob = tf.placeholder(tf.float32)


# Define a dense layer
def dense(inputs, weight_shape, name_scope, relu=False):
    with tf.name_scope(name_scope):
        weight = tf.get_variable(name=name_scope + 'weight', shape=weight_shape,
                                 initializer=tf.contrib.keras.initializers.he_normal())
        bias = tf.Variable(tf.constant(value=0.1, shape=[weight_shape[-1]], dtype=tf.float32), name=name_scope + 'bias')
        if relu:
            return tf.nn.relu(tf.matmul(inputs, weight) + bias)
        else:
            return tf.matmul(inputs, weight) + bias


# Define a convolution layer
def convolution(inputs, kernel_shape, name_scope):
    with tf.name_scope(name_scope):
        kernel = tf.get_variable(name_scope + '_kernel', shape=kernel_shape, initializer=tf.contrib.keras.initializers.he_normal())
        bias = tf.Variable(tf.constant(value=0.1, shape=[kernel_shape[-1]], dtype=tf.float32), name=name_scope + '_bias')
        return tf.nn.relu(tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME') + bias)


# Define the neural network
class Dense_Network(object):
    def __init__(self):
        # Layers
        self.input_image = tf.reshape(X_image, [-1, 28, 28, 1])
        self.convolution_layer_1_1 = convolution(self.input_image, [3, 3, 1, 64], 'conv1_1')
        self.convolution_layer_1_2 = convolution(self.convolution_layer_1_1, [3, 3, 64, 64], 'conv1_2')
        self.max_pool_1 = tf.nn.max_pool(self.convolution_layer_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.convolution_layer_2_1 = convolution(self.max_pool_1, [3, 3, 64, 128], 'conv2_1')
        self.convolution_layer_2_2 = convolution(self.convolution_layer_2_1, [3, 3, 128, 128], 'conv2_2')
        self.max_pool_2 = tf.nn.max_pool(self.convolution_layer_2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.flatten = tf.reshape(self.max_pool_2, [-1, 7*7*128])

        self.dense_layer_1 = dense(self.flatten, [7*7*128, 1024], 'dense1', relu=True)
        self.drop_1 = tf.nn.dropout(self.dense_layer_1, keep_prob=keep_prob)
        self.output = dense(self.drop_1, [1024, 10], 'output', relu=False)

        # Loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_label, logits=self.output))

        # Training op
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        # Prediction accuracy
        self.correct_predictions = tf.equal(tf.argmax(self.output, 1), tf.argmax(Y_label, 1))
        self.eval_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

        # Saving model weights
        self.saver = tf.train.Saver()

    def initialize_weights(self, sess):
        # Use global_variables_initializer() to initialize all weights (randomly)
        sess.run(tf.global_variables_initializer())

    def load_weights(self, sess):
        # Load weights previously saved to file
        self.saver.restore(sess, MODEL_FILEPATH)

    def save_weights(self, sess):
        # Save weights to file
        self.saver.save(sess, MODEL_FILEPATH)

    def training(self, sess, input_images, input_labels, keep_prob_value):
        # Train the model for one step
        sess.run(self.train_step, feed_dict={X_image: input_images, Y_label: input_labels, keep_prob: keep_prob_value})

    def eval(self, sess, input_images, input_labels):
        # Evaluate the current model with a data-batch, returning accuracy and loss
        return sess.run([self.eval_accuracy, self.loss],
                        feed_dict={X_image: input_images, Y_label: input_labels, keep_prob: 1.0})

    def predict(self, sess, input_images):
        # Returning the prediction for a data-batch
        return sess.run(tf.nn.softmax(self.output), feed_dict={X_image: input_images, keep_prob: 1.0})


if __name__ == '__main__':
    print('Loading data...')
    sess = tf.Session()
    network = Dense_Network()

    # Loading data from file, and data will be downloaded if file does not exist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Initialize or load weights
    if CONTINUE_TRAINING:
        network.load_weights(sess)
    else:
        network.initialize_weights(sess)

    print('Start training...')
    for step in range(ITERATIONS):
        data_batch = mnist.train.next_batch(BATCH_SIZE)
        network.training(sess=sess, input_images=data_batch[0], input_labels=data_batch[1], keep_prob_value=0.5)
        # Evaluate the model every 100 steps
        if (step + 1) % 100 == 0:
            train_accuracy, train_loss = network.eval(sess=sess, input_images=data_batch[0], input_labels=data_batch[1])
            print('Step %d, training accuracy %g, loss %g' % (step + 1, train_accuracy, train_loss))

    print('Saving weights...')
    network.save_weights(sess)

    print('Testing model...')
    test_accuracy = 0.0
    for step in range(10):
        test_batch = mnist.test.next_batch(1000)
        accuracy, _ = network.eval(sess=sess, input_images=test_batch[0], input_labels=test_batch[1])
        test_accuracy += accuracy
    print('Accuracy on testing data set %g' % (test_accuracy / 10.0))

    sess.close()
    print('Done.')

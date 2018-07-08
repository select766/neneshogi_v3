# based on CNTK tutorial Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb

import numpy as np
import os
import sys
import time

import cntk as C
import cntk.tests.test_utils

cntk.tests.test_utils.set_device_from_pytest_env()  # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)  # fix a random seed for CNTK components

from neneshogi_cpp import DNNConverter

cvt = DNNConverter(0, 0)
input_dim_model = cvt.board_shape()
output_dim_model = cvt.move_shape()
output_dim = int(np.prod(output_dim_model))

x = C.input_variable(input_dim_model)
y = C.input_variable(output_dim)


def create_model(features):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        h = features
        h = C.layers.Convolution2D(filter_shape=(5, 5),
                                   num_filters=8,
                                   strides=(1, 1),
                                   pad=True, name='first_conv')(h)
        h = C.layers.Convolution2D(filter_shape=(5, 5),
                                   num_filters=64,
                                   strides=(1, 1),
                                   pad=True, name='second_conv')(h)
        h = C.layers.Convolution2D(filter_shape=(5, 5),
                                   num_filters=output_dim_model[0],
                                   strides=(1, 1),
                                   pad=True, name='classify',
                                   activation=None)(h)
        r = C.reshape(h, shape=(139 * 9 * 9,))
        return r


def create_criterion_function(model, labels):
    loss = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return loss, errs  # (model, labels) -> (loss, error metric)


# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, output_dim):
    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels=C.io.StreamDef(field='labels', shape=output_dim, is_sparse=False),
        features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))

    return C.io.MinibatchSource(ctf,
                                randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)


# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]  # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))

    return mb, training_loss, eval_error


def train_test(train_reader, test_reader, model_func, num_sweeps_to_train_with=1):
    # Instantiate the model function; x is the input (feature) variable
    # We will scale the input image pixels within 0-1 range by dividing all input value by 255.
    # model = model_func(x / 255)
    model = model_func(x)

    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)

    # Instantiate the trainer object to drive the model training
    learning_rate = 0.2
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])

    # Initialize the parameters for the trainer
    minibatch_size = 64
    num_samples_per_sweep = 100000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    # Map the data streams to the input and labels.
    input_map = {
        y: train_reader.streams.labels,
        x: train_reader.streams.features
    }

    # Uncomment below for more detailed logging
    training_progress_output_freq = 100

    # Start a timer
    start = time.time()

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data = train_reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(data)
        print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    # Print training time
    print("Training took {:.1f} sec".format(time.time() - start))

    # Test the model
    test_input_map = {
        y: test_reader.streams.labels,
        x: test_reader.streams.features
    }

    # Test data for trained model
    test_minibatch_size = 128
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0

    for i in range(num_minibatches_to_test):
        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result * 100 / num_minibatches_to_test))


def do_train_test():
    global z
    z = create_model(x)
    train_file = "train_data.txt"
    test_file = "test_data.txt"
    reader_train = create_reader(train_file, True, int(np.prod(input_dim_model)), int(np.prod(output_dim_model)))
    reader_test = create_reader(test_file, False, int(np.prod(input_dim_model)), int(np.prod(output_dim_model)))
    train_test(reader_train, reader_test, z)
    z.save("policy.cmf")

do_train_test()

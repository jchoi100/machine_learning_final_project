import os
import argparse
import sys
import pickle
import random

from predictor_subclasses import *
from sklearn.datasets import fetch_mldata
import numpy as np
from cs475_types import Instance

CUSTOM_DATA_HOME_JW = '/Users/home/Desktop/machine_learning_final_project/code'
CUSTOM_DATA_HOME_JH = 'C:\Users\user01\Dropbox\School\Johns_Hopkins_University\Senior\FALL_2016\Intro_to_Machine_Learning_(EN.600.475)\project\code'
CUSTOM_DATA_HOME_JC = 'C:\Users\James\Desktop\FA16\Machine Learning\machine_learning_final_project\code'
CUSTOM_DATA_HOME_JH_UGRAD = '/home/jchoi100/Desktop/machine_learning_final_project/code'
CUSTOM_DATA_HOME_JC_UGRAD = '/home/jlee381/machine_learning_final_project/code'

def load_data(filename):
    instances = []
    # feature_vectors = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]
            label_string = label_string.split(".")[0]

            label = -1
            try:
                label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            feature_vector = []

            for item in split_line[1:257]:
                value = 0.0
                try:
                    value = float(item)
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                feature_vector.append(value)
            # feature_vectors.append(np.array(feature_vector))
            instance = Instance(feature_vector, label)
            instances.append(instance)
    # feature_vectors = np.array(feature_vectors)
    # np.savetxt(fname='usps_training_vectors.txt', X=feature_vectors, delimiter=',', fmt='%.4f')
    return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    # Default arguments.
    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True, help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    # New command line arguments.
    parser.add_argument("--knn", type=int, help="The value of K for KNN classification.", default=5)
    parser.add_argument("--train-dataset", type=str, help="The data to use for training.")
    parser.add_argument("--test-dataset", type=str, help="The data to use for testing.")

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--prediction file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(train_set, train_labels, args):
    if args.algorithm == "svm_knn":
        predictor = KNN(args.knn, True)
    elif args.algorithm == "knn":
        predictor = KNN(args.knn, False)
    elif args.algorithm == "distance_knn":
        predictor = KNN(args.knn, False, True)
    else:
        raise ValueError("Unsupported algorithm type: check your --algorithm argument.")
    predictor.train(train_set, train_labels)
    return predictor


def write_predictions(predictor, test_set, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in test_set:
                label = predictor.predict(instance)
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.data == "mnist":

        mnist = fetch_mldata('MNIST original', data_home=CUSTOM_DATA_HOME_JH)

        if args.mode.lower() == "train":
            # Load the training data.

            # Sampled dataset training.
            # train_set = []
            # train_labels = []
            # for i in range(0, 60000, 4):
            #     train_set.append(mnist.data[i])
            #     train_labels.append(mnist.target[i])
            # train_set = np.array(train_set)
            # train_labels = np.array(train_labels)

            # Full dataset training.
            train_set = mnist.data[:60000]
            train_labels = mnist.target[:60000]

            # # 7291 sampled version.
            # indices = random.sample(range(60000), 7291)
            # indices.sort()

            # # Randomly sampled data.
            # train_set = []
            # train_labels = []
            # for i in indices:
            #     train_set.append(mnist.data[i])
            #     train_labels.append(mnist.target[i])

            # Train the model.
            predictor = train(train_set, train_labels, args)
            try:
                with open(args.model_file, 'wb') as writer:
                    pickle.dump(predictor, writer)
            except IOError:
                raise Exception("Exception while writing to the model file.")        
            except pickle.PickleError:
                raise Exception("Exception while dumping pickle.")
                
        elif args.mode.lower() == "test":
            # Load the test data.

            # Sampled dataset testing.
            # test_set = []
            # true_labels = []
            # for i in range(60000, len(mnist.data), 90):
            #     test_set.append(mnist.data[i])
                # true_labels.append(mnist.target[i])
            # test_set = np.array(test_set)
            # true_labels = np.array(true_labels)

            # Full dataset testing.
            test_set = mnist.data[60000:]
            true_labels = mnist.target[60000:]
            
            # indices = random.sample(range(60000, 70001), 2007)
            # indices.sort()

            # # Randomly sampled data.
            # test_set = []
            # true_labels = []
            # for i in indices:
            #     test_set.append(mnist.data[i])
            #     true_labels.append(mnist.target[i])

            predictor = None
            # Load the model.
            try:
                with open(args.model_file, 'rb') as reader:
                    predictor = pickle.load(reader)
            except IOError:
                raise Exception("Exception while reading the model file.")
            except pickle.PickleError:
                raise Exception("Exception while loading pickle.")
                
            write_predictions(predictor, test_set, args.predictions_file)
        else:
            raise Exception("Unrecognized mode.")
    elif args.data == "usps":
        if args.mode.lower() == "train":
            # Load the training data.
            train_instances = load_data(args.train_dataset)

            # Full dataset training.
            train_set = [np.array(x._feature_vector) for x in train_instances]
            train_labels = [np.array(x._label) for x in train_instances]
            train_set = np.array(train_set)
            train_labels = np.array(train_labels)
            # Train the model.
            predictor = train(train_set, train_labels, args)
            try:
                with open(args.model_file, 'wb') as writer:
                    pickle.dump(predictor, writer)
            except IOError:
                raise Exception("Exception while writing to the model file.")        
            except pickle.PickleError:
                raise Exception("Exception while dumping pickle.")
                
        elif args.mode.lower() == "test":
            # Load the test data.
            test_instances = load_data(args.test_dataset)

            # Full dataset testing.
            test_set = [np.array(x._feature_vector) for x in test_instances]
            predictor = None
            # Load the model.
            try:
                with open(args.model_file, 'rb') as reader:
                    predictor = pickle.load(reader)
            except IOError:
                raise Exception("Exception while reading the model file.")
            except pickle.PickleError:
                raise Exception("Exception while loading pickle.")
                
            write_predictions(predictor, test_set, args.predictions_file)
        else:
            raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()


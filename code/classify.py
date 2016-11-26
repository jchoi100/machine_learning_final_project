import os
import argparse
import sys
import pickle

# from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor
from predictor_subclasses import *
from sklearn.datasets import fetch_mldata
import numpy as np

CUSTOM_DATA_HOME = 'C:\Users\user01\Dropbox\School\Johns_Hopkins_University\Senior\FALL_2016\Intro_to_Machine_Learning_(EN.600.475)\project\code'

# def load_data(filename):
#     instances = []
#     with open(filename) as reader:
#         for line in reader:
#             if len(line.strip()) == 0:
#                 continue
            
#             # Divide the line into features and label.
#             split_line = line.split(" ")
#             label_string = split_line[0]

#             int_label = -1
#             try:
#                 int_label = int(label_string)
#             except ValueError:
#                 raise ValueError("Unable to convert " + label_string + " to integer.")

#             label = ClassificationLabel(int_label)
#             feature_vector = []
            
#             for item in split_line[1:]:
#                 value = 0.0
#                 try:
#                     index = int(item.split(":")[0])
#                 except ValueError:
#                     raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
#                 try:
#                     value = float(item.split(":")[1])
#                 except ValueError:
#                     raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
#                 feature_vector.append(value)

#             instance = Instance(feature_vector, label)
#             instances.append(instance)

#     return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    # Default arguments.
    # parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True, help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    # New command line arguments.
    parser.add_argument("--knn", type=int, help="The value of K for KNN classification.", default=5)

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
        predictor = KNN(args.knn)
    else:
        raise ValueError("Unsupported algorithm type: check your --algorithm argument.")
    predictor.train(train_set, train_labels)
    return predictor


def write_predictions(predictor, test_set, true_labels, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for i in range(len(test_set)):
                label = predictor.predict(test_set[i])
                writer.write(str(label))
                writer.write('\n')
                print(label, true_labels[i])
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()
    mnist = fetch_mldata('MNIST original', data_home=CUSTOM_DATA_HOME)

    if args.mode.lower() == "train":
        # Load the training data.

        # Sampled dataset training.
        train_set = []
        train_labels = []
        for i in range(0, len(mnist.data), 4):
            train_set.append(mnist.data[i])
            train_labels.append(mnist.target[i])
        train_set = np.array(train_set)
        train_labels = np.array(train_labels)

        # Full dataset training.
        # train_set = mnist.data[:60000]
        # train_labels = mnist.target[:60000]

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

        test_set = []
        true_labels = []
        for i in range(60000, len(mnist.data), 90):
            test_set.append(mnist.data[i])
            true_labels.append(mnist.target[i])
        test_set = np.array(test_set)
        true_labels = np.array(true_labels)

        # Full dataset testing.
        # test_set = mnist.data[60000:]
        # true_labels = mnist.target[60000:]
        
        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, test_set, true_labels, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()


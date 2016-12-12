import numpy as np
import sys
from sklearn.datasets import fetch_mldata
from cs475_types import Instance
from scipy.spatial.distance import euclidean

CUSTOM_DATA_HOME_JH = 'C:\Users\user01\Dropbox\School\Johns_Hopkins_University\Senior\FALL_2016\Intro_to_Machine_Learning_(EN.600.475)\project\code'

def load_data(filename):
    instances = []
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

            instance = Instance(feature_vector, label)
            instances.append(instance)

    return instances

def sort_out_data(feature_vectors, labels):
    classified_data = {}
    for i in range(10):
        classified_data[i] = []
    for i in range(len(feature_vectors)):
        classified_data[labels[i]].append(feature_vectors[i])
    return classified_data

def compute_statistics(feature_vectors, labels, dataset_name):
    classified_data = sort_out_data(feature_vectors, labels)
    variances = {}
    for i in range(10):
        variances[i] = 0.0

    for i in range(10):
        datapoints = classified_data[i]
        mu_i = np.mean(datapoints, 0) / 255
        if dataset_name == "MNIST":
            mu_i / 255
        N = len(datapoints)
        variance = 0.0
        for j in range(N):
            x_j = np.array(datapoints[j])
            if dataset_name == "MNIST":
                x_j /= 255
            variance += euclidean(mu_i, x_j)**2
        variances[i] = variance / N

    print_statistics(variances, dataset_name)

def print_statistics(variances, dataset_name):
    print("========" + dataset_name + "========")
    # print(variances)
    for i in range(10):
        print("--------------------")
        print(str(i) + ": " + str(variances[i]))

def main():
    mnist = fetch_mldata('MNIST original', data_home=CUSTOM_DATA_HOME_JH)
    mnist_train_set = mnist.data[:60000]
    mnist_train_labels = mnist.target[:60000]

    usps = load_data("mldata/zip.train")
    usps_train_set = [np.array(x._feature_vector) for x in usps]
    usps_train_labels = [np.array(x._label) for x in usps]
    usps_train_set = np.array(usps_train_set)
    usps_train_labels = np.array(usps_train_labels)

    compute_statistics(mnist_train_set, mnist_train_labels, "MNIST")
    compute_statistics(usps_train_set, usps_train_labels, " USPS")

if __name__ == "__main__":
    main()


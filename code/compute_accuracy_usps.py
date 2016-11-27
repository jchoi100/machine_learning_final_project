import sys
from sklearn.datasets import fetch_mldata
from cs475_types import Instance

if len(sys.argv) != 3:
    print 'usage: %s data predictions' % sys.argv[0]
    sys.exit()


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


data_file = sys.argv[1]
predictions_file = sys.argv[2]

predictions = open(predictions_file)

# Load the real labels.
true_instances = load_data(data_file)
true_labels = [str(x._label) for x in true_instances]

predicted_labels = []
for line in predictions:
    predicted_labels.append(line.strip())

predictions.close()

# print predicted_labels
# print true_labels

if len(predicted_labels) != len(true_labels):
    print 'Number of lines in two files do not match.'
    sys.exit()
    
match = 0
total = len(predicted_labels)

for ii in range(len(predicted_labels)):
    if predicted_labels[ii] == true_labels[ii]:
        match += 1

print 'Accuracy: %f (%d/%d) | %s' % ((float(match)/float(total)), match, total, str(sys.argv[1]))

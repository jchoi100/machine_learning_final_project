import sys
from sklearn.datasets import fetch_mldata

CUSTOM_DATA_HOME_JW = '/Users/home/Desktop/machine_learning_final_project/code'
CUSTOM_DATA_HOME_JH = 'C:\Users\user01\Dropbox\School\Johns_Hopkins_University\Senior\FALL_2016\Intro_to_Machine_Learning_(EN.600.475)\project\code'
CUSTOM_DATA_HOME_JC = 'C:\Users\James\Desktop\FA16\Machine Learning\machine_learning_final_project\code'
CUSTOM_DATA_HOME_JH_UGRAD = '/home/jchoi100/Desktop/machine_learning_final_project/code'
CUSTOM_DATA_HOME_JC_UGRAD = '/home/jlee381/machine_learning_final_project/code'

mnist = fetch_mldata('MNIST original', data_home=CUSTOM_DATA_HOME_JH_UGRAD)

if len(sys.argv) != 2:
    print 'usage: %s data predictions' % sys.argv[0]
    sys.exit()

# data_file = sys.argv[1]
predictions_file = sys.argv[1]

# data = open(data_file)
predictions = open(predictions_file)

# Load the real labels.
true_labels = []
# for i in range(60000, len(mnist.data)):
for i in range(60000, len(mnist.data), 90):
    true_labels.append(str(mnist.target[i]))

predicted_labels = []
for line in predictions:
    predicted_labels.append(line.strip())

# data.close()
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

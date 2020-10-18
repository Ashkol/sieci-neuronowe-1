import random
import numpy as np
import matplotlib.pyplot as plt

class Adaline(object):

    def __init__(self, allowed_error, num_of_inputs=2, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_of_inputs + 1) #weights[0] is reserved for bias
        self.allowed_error = allowed_error

    def set_starting_weights(self, min, max):
        for w in range(len(self.weights)):
            self.weights[w] = random.uniform(min, max)
        self.weights[0] = 0

    def _add_bias(self, learning_set):
        #Add bias whose signal is always equal to 1 at the beginning as x0
        bias = np.ones((learning_set.shape[0], 1))
        learning_set_biased = np.hstack((bias, learning_set))
        return learning_set_biased


    def predict(self, input):
        summation = np.dot(input, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else -1

    def predict_array(self, X):
        X = self._add_bias(X)
        return np.where(self._activation(self._weigthed_sum(X)) >= 0.0, 1, -1)

    def _sum_squared_errors(self, y, activation_output):
        errors = y - output_pred
        return (errors**2).sum() / 2.0

    def _weigthed_sum(self, X):
        return np.dot(X, self.weights)

    def _activation(self, X):
        return X

    def train(self, learning_set, labels, epochs_left):
        self.weights[0] = 1
        learning_set = self._add_bias(learning_set)
        print("Training model according to data")
        for epoch in range(epochs_left):
            # ALC
            activation_output = self._activation(self._weigthed_sum(learning_set))
            errors = labels - activation_output
            errors_squared = (errors ** 2).sum() / 2.0
            self.weights +=  self.learning_rate * learning_set.T.dot(errors)
            #print(self.weights[0])
            # print("Cost = " + str(errors_squared))
            if errors_squared < self.allowed_error:
                return epoch + 1

        print("Training failed in given number of epochs")
        return -1


def generate_extra_examples(examples, precision, extra_num):
    extras = []
    labels = []
    for n in range(extra_num):
        example = random.choice(examples)
        extras.append((example[0] + random.uniform(-precision, precision), example[1] + random.uniform(-precision, precision)))
        labels.append(example[2])
    return (np.asarray(extras), np.asarray(labels))

class_examples = [(0, 0, -1), (1, 0, -1), (0, 1, -1), (1, 1, 1)]
sum = 0
sum_correct_labels_percentage = 0
for n in range(10):
    inputs, labels = generate_extra_examples(class_examples, 0.1, 100)
    adaline = Adaline(14, 2, learning_rate=0.01)
    adaline.set_starting_weights(-0.0, 0.0)
    epochs = adaline.train(inputs, labels, 200)
    sum += epochs

    inputs, labels = generate_extra_examples(class_examples, 0.1, 20)
    sum_correct_labels_percentage += np.count_nonzero((adaline.predict_array(inputs) == labels) == True) / 20

print("Epochs to learn = " + str(sum / 10))
print("Correct labels = " + str(sum_correct_labels_percentage * 10) + "%")

input_to_predict = [0.9, 0.9]
print("Predicted result: " + str(adaline.predict(input_to_predict)) + " of " + str(input_to_predict) +
      " in " + str(epochs) + " epochs")



x = np.asarray(inputs)[:,0]
y = np.asarray(inputs)[:,1]
colors = adaline.predict_array((inputs))
plt.scatter(x, y, c=colors)
plt.axline((0, -adaline.weights[1] * 0 / adaline.weights[2] - adaline.weights[0]),
           (1, -adaline.weights[1] * 1 / adaline.weights[2] - adaline.weights[0]))
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title("Adaline")
plt.show()
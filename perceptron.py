import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def generate_extra_examples(examples, precision, extra_num):
    extras = []
    for n in range(extra_num):
        example = random.choice(examples)
        extras.append((example[0] + random.uniform(-precision, precision), example[1] + random.uniform(-precision, precision), example[2]))
    return extras


class MyPerceptronDynamicBias(object):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = [0, 0]
        self.bias_weight = 0
        self.bipolar = False

    def set_starting_weights(self, min, max):
        self.weights = [random.uniform(min, max), random.uniform(min, max)]

    def predict(self, input):
        summation = input[0] * self.weights[0] + input[1] * self.weights[1] + self.bias_weight
        if summation > 0:
            activation = 1
        else:
            activation = (-1 if self.bipolar else 0)
        return activation

    def predict_array(self, arr):
        predictions = []
        for a in arr:
            predictions.append(1 if self.predict(a) >= 0.0 else -1 if self.bipolar else 0)
        return np.asarray(predictions)

    def train(self, learning_set, epochs_left):
        #print("Learning bias = " + str(self.learning_rate))
        #print(self.weights)
        for epoch in range(epochs_left):
            predictions = []
            #print("Epoch " + str(epoch + 1))

            for example in learning_set:
                awake = self.weights[0] * example[0] + self.weights[1] * example[1]
                predictions.append((self.predict(example)))

            errors = []
            for n in range(len(learning_set)):
                errors.append(learning_set[n][2] - predictions[n])
                self.weights[0] += errors[n] * self.learning_rate * learning_set[n][0]
                self.weights[1] += errors[n] * self.learning_rate * learning_set[n][1]
                self.bias_weight += self.learning_rate * errors[n]

            if self.bipolar:
                if not (2 in errors or -2 in errors):
                    return epoch + 1
            else:
                if not (1 in errors or -1 in errors):
                    return epoch + 1

        #print(str(errors.count(0) * 100 / len(errors)) + "% good fits")
        return epochs_left


class MyPerceptron(object):

    def __init__(self, threshold=-0.03, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = [0, 0]
        self.bipolar = False

    def set_starting_weights(self, min, max):
        self.weights = [random.uniform(min, max), random.uniform(min, max)]

    def predict(self, input):
        summation = input[0] * self.weights[0] + input[1] * self.weights[1]
        if summation > self.threshold:
            activation = 1
        else:
            activation = -1 if self.bipolar else 0
        return activation

    def train(self, learning_set, epochs):

        for epoch in range(epochs):
            predictions = []
            #print("Epoch " + str(epoch + 1))

            for example in learning_set:
                awake = self.weights[0] * example[0] + self.weights[1] * example[1]
                predictions.append((self.predict(example)))

            errors = []
            for n in range(len(learning_set)):
                errors.append(learning_set[n][2] - predictions[n])
                self.weights[0] += errors[n] * self.learning_rate * learning_set[n][0]
                self.weights[1] += errors[n] * self.learning_rate * learning_set[n][1]

            # if self.bipolar:
            #     if not (2 in errors or -2 in errors):
            #         return epoch + 1
            if not (1 in errors or -1 in errors):
                #print("Training set properly labeled ")
                return epoch + 1

        #print("Reached the last epoch")
        #print(str(errors.count(0) * 100 / len(errors)) + "% good fits")
        return epochs


###
class_examples = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
class_examples = [(0, 0, -1), (1, 0, -1), (0, 1, -1), (1, 1, 1)]
examples = generate_extra_examples(class_examples, 0.1, 100)

perceptron = MyPerceptron(learning_rate=0.01)
perceptron.threshold = 0.5
perceptron.bipolar = True

epochs_sum = 0
for _ in range(0, 10):
    examples = generate_extra_examples(class_examples, 0.1, 100)
    #perceptron.weights = [0, 0]
    perceptron.set_starting_weights(0, 0)
    epochs_sum += perceptron.train(examples, 500)
print("--- Perceptron ---\nAverage epochs number = " + str(epochs_sum/10))

perceptron_bias = MyPerceptronDynamicBias(learning_rate=0.01)
perceptron_bias.bipolar = True

epochs_sum = 0
for _ in range(0, 10):
    examples = generate_extra_examples(class_examples, 0.1, 100)
    #perceptron_bias.weights = [0, 0]
    perceptron_bias.set_starting_weights(-0, 0)
    epochs_sum += perceptron_bias.train(examples, 500)
print("--- Perceptron with bias ---\nAverage epochs number = " + str(epochs_sum/10))

print("Predicted result: " + str(perceptron_bias.predict([0.8, 0.9])))

examples = generate_extra_examples(class_examples, 0.1, 20)
x = np.asarray(examples)[:, 0]
y = np.asarray(examples)[:, 1]
colors = perceptron_bias.predict_array(examples)
plt.scatter(x, y, c=colors)
plt.axline((0, -perceptron_bias.weights[0] * 0 / perceptron_bias.weights[1] - perceptron_bias.bias_weight),
           (1, -perceptron_bias.weights[0] * 1 / perceptron_bias.weights[1] - perceptron_bias.bias_weight))
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title("Perceptron")
plt.show()
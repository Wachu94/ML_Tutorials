import numpy as np, random, pickle
from tqdm import trange

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

for i in range(len(X)):
    X[i].append(1)

def save(data):
    with open("logic_gates.p", 'wb') as file:
        pickle.dump(data, file)

def load():
    with open("logic_gates.p", 'rb') as file:
        return pickle.load(file)

def setup_weights(*hidden_layers):
    input_size = len(X[0])
    layers_sizes = [input_size, *hidden_layers, len(Y[0])]
    weights = []
    for i in range(len(layers_sizes)-1):
        current_weights = []
        for j in range(layers_sizes[i]*layers_sizes[i+1]):
            current_weights.append((0.5 - random.random()) * 2)
        current_weights = np.reshape(current_weights,(layers_sizes[i+1], layers_sizes[i]))
        weights.append(current_weights)
    return weights

def calculate_neuron_values(observation, weights):
    neuron_values = [observation]
    output = observation
    for i in range(len(weights)):
        output = np.matmul(weights[i], output)
        for j in range(len(output)):
            if output[j] < 0:
                output[j] /= 1000
        neuron_values.append(output)
    return neuron_values

def calculate_output(observation, weights):
    output = observation
    for i in range(len(weights)):
        output = np.matmul(weights[i], output)
        for j in range(len(output)):
            if output[j] < 0:
                output[j] = 0
    return output

def calculate_errors(neuron_values, weights, y):
    combined_errors = 0
    errors = [[]]*(len(neuron_values)-1)
    errors[len(errors) - 1] = y - neuron_values[len(errors)]
    for i in range(2, len(weights)+1):
        errors[len(errors) - i] = np.matmul(errors[len(errors) - (i-1)], weights[len(weights)-(i-1)])
    for i in range(len(errors[len(errors) - 1])):
        combined_errors +=  abs(errors[len(errors) - 1][i])
    return errors, combined_errors

def update_weights(errors, weights, neuron_values, lr=0.01):
    for i in range(len(errors)):
        for j in range(len(errors[i])):
            for k in range(len(weights[i][j])):
                correction = errors[i][j] * neuron_values[i][k]  * lr
                weights[i][j][k] += correction
    return weights

def train(episodes):
    try:
        progress_bar = trange(episodes)
        weights = setup_weights(16)
        for _ in progress_bar:
            combined_errors = 0
            for i in range(len(X)):
                neuron_values = calculate_neuron_values(X[i], weights)
                errors, output_errors = calculate_errors(neuron_values, weights, Y[i])
                weights = update_weights(errors, weights, neuron_values)
                combined_errors += output_errors
            progress_bar.set_description("Error: %f" %  combined_errors)
            if combined_errors < 10e-10:
                progress_bar.close()
                break
        return weights
    except KeyboardInterrupt:
            progress_bar.close()
            return weights

weights = train(100000)
save(weights)
# weights = load()

for i in range(len(X)):
    print(calculate_output(X[i],weights))
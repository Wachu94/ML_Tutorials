import numpy as np, gym, random, pickle, time
from tqdm import trange

train_episodes = 100
batch_size = 100
test_episodes = 50

H = 100
# gamma = 1 - 1e-2
gamma = 0.99

env_name = "CartPole-v0"
env = gym.make(env_name)

def save(data):
    with open(env_name + "_critic.p", 'wb') as file:
        pickle.dump(data, file)

def load():
    with open(env_name + "_critic.p", 'rb') as file:
        return pickle.load(file)

def setup_weights(*layers_sizes):
    weights = []
    for i in range(len(layers_sizes) - 1):
        weights.append(np.random.uniform(size=(layers_sizes[i+1], layers_sizes[i] + 1)))
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

def calculate_errors(neuron_values, weights, y):
    combined_errors = 0
    errors = [[]]*(len(neuron_values)-1)
    errors[len(errors) - 1] = y - neuron_values[len(errors)]
    for i in range(2, len(weights)+1):
        errors[len(errors) - i] = np.matmul(errors[len(errors) - (i-1)], weights[len(weights)-(i-1)])
    for i in range(len(errors[len(errors) - 1])):
        combined_errors +=  abs(errors[len(errors) - 1][i])
    return errors, combined_errors

def calculate_predictions(X, Y, weights, activation="sigmoid", advantage=[]):
    neuron_values = []
    output = X
    for i in range(len(weights)):
        output = np.c_[output, np.ones(output.shape[0])]
        neuron_values.append(np.array(output))
        output = activation_function(output.dot(weights[i].T), activation)
    if len(advantage) > 0:
        errors = [advantage]
    else:
        errors = [Y - output]
    for i in range(len(weights) - 1):
        errors.append(errors[i].dot(weights[len(weights) - 1 - i]))
        errors[i + 1] = errors[i + 1] * neuron_values[len(neuron_values) - 1 - i]
        errors[i + 1] = np.delete(errors[i + 1], errors[i + 1].shape[1] - 1, 1)
        if activation == "sigmoid":
            errors[i+1] = errors[i+1] * (1 - errors[i+1])
    errors.reverse()
    return neuron_values, errors


def update_weights(errors, weights, neuron_values, lr=10e-6):
    for i in range(len(errors)):
        for j in range(len(errors[i])):
            for k in range(len(weights[i][j])):
                correction = errors[i][j] * neuron_values[i][k]  * lr
                weights[i][j][k] += correction
    return weights

def SGD(X, Y, weights, lr=1e-2, epochs=10, activation_func="relu", advantage=[]):
    for i in range(epochs):
        # if len(advantage) != 0:
        #     errors = np.array([advantage])
        #     neuron_values = []
        #     output = X
        #     for i in range(len(weights)):
        #         output = np.c_[output, np.ones(output.shape[0])]
        #         neuron_values.append(np.array(output))
        #         output = activation_function(output.dot(weights[i].T), activation_func)
        # else:
        #     neuron_values, errors = calculate_predictions(X, Y, weights, activation_func)
        neuron_values, errors = calculate_predictions(X, Y, weights, activation_func, advantage)

        loss = 0
        for e in errors:
            loss += np.sum(e**2)
        # print(loss)

        gradient = []
        for i in range(len(neuron_values)):
            gradient.append(errors[i].T.dot(neuron_values[i]) / -neuron_values[i].shape[0])
        for i in range(len(weights)):
            weights[i] -= lr * gradient[i]
    return loss

def activation_function(input, type = "relu"):
    if type == "sigmoid":
        return 1 / (1 + np.exp(-input))
    elif type == "relu":
        for i in range(len(input)):
            if isinstance(input[i],float):
                if input[i] < 0:
                    input[i] = 0
                continue
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = 0
        return input
    elif type == "unipolar":
        for i in range(len(input)):
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = 0
                else:
                    input[i][j] = 1
        return input
    elif type == "elu":
        for i in range(len(input)):
            if isinstance(input[i],float):
                if input[i] < 0:
                    input[i] = np.exp(input[i]) - 1
                continue
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = np.exp(input[i][j]) - 1
        return input
    elif type == "softmax":
        sum = 0.0
        for i in range(len(input)):
            input[i] = np.exp(input[i])
            sum += input[i]
        input /= sum
        return input

def calculate_output(state, weights, activation='relu', prob_based=False, show_probs=False):
    output = state
    for i in range(len(weights)):
        output = np.append(output, 1)
        output = np.dot(weights[i], output)
        if i != len(weights) - 1:
            output = activation_function(output, activation)
    if prob_based:
        output = activation_function(output, 'softmax')
        if show_probs:
            print(output)
        temp = 0
        rand_value = np.random.random()
        for i in range(len(output)):
            temp += output[i]
            if temp > rand_value or i == len(output) - 1:
                return i
    return output

def train():
    input_size = len(np.reshape(env.reset(),-1))
    if isinstance(env.action_space.sample(),int):
        output_size = env.action_space.n
        prob_based = True
    else:
        output_size = len(env.action_space.sample())
        prob_based = False
    policy = setup_weights(input_size, output_size)
    baseline = setup_weights(input_size, 1)
    progress_bar = trange(train_episodes)
    for i in progress_bar:
        states_buffer, action_buffer, reward_buffer = [], [], []
        for j in range(batch_size):
            reward_buffer.append([])
            observation = env.reset()
            while True:
                observation = np.reshape(observation, -1)
                states_buffer.append(observation)
                action = calculate_output(observation, policy, "relu", prob_based)
                action_buffer.append(action)
                observation, reward, done, _ = env.step(action)
                reward_buffer[j].append(reward)
                if done:
                    break
        states_buffer = np.array(states_buffer)
        V = []
        for r in range(batch_size):
            for t in range(len(reward_buffer[r])):
                V.append([0])
                for t_ in range(t, min(t + H, len(reward_buffer[r]))):
                    V[len(V)-1][0] += reward_buffer[r][t_] * (gamma**(t_ - t + 1))
        V = np.array(V)
        # print(V)
        # return

        if i == 0:
            SGD(states_buffer, V, baseline, lr=1, epochs=5)
        SGD(states_buffer, V, baseline, lr=1e-6, epochs=1)

        for _ in range(1):
            advantage = []
            if prob_based:
                for i in range(len(V)):
                    advantage.append([0 for _ in range(output_size)])
                    advantage[i][action_buffer[i]] = (V[i] - calculate_output(states_buffer[i], baseline, 'elu'))[0]
                    # if i != len(V) - 1:
                    #     advantage[i][action_buffer[i]] += V[i + 1][0] - V[i][0]
            else:
                advantage.append((V[i] - calculate_output(states_buffer[i], baseline))[0])
            advantage = np.array(advantage)
            SGD(states_buffer, None, policy, lr=1e-2, epochs=1, advantage=advantage)
        # print(policy)
    save(policy)
    return policy, baseline

def test(policy, baseline):
    try:
        # weights = load()
        for i in range(test_episodes):
            score = 0
            observation = env.reset()
            while True:
                env.render()
                # time.sleep(0.1)
                observation = np.reshape(observation, (-1, 1))
                # print(calculate_output(observation, baseline)[0])
                action = calculate_output(observation, policy, prob_based=True)
                observation, reward, done, info = env.step(action)
                score += reward
                if done:
                    break
            print("Score in episode",i,"=",score)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

policy, baseline = train()
test(policy, baseline)
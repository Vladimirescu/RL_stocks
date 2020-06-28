import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
from sklearn.preprocessing import StandardScaler
from Environment import Environment
from Agent import Agent
import seaborn as sns


def get_scaler(env):

    # scalar object for states - run an episode -> get some states -> normalize along dimensions
    # with this gathered data
    # state.size = 2 * n_stocks + 1
    # number of shares for each n_stocks + prices for each stock + total amount of cash in hand
    states = []

    for i in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def play_one_episode(agent, env, scaler, is_train=True):

    state = env.reset() # set initial values for stock prices and initial investment
    state = scaler.transform([state]) # normalize state
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, cur_val = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train:
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return cur_val


def get_stock_values(files):
    """
    :param files: array of .csv files, each containing stock prices over time for that company
    :return: Pandas dataframe with all necessary info
    """
    df = pd.read_csv(files[0])[' Open']
    for i in range(1, len(files)):
        df_new = pd.read_csv(files[i])[' Open']
        df = pd.concat([df, df_new], axis=1, sort=False)

    stock_values = df.values
    good_stock_values = []

    found = False
    for d in stock_values:
        for i in d:
            if not isinstance(i, str):
                found = True
        if found:
            break

        x = [j.strip(' ') for j in d]
        x = [float(k[1:]) for k in x]
        good_stock_values.append(x)

    good_stock_values = np.array(good_stock_values)
    for s in range(good_stock_values.shape[1]):
        good_stock_values[:, s] = good_stock_values[::-1, s]

    return good_stock_values


def plot_stocks(data, train_data, test_data):
    colors = ["r", "g", "b", "gold", "k", "m", "lime", "slateblue", "aqua", "fucshia"]
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title("Train data")
    for i in range(data.shape[1]):
        plt.plot(train_data[:, i], colors[i], label=names[i])
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Test data")
    for i in range(data.shape[1]):
        plt.plot(test_data[:, i], colors[i], label=names[i])
    plt.legend()
    plt.show(1)


def train_agent(initial_investment, train_data, num_episodes):
    env = Environment(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)

    # default exploration rate = 0.1
    agent = Agent(state_size, action_size)
    scaler = get_scaler(env)

    final_values = []

    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, scaler)
        dt = datetime.now() - t0

        print(f"Episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        final_values.append(val)

    agent.save()

    # save the scaler
    with open(f'Models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # save portfolio value for each episode
    np.savetxt("Rewards/train_multiple_stocks", final_values, delimiter=' ', fmt="%.6f")


def test_agent(initial_investment, test_data, num_episodes, epsilon):
    # load scaler from training agent
    with open(f'Models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    env = Environment(test_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)

    agent = Agent(state_size, action_size)
    # eps = 1 for random behaviour
    agent.epsilon = epsilon
    # load trained weights
    agent.load()
    final_values = []

    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, scaler, is_train=False)
        dt = datetime.now() - t0

        print(f"Episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        final_values.append(val)

    file_name = "test_multiple_stocks"
    if agent.epsilon == 1:
        file_name += "_random"
    np.savetxt("Rewards/" + file_name, final_values, delimiter=' ', fmt="%.6f")


if __name__ == '__main__':

    # TO DO : check the volume when buying

    num_episodes = 30
    initial_investment = 2000

    names = [#"Amazon",
             #"Amd",
             "Apple",
             #"Cisco",
             "Facebook",
             #"Microsoft",
             #"Qualcomm",
             "Starbucks",
             #"Tesla",
             #"Zinga"
             ]

    files = ["Stocks/" + i + ".csv" for i in names]

    data = get_stock_values(files)
    n_times, n_stocks = data.shape
    test_size = 0.5
    n_train = int(n_times * (1 - test_size))

    train_data = data[:n_train, :]
    test_data = data[n_train:, :]

    # plot_stocks(data, train_data, test_data)

    train_agent(initial_investment, train_data, num_episodes)
    test_agent(initial_investment, test_data, num_episodes, 0.1)
    test_agent(initial_investment, test_data, num_episodes, 1)

    train = np.loadtxt("Rewards/train_multiple_stocks", delimiter=' ')
    test_random = np.loadtxt("Rewards/test_multiple_stocks_random", delimiter=' ')
    test = np.loadtxt("Rewards/test_multiple_stocks", delimiter=' ')

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title("Random behaviour")
    plt.grid()
    sns.distplot(test_random)
    plt.subplot(2, 1, 2)
    plt.title("Trained agent")
    plt.grid()
    sns.distplot(test)
    plt.xlabel("Final amount of cash")
    plt.show()

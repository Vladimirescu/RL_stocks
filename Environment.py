import numpy as np
from itertools import product


class Environment:
    """
    Sock prices environment
    n_step = number of days for which we know the stock values
    n_stock = number of different stocks
    cur_step = current day
    data = numpy array with dimension (n_step, n_stock) containing all stock prices
    """

    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3 ** self.n_stock)

        # an action may look like : [2, 1, 1, 0, ....]
        # i.e. buy first stock, hold second, hold third, sell fourth, etc.
        # 0 = sell all
        # 1 = hold
        # 2 = buy

        # all possible actions
        self.action_list = get_all_possible_actions(self.n_stock)
        # n_stock prices + n_stock shares + total amount of money
        self.state_dim = self.n_stock * 2 + 1

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self.get_obs()

    def step(self, action):
        assert action in self.action_space

        # current cash in hand
        prev_val = self.get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # buy, sell, hold - specified for each stock
        self.trade(action)

        # after trade cash in hand
        cur_val = self.get_val()

        # reward is the increase in porfolio value - what we're trying to maximize
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        return self.get_obs(), reward, done, cur_val

    def get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def trade(self, action):
        action_vec = self.action_list[action]

        sell_index = []  # stores index of stocks we want to sell
        buy_index = []  # stores index of stocks we want to buy
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        if len(sell_index) != 0:
            # if we sell, we sell all owned stock simplified approach
            # TO DO : try to keep some, but how much ?
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0

        if len(buy_index) != 0:
            # buy as much as you can (simple but inefficient)
            # use only a fraction of your current cash_in_hand to buy shares
            percent_keep = 0.0
            to_spend = (1 - percent_keep) * self.cash_in_hand
            self.cash_in_hand *= percent_keep
            can_buy = [1 for i in buy_index]
            # keep buying until you can't afford any share
            while sum(can_buy) >= 1:
                for idx, i in enumerate(buy_index):
                    if to_spend > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        to_spend -= self.stock_price[i]
                    else:
                        can_buy[idx] = 0
            # take back if something remained
            self.cash_in_hand += to_spend

            # TO DO : approx how much to buy from each stock (NNs?)


def get_all_possible_actions(n_stocks):
    # returns all possible lists with n_stocks elements from the set {0, 1, 2}
    A = [[0, 1, 2] for i in range(n_stocks)]
    return list(product(*A))


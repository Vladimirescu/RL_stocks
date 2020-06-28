import numpy as np
from keras.layers import Dense, Input
from keras.models import Model


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = extended_linear_model(state_size, action_size,
                                           state_size*action_size,
                                           state_size*action_size)

    def act(self, state):
        # self.epsilon - probability to do something random
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        # given state, output Q(s, a)
        act_values = self.model.predict(state)
        # return best action to-do - highest return
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            # Q-learning rule : set target to be the maximal Q(s, a) among all actions
            # even if we don't chose to do that action
            # Q(s,a) <- Q(s,a) + lerning_rate*(reward + gamma * argmax{Q(s,a)} - Q(s,a))
            # the 'argmax' is taken along actions dimension
            # In our case, learning_rate = 1, which results in the formula below :
            target = reward + self.gamma * np.max(self.model.predict(next_state), axis=1)

        # prepare the target output our model should compute
        target_full = self.model.predict(state)
        target_full[0, action] = target

        # Run one training step - force the model to output the "correct" Q(s, a) for
        # given state s and action a - don't care for Q of the other actions
        # force the Q(s, a) = Q(s_next, a) -> convergence of algorithm
        self.model.fit(x=state, y=target_full, batch_size=1, verbose=0, epochs=1)

        # Reduce probability of exploring
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self):
        self.model.load_weights("Models/linear_model.h5")

    def save(self):
        self.model.save("Models/linear_model.h5")


# Linear model trying to approx. action-value function
# given state s we want to output the best approx for Q(s, a)
def linear_model(inpt_dim, outpt_dim):
    inpt = Input(shape=(inpt_dim,))
    outpt = Dense(outpt_dim)(inpt)
    model = Model(inputs=inpt, outputs=outpt)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Linear model with extended number of hidden layers
def extended_linear_model(inpt_dim, outpt_dim, *args):
    inpt = Input(shape=(inpt_dim,))

    if len(args) == 1:
        hl = Dense(args[0])(inpt)
        outpt = Dense(outpt_dim)(hl)
    elif len(args) > 1:
        hl = Dense(args[0])(inpt)
        for i in range(1, len(args)):
            hl = Dense(args[i])(hl)
        outpt = Dense(outpt_dim)(hl)
    else:
        outpt = Dense(outpt_dim)(inpt)

    model = Model(inputs=inpt, outputs=outpt)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


"""
모두를 위한 머신러닝 / 딥너링 강의 - Deep Reinforcement Learning
Lecture7 : DQN, Lab 7-2 DQN 2 (Nature 2015) 에서 사용된 코드 기반으로
openAI gym의 Pendulum-v0 환경에 DQN을 적용시켜보는
개인적인 공부를 위해 각종 파라미터를 수정하기 용이하도록 수정된 코드입니다.
해당 강의 영상은 아래 주소를 통해 확인하실 수 있습니다.
https://www.youtube.com/watch?v=ByB49iDMiZE&feature=youtu.be
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt

import gym
env = gym.make('Pendulum-v0')

input_size = env.observation_space.shape[0]
power_unit = 0.1
output_size = int((2 - (-2)) / power_unit + 1) # Output dimension

dis = 1
REPLAY_MEMORY = 100000
l_rate = 3e-3
max_episodes = 10000
regularization = False
reg_rate = 1
batch_size = 32
hidden_layer_size = (50,40)
optimizer = "relu"
consecutive_trial = 100
n_bot_play = 10

iteration = 1

def lrelu(x, alpha = 0.01):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def which_optimizer(X, opt = "tanh"):
    if opt == "tanh":
        return tf.nn.tanh(X)
    elif opt == "relu":
        return tf.nn.relu(X)
    elif opt == "sigmiod":
        return tf.nn.sigmoid(X)
    elif opt == "lrelu":
        return lrelu(X, 0.1)
    else:
        print(opt, " is NOT optimizer, use tanh.")
        return tf.nn.tanh(X)

def action_parser(array):
    return np.argmax(array) * power_unit - 2

def action_deparser(scalar):
    return int((scalar + 2) / power_unit)

###########################

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=hidden_layer_size, l_rate=l_rate):
        layer_depth = len(h_size)
        with tf.variable_scope(self.net_name):
            W_list = []
            self._X = tf.placeholder(tf.float32,
                                     [None, self.input_size], name="input_x")
            for i in range(layer_depth):
                if i is 0:
                    w_input = self._X
                else:
                    w_input = W_list[-1]
                W_list.append(tf.layers.dense(inputs=w_input, units=h_size[i], activation=tf.nn.relu))
            self._Qpred = tf.layers.dense(inputs=W_list[-1], units=output_size, activation=None)


        self._Y = tf.placeholder(shape=[None, self.output_size],
                                 dtype=tf.float32)

        if regularization:
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred)) \
                     + reg_rate * tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2))
        else :
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        #self._train = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train],
                                feed_dict={self._X: x_stack, self._Y: y_stack})


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        action = action_deparser(action[0])
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name = "main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = [action_parser(mainDQN.predict(s))]
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score : {}".format(reward_sum))
            break


def main():

    replay_buffer = deque()
    log = np.empty(0)

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name = "main")
        targetDQN = DQN(sess, input_size, output_size, name = "target")
        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
        success = []
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            done_check = []
            flag = True

            while not done:
                if flag or np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = [action_parser(mainDQN.predict(state))]

                next_state, reward, done, _ = env.step(action)
                #reward = policy(done, step_count, next_state)
                if flag and reward > -1 and episode > 500:
                    flag = False
                next_state = next_state.reshape((1,-1))
                done_check.append(reward > -1)
                if len(done_check) >= 30:
                    if all(done_check[-30:]):
                        done = True
                replay_buffer.append((state, action, reward, next_state, done))
                while len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

            success.append(step_count < 200 and done)
            print("Episode: {} step: {}".format(episode, step_count))
            log = np.append(log, step_count)

            if len(success) >= 50:
                if all(success[-50:]):
                    break

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                sess.run(copy_ops)

        for _ in range(n_bot_play):
            input("Press any key.")
            bot_play(mainDQN)
    #plt.plot(log)
    #plt.show()
    return len(log)


if __name__ == "__main__":
    save_result = np.empty(0)
    for i in range(iteration):
        tf.reset_default_graph()
        print("stage: ", i)
        save_result = np.append(save_result, main())
    print("mean steps: {}, result: {}".format(np.mean(save_result), save_result) )
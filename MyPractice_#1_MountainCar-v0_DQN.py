"""
모두를 위한 머신러닝 / 딥너링 강의 - Deep Reinforcement Learning
Lecture7 : DQN, Lab 7-2 DQN 2 (Nature 2015) 에서 사용된 코드 기반으로
openAI gym의 MountainCar-v0 환경에 DQN을 적용시켜보는
개인적인 공부를 위해 각종 파라미터를 수정하기 용이하도록 수정된 코드입니다.
해당 강의 영상은 아래 주소를 통해 확인하실 수 있습니다.
https://www.youtube.com/watch?v=ByB49iDMiZE&feature=youtu.be
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque

import gym
env = gym.make('MountainCar-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 1
REPLAY_MEMORY = 50000
l_rate = 1e-2
lr_decay = 1e-2
max_episodes = 5000
regularization = False
reg_rate = 1
batch_size = 32
hidden_layer_size = (10,5)
optimizer = "relu"
consecutive_trial = 100
middle_reward_boundary = -0.25

iteration = 10

def lrelu(x, alpha = 0.01):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def policy(done, step_count = 0, next_state = 0, action_count = []):
    if next_state[0] >= 0.5:
        return 100
    if max(action_count) > 190:
        return -100
    if next_state[0] > middle_reward_boundary:
        return (next_state[0] + 0.5)**2 * 100
    return -1


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

###########################

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=hidden_layer_size, l_rate=l_rate):
        h1_size, h2_size = h_size
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32,
                                     [None, self.input_size], name="input_x")
            W1 = tf.get_variable("W1", shape=[self.input_size, h1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            h1 = tf.matmul(self._X, W1)
            layer1 = which_optimizer(h1, optimizer)

            W2 = tf.get_variable("W2", shape=[h1_size, h2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            h2 = tf.matmul(layer1, W2)
            layer2 = which_optimizer(h2, optimizer)

            W3 = tf.get_variable("W3", shape=[h2_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            self._Qpred = tf.matmul(layer2, W3)

        self._Y = tf.placeholder(shape=[None, self.output_size],
                                 dtype=tf.float32)

        if regularization:
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred)) \
                     + reg_rate * tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2))
        else :
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

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

        if done:
            Q[0, action] = reward
        else:
            try:
                Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))
            except:
                print(action, reward, np.max(targetDQN.predict(next_state)))
                Q[0, action] = reward

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
        a = np.argmax(mainDQN.predict(s))
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

        sess.run(copy_ops)

        flag = True
        sr = deque()

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            max_velocity = -2
            max_position = -2
            state = env.reset()
            action_count = [0,0,0]
            success = 0

            while not done:
                if flag or np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
                action_count[action] += 1

                next_state, reward, done, _ = env.step(action)
                reward = policy(done, step_count, next_state, action_count)

                if next_state[0] > middle_reward_boundary:
                    for i in range(99):
                        replay_buffer.append((state, action, reward, next_state, done))
                replay_buffer.append((state, action, reward, next_state, done))
                while len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                last_position = next_state[0]

                # middle_reward_boundary를 넘길때까지, random으로 action을 취함
                if flag and last_position >= middle_reward_boundary:
                    flag = False
                if last_position >= 0.5:
                    success = 1
                if next_state[0] > max_position:
                    max_position = next_state[0]
                if next_state[1] > max_velocity:
                    max_velocity = next_state[1]
                step_count += 1

            sr.append(success)
            if len(sr) > 10:
                sr.popleft()

            if sum(sr) == 10:
                break

            print("Episode: {} step: {} max_velocity: {} max_position : {} / {}".format(episode, step_count, max_velocity, max_position, action_count))

            #log = np.append(log, step_count)

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, batch_size)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                sess.run(copy_ops)

        bot_play(mainDQN)
    return len(log)


if __name__ == "__main__":
    save_result = np.empty(0)
    for i in range(iteration):
        tf.reset_default_graph()
        print("stage: ", i)
        save_result = np.append(save_result, main())
    print("mean steps: {}, result: {}".format(np.mean(save_result), save_result) )
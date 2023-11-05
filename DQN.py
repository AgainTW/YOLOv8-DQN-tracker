import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import glob
import cv2
from PIL import Image as im

import MOT_func as MOT

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop


# build Q learning
## 實際進行訓練的 evaluation network
class Eval_Q_Model(tf.keras.Model):
  # Evaluate Q Model 的模型架構
  def __init__(self, num_actions):
    super().__init__('mlp_q_network')
    self.layer1 = layers.Dense(10, activation='relu')
    self.logits = layers.Dense(num_actions, activation=None)

  def call(self, inputs):
    x = tf.convert_to_tensor(inputs)
    layer1 = self.layer1(x)
    logits = self.logits(layer1)
    return logits

## 久久更新一次的 evaluation network
class Target_Q_Model(tf.keras.Model):
  # Target Q Model 的模型架構(不更新layer權重)
  def __init__(self, num_actions):
    super().__init__('mlp_q_network_1')
    self.layer1 = layers.Dense(10, trainable=False, activation='relu')
    self.logits = layers.Dense(num_actions, trainable=False, activation=None)

  def call(self, inputs):
    x = tf.convert_to_tensor(inputs)
    layer1 = self.layer1(x)
    logits = self.logits(layer1)
    return logits

class DeepQNetwork:
  def __init__(self, n_actions, n_features, eval_model, target_model):
    # 利用dic儲存參數
    self.params = {
        'n_actions': n_actions,       # actions數量
        'n_features': n_features,     # features數量 
        'learning_rate': 0.01,
        'reward_decay': 0.9,
        'e_greedy': 0.9,              # 選擇隨機action的機率
        'replace_target_iter': 300,   # 從 Eval_Q_Model 更新 Target_Q_Model 需要的迭代次數
        'memory_size': 500,
        'batch_size': 32,
        'e_greedy_increment': None
        }

    # total learning step
    self.learn_step_counter = 0
    # epsilon 表機率，訓練過程中有 epsilon 的機率 agent 會選擇隨機
    self.epsilon = 0 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
    # initialize zero memory ，每個 memory 中的 experience 大小為 (state + next state + reward + action)
    self.memory = np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2))

    self.eval_model = eval_model
    self.target_model = target_model

    self.eval_model.compile(
        optimizer=RMSprop(learning_rate=self.params['learning_rate']),
        loss='mse'
        )
    self.cost_his = []

  ## 儲存經驗
  def store_transition(self, s, a, r, s_):
    if not hasattr(self, 'memory_counter'):
      self.memory_counter = 0

    # 打包agent經驗
    transition = np.hstack((s, [a, r], s_))

    # 用新memory取代舊memory
    index = self.memory_counter % self.params['memory_size']
    self.memory[index, :] = transition
    self.memory_counter += 1

  ## 決定是否具有隨機以及隨機的方式
  def choose_action(self, observation):
    # to have batch dimension when feed into tf placeholder
    observation = observation[np.newaxis, :]

    if np.random.uniform() < self.epsilon:
    # 透過觀察得到每個 action 的 q 值
      actions_value = self.eval_model.predict(observation)
      print(actions_value)
      action = np.argmax(actions_value)
    else: # 隨機移動
      action = np.random.randint(0, self.params['n_actions'])
    return action

  ## 從 memory 中取樣學習
  def learn(self):
    # 從 memory 取樣 batch memory
    if self.memory_counter > self.params['memory_size']:
      sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
    else:
      sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])

    batch_memory = self.memory[sample_index, :]

    # 計算現在 eval net 和 target net 得出 Q value 的落差
    q_next = self.target_model.predict(batch_memory[:, -self.params['n_features']:])
    q_eval = self.eval_model.predict(batch_memory[:, :self.params['n_features']])

    # 根據 q_eval 改變 q_target
    q_target = q_eval.copy()

    batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
    eval_act_index = batch_memory[:, self.params['n_features']].astype(int)
    reward = batch_memory[:, self.params['n_features'] + 1]

    q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * np.max(q_next, axis=1)

    # check to replace target parameters
    if self.learn_step_counter % self.params['replace_target_iter'] == 0:
      for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
        target_layer.set_weights(eval_layer.get_weights())
      print('\ntarget_params_replaced\n')

    # train eval network
    self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.params['n_features']], q_target)
    self.cost_his.append(self.cost)

    # 增加 epsilon ，使 agent 傾向不隨機
    self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] else self.params['e_greedy']
    self.learn_step_counter += 1

  def plot_cost(self):
    plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()

def run_cleaner(ag, RL):
  step = 0

  action = 1
  n_states = ag.get_env()
  n_states = np.hstack((n_states, ag.locx, ag.locy))
  error_flag, reward, done_flag, n_states_new = ag.step(action)
  RL.store_transition(n_states, action, reward, n_states_new)

  for episode in range(300):
    # RL 選擇 action
    action = RL.choose_action(n_states_new)
    # 執行 action 並得到下一階段 info
    n_states = n_states_new
    error_flag, reward, done_flag, n_states_new = ag.step(action)
    RL.store_transition(n_states, action, reward, n_states_new)

    if (step > 200) and (step % 5 == 0):
      RL.learn()

    # break while loop when end of this episode
    if done_flag:
      print('Episode finished after {} timesteps, total rewards {}'.format(step+1, rewards))
      break
    step += 1
  # end of game
  print('over')

## built agent
class agent():
  def __init__(self):
    self.loc = [0, 0]
    self.env = [0, 0]
    self.distance = 0
    self.n_actions = 4
    self.n_features = np.zeros(6)
    self.done = 0

  def go(self,do):
    if(do==1 and self.loc_x<600): self.loc_x = self.loc_x + 1
    elif(do==2 and 0<self.loc_x): self.loc_x = self.loc_x - 1
    elif(do==3 and self.loc_y<600): self.loc_y = self.loc_y + 1
    elif(do==4 and 0<self.loc_y): self.loc_y = self.loc_y - 1

  def step(self):
    self.go(action)

    dist = MOT.distance(self.loc, self.env)
    # reward設計
    reward = (self.distance-dist)/10

    return reward, done ,dist


# DQN 模型建立
eval_model_1 = Eval_Q_Model(num_actions=4)
target_model_1 = Target_Q_Model(num_actions=4)
RL_1 = DeepQNetwork(4, 11, eval_model_1, target_model_1)
RL_1.params['memory_size'] = 500
RL_1.params['e_greedy'] = 0.4


for i in range(10):
  run_cleaner(ag1, RL_1)

RL_1.plot_cost()

"""## predict test"""

# agent初始化
ag = agent()
ag.initial(env1)
ag.set_loc(0, 0)

for i in range(100):
  state = ag.get_env()
  state = np.hstack((state, ag.locx, ag.locy))
  ag.go(RL_1.choose_action(state))
  picture = ag.full_get()
  picture = scipy.ndimage.zoom(picture, 500, order=0)
  picture = im.fromarray(picture)
  picture = picture.convert('RGB')
  picture = picture.resize((512,512))



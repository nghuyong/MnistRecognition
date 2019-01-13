import csv
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_cnn_fc, model_fc
import numpy as np

# create a shared session between Keras and Tensorflow
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
policy_sess = tf.Session()
K.set_session(policy_sess)
MAX_TRIALS = 250  # maximum number of models generated

MAX_EPOCHS = 10  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = False  # restore controller to continue training

# construct a state space
state_space = StateSpace()
# for fc
# NUM_LAYERS = 1
# state_space.add_state(name='hidden', values=list(np.arange(10, 800, 10, dtype=np.int32)))
# add states
NUM_LAYERS = 2
state_space.add_state(name='kernel', values=[2, 3, 4, 5])
state_space.add_state(name='filters', values=[16, 32, 64])
# state_space.add_state(name='pool_size', values=[1, 2, 3, 4])
# state_space.add_state(name='hidden_size', values=[500, 100, 200, 300])

# print the state space being searched
state_space.print_state_space()

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    reward, previous_acc = manager.get_rewards(model_cnn_fc, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = []
            data.extend(state_space.parse_state_space_list(state))
            data.extend([previous_acc, round(reward, 5),
                         round(total_reward, 5)])
            writer = csv.writer(f)
            writer.writerow(data)
    print()
print("Total Reward : ", total_reward)

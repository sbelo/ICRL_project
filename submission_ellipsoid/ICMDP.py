##################################################
#   file name: ICMDP
#
#   description:
#   this file defines ICMDP class and all of its functions.
#   ICMDP is an Inverse Contextual Markov Decision Proces setup, it includes:
#   1. THETA - transitions map for the MDP
#   2. F = state-feature map the MDP
#   3. w - a vector that represent the linear dependency of the reward in a feature
#   4. C - a context matrix
#   5. W - a matrix that represent the linear dependency of w in C (the context)
#   6. solve_MDP - a method that solves a MDP
#   7. solve_CMDP - a method that solves a CMDP
#   8. solve_IMDP - a method that solves an IMDP
#   9. solve_ICMDP - a method that solves an ICMDP
#                   imports
##################################################
import numpy as np
#from constraint_gen import *
from constraint_gen_norm2 import *
from tqdm import tqdm
# end of imports
##################################################

#######################################################################################################################
# a class to know if an agrument wasn't given:
class empty: pass

#######################################################################################################################
# Feature class:
class Feature:

    ##################################################
    # function name: __init__
    # inputs:
    # dim_features  - the desired number of dimensions
    # feature       - a state-feature mapping in an array-like object, for 1 state
    # description:
    #   initiates features as an empty array in a specified size or as the given state-feature mapping
    def __init__(self, dim_features = -1,feature = 0):

        if feature == 0:
            assert(dim_features > 0), "dimension of feature must be positive"
            self.feature = np.zeros(dim_features)
        else:
            self.feature = np.zeros(len(feature))
            for i in range(len(feature)):
                self.feature[i] = feature[i]

    # end of function __init__
    ##################################################

    ######################################################################
    # __getitem__:
    # i - index or a slice
    # description:
    # returns the i'th feature
    def __getitem__(self, i):
        return self.feature[i]
    # end of __getitem__
    ######################################################################

    ######################################################################
    # size :
    # description:
    # returns the dimension of the feature
    def size(self):
        return len(self.feature)
    # end of size
    ######################################################################

#######################################################################################################################
# Features class:
class Features:

    ##################################################
    # function name: __init__
    # inputs:
    # num_states    - the number of states
    # dim_features  - the dimension of the features
    # features      - a 'Features' object to copy from
    # description:
    #   initiates an empty 'Features' object with the wanted number of states and features or uses 'features' input to construct an identical object.
    def __init__(self, num_states = 0, dim_features = -1,features = 0):

        self.__features = []

        if features == 0:
            assert(dim_features > 0), "feature dimension must be positive"
            self.__num_states = num_states
            self.__dim_features = dim_features
            if num_states > 0:
                for i in range(num_states):
                    self.__features.append(Feature(dim_features=dim_features))
        else:
            [self.__num_states, self.__dim_features] = features.size()
            for i in self.__num_states:
                self.__features.append(Feature(feature=features[i]))
    # end of function __init__
    ##################################################

    ######################################################################
    # __getitem__:
    # i - index or a slice
    # description:
    # returns the i'th state feature vector
    def __getitem__(self, i):
        return self.__features[i]
    # end of __getitem__
    ######################################################################

    ######################################################################
    # __setitem__ :
    # i     - index or a slice
    # value - a correct size vector or matrix
    # description:
    # sets the argument in the 'i' index to value. used only to change existing value
    def __setitem__(self, i, value):
        assert(value.size() == self.__dim_features), "feature dimensions must match"
        self.__features[i] = value
    # end of __setitem__
    ######################################################################

    ######################################################################
    # mat_form :
    # i     - index or a slice
    # value - a correct size vector or matrix
    # description:
    # sets the argument in the 'i' index to value. used only to change existing value
    def mat_form(self, num_actions = 0):
        if num_actions == 0:
            mat = np.zeros([self.__num_states, self.__dim_features])
            for i in range(self.__num_states):
                for j in range(self.__dim_features):
                    mat[i,j] = self.__features[i][j]

            return mat

        else:
            mat = np.zeros([self.__num_states* num_actions, self.__dim_features])
            for i in range(self.__num_states):
                for j in range(self.__dim_features):
                    for k in range(num_actions):
                        mat[i*num_actions + k,j] = self.__features[i][j]
            return mat
    # end of mat_form
    ######################################################################

    ######################################################################
    # add_feature :
    # inputs:
    # feature - a state-feature mapping in an array-like object, for 1 state
    # description:
    # adds a new feature to the 'Features' object
    def add_feature(self, feature):
        assert(len(feature) == self.__dim_features), "feature dimensions must agree"
        self.__features.append(Feature(feature=feature))
        self.__num_states = self.__num_states + 1
    # end of add_feature
    ######################################################################

    ######################################################################
    # export :
    # inputs:
    # filename  - a desired name for the text file
    # description:
    # exports the Features into a text file
    def export(self, filename = 'features.txt'):
        file = open(filename, 'w+')
        file.write("function F = make_F\n")
        file.write("F = zeros(%d, %d);\n" % (self.__num_states, self.__dim_features))
        # file.write("F(1, 1) = 0.75;\nF(1, 2) = 0.5;\nF(1, 3) = 0.5;\n")
        for i in range(self.__num_states):
            for j in range(self.__dim_features):
                # file.write((",%f" % self.__features[i].feature[j]).rstrip('0').rstrip('.'))
                file.write(("F(%d, %d) = %f" % (i+1,j+1,self.__features[i].feature[j])).rstrip('0').rstrip('.'))
                file.write(";\n")
        file.close()
    # end of export
    ######################################################################

    ######################################################################
    # size :
    # outputs:
    # [number of states, dimension of feature]
    # description:
    # returns a list with the number of states and the dimension of the feature
    def size(self):
        return [self.__num_states, self.__dim_features]
    # end of size
    ######################################################################

#######################################################################################################################
# State class:
class State:
     ##################################################
    # function name: __init__
    # inputs:
    # speed     - speed
    # x         - the x of my car
    # other_car - the info about the other car in a list
    # description:
    #   initiating a 'State' object with th desired parameters.

    def __init__(self, speed, x, other_car):
        self.speed = speed
        self.x = x
        self.other_car = other_car
    # end of function __init__
    ##################################################

    ##################################################
    # function name: as_list
    # outputs:
    # a list containing the 'State' object arguments
    def as_list(self):
        return [self.speed, self.x] + self.other_car
    # end of function as_list
    ##################################################

#######################################################################################################################
# SA_Trans (State-Action Transition object):
class SA_Trans:
    ##################################################
    # function name: __init__
    # inputs:
    # num_states         - number of states
    # state              - the start state for the object
    # action             - the action for the object
    # state_action_trans - an 'SA_Trans' object
    # description:
    # initiates a state-action transitions in the folowing way:
    # 1. by defining the number of states, and the state and the action that creates the state-action pair
    #
    def __init__(self, num_states=-1, state=-1, action=-1, state_action_trans = 0):

        self.__sa_transitions = []
        # construct an empty SA_Trans object in the desired size:
        if state_action_trans == 0:
            assert(state >=0 and action >=0), "state and/or action aren't defined"
            assert(num_states > 0), "states number has to be > 0 "
            self.__num_states = num_states
            for i in range(self.__num_states):
                self.__sa_transitions.append(float(0))

            self.state = state
            self.action = action
        # construct SA_Trans object, identical to state_action_trans:
        else:
            self.__num_states = state_action_trans.size()

            for i in range(self.__num_states):
                self.__sa_transitions.append(state_action_trans.get_trans(i))

            self.state = state_action_trans.state
            self.action = state_action_trans.action
    # end of function __init__
    ##################################################

    ##################################################
    # function name: size
    # inputs:
    # state    - state # of transition
    # description:
    #  returns the number of states
    def size (self):
        return self.__num_states
    # end of function size
    ##################################################

    ##################################################
    # function name: get_trans
    # inputs:
    # state    - state # of transition
    # description:
    #  returns the transition probability of state
    def get_trans (self, state):
        assert(state <= self.__num_states), "State is out of range."
        return self.__sa_transitions[state]
    # end of function get_trans
    ##################################################

    ##################################################
    # function name: set_trans
    # inputs:
    # state    - state # of transition
    # trans    - transition transition probability
    # description:
    #   sets trans as the transition probabilty to state from self.state with action self.action
    def set_trans (self, state, trans):
        assert(trans >=0 and trans <=1), "Probability has to be between 0 to 1."
        assert(state <= self.__num_states), "State is out of range."
        self.__sa_transitions[state] = trans
    # end of function set_trans
    ##################################################

#######################################################################################################################
# State_Trans class - contains information about the transitions of a state with every possible action:
class State_Trans:
    ##################################################
    # function name: __init__
    # inputs:
    # action        - action #
    # start_state    - initial state # of transition
    # end_state     - end state # of transition
    # outputs:
    # description:
    #   initiating simulation: parameters, memory, counters

    def __init__(self, num_states = -1, num_actions = -1, state = -1, transitions = 0):

        if transitions == 0:
            assert(state >= 0), "state is not defined"
            assert(num_states > 0 and num_actions >= 0), "num_states / num_actions isn't defined"
            self.__transitions = []
            for i in range(num_actions):
                self.__transitions.append(SA_Trans(num_states, state, i))

            self.__num_actions = num_actions
            self.__num_states = num_states
            self.state = state

        else:
            for i in range(self.__num_actions):
                self.__transitions.append(SA_Trans(transitions.get_action_trans(i)))
            [self.__num_actions, self.__num_states] = transitions.size()
            self.state = transitions.state
    # end of function __init__
    ##################################################

    ##################################################
    # function name: size
    # inputs:
    # state    - state # of transition
    # description:
    #  returns the number of states
    def size (self):
        return [self.__num_actions, self.__num_states]
    # end of function size
    ##################################################

    ##################################################
    # function name: get_trans
    # inputs:
    # action        - action #
    # start_state    - initial state # of transition
    # end_state     - end state # of transition
    # outputs:
    # description:
    #   initiating simulation: parameters, memory, counters
    def get_trans (self, action,state):
        assert(action >= 0 and action < self.__num_actions), "action is out of range"
        return self.__transitions[action].get_trans(state)
    # end of function get_trans
    ##################################################

    ##################################################
    # function name: set_trans
    # inputs:
    # action        - action #
    # start_state    - initial state # of transition
    # end_state     - end state # of transition
    # outputs:
    # description:
    #   initiating simulation: parameters, memory, counters
    def set_trans (self, action,state,trans):
        assert(action >= 0 and action < self.__num_actions), "action is out of range"
        self.__transitions[action].set_trans(state,trans)
    # end of function set_trans
    ##################################################

    ##################################################
    # function name: get_action_trans
    # inputs:
    # action        - action #
    # start_state    - initial state # of transition
    # end_state     - end state # of transition
    # outputs:
    # description:
    #   initiating simulation: parameters, memory, counters
    def get_action_trans (self, action):
        assert(action >= 0 and action < self.__num_actions), "action is out of range"
        return self.__transitions[action]
    # end of function get_action_trans
    ##################################################

    ##################################################
    # function name: set_action_trans
    # inputs:
    # action        - action #
    # start_state    - initial state # of transition
    # end_state     - end state # of transition
    # outputs:
    # description:
    #   initiating simulation: parameters, memory, counters
    def set_action_trans (self, action_trans):
        assert(action_trans.size() <= self.__num_states), "states number doesn't fit."
        assert(action_trans.action < self.__num_actions), "action is out of range"
        assert(action_trans.state == self.state), "state doesn't match"
        self.__transitions[action_trans.action] = SA_Trans(state_action_trans = action_trans)
    # end of function set_action_trans
    ##################################################

#######################################################################################################################
# Transitions class:
class Transitions:

    ##################################################
    # function name: __init__
    # inputs:
    #
    # outputs:
    # description:
    #   initiating simulation: parameters, memory, counters

    def __init__(self, num_states = -1, num_actions = -1, transitions=0):

        self.__transitions = []

        if transitions == 0:
            assert(num_actions > 0 and num_states > 0), " illegal number of actions or states."
            for i in tqdm(range(num_states)):
                self.__transitions.append(State_Trans(num_states,num_actions, state = i))

            self.__num_actions = num_actions
            self.__num_states = num_states

        else:
            [self.__num_states, self.__num_actions] = transitions.size()

            for i in range(self.__num_states):
                self.__transitions.append(State_Trans(transitions.get_state_trans(i)))

    # end of function __init__
    ##################################################

    ######################################################################
    # set_state_trans:
    # description:
    # returns the i'th state feature vector
    def set_state_trans (self, start_state, state_transition):
        self.__transitions[start_state] = State_Trans(state_transition)
        # end of set_state_trans
    ######################################################################

    ######################################################################
    # get_state_trans:
    # description:
    # returns the i'th state feature vector
    def get_state_trans (self, start_state):
        return self.__transitions[start_state]
    # end of get_state_trans
    ######################################################################

    ######################################################################
    # set_sa_trans:
    # description:
    # returns the i'th state feature vector
    def set_sa_trans (self, start_state, sa_transition):
        self.__transitions[start_state].set_action_trans(sa_transition)
    # end of set_sa_trans
    ######################################################################

    ######################################################################
    # get_sa_trans:
    # description:
    # returns the i'th state feature vector
    def get_sa_trans (self, start_state, action):
        return self.__transitions[start_state].get_action_trans(action)
    # end of get_sa_trans
    ######################################################################

    ######################################################################
    # set_trans:
    # description:
    # returns the i'th state feature vector
    def set_trans (self, start_state, action, end_state, trans):
        self.__transitions[start_state].set_trans(action,end_state,trans)
    # end of set_trans
    ######################################################################

    ######################################################################
    # get_trans:
    # description:
    # returns the i'th state feature vector
    def get_trans (self, start_state, action,end_state):
        return self.__transitions[start_state].get_trans(action,end_state)
    # end of get_trans
    ######################################################################

    ######################################################################
    # size :
    # description:
    # returns the number of transitions = #actions * (#states)^2
    def size(self):
        return [self.__num_states, self.__num_actions]

    # end of size
    ######################################################################

    ######################################################################
    # mat_form :
    # description:
    # returns a numpy array from of the transitions
    def mat_form(self):
        mat = np.zeros([self.__num_states*self.__num_actions, self.__num_states])
        for start in tqdm(range(self.__num_states)):
            for action in range(self.__num_actions):
                for end in range(self.__num_states):
                    mat[start*self.__num_actions + action, end] = self.get_trans(start_state=start,action=action,end_state=end)
        return mat

    # end of mat_form
    ######################################################################

    ######################################################################
    # export :
    # description:
    # exports to matlab format
    def export(self, filename = 'transitions.txt'):
        file = open(filename, 'w+')
        file.write("function THETA = make_THETA\n")
        file.write("THETA = zeros(%d, %d, %d);\n" % (self.__num_states, self.__num_actions, self.__num_states))

        start = 0
        for act in [1,0,2]:
            for end in range(self.__num_states):
                trans = self.get_trans(start,act,end)
                if trans > 0:
                    file.write(("THETA(%d, %d, %d) = %.15f" % (start+1,act+1,end+1,trans)).strip('0').strip('.'))
                    file.write(";\n")
        for act in [1,0,2]:
            for start in range(1,self.__num_states):
                for end in range(self.__num_states):
                    trans = self.get_trans(start,act,end)
                    if trans > 0:
                        file.write(("THETA(%d, %d, %d) = %.15f" % (start+1, act+1, end+1, trans)).strip('0').strip('.'))
                        file.write(";\n")

        file.close()

    # end of export
    ######################################################################

#######################################################################################################################
# IMDP_SOL class - contains the solution IMDP using MWAL:	
class IMDP_SOL:

    ##################################################
    # function name: __init__
    # inputs:
    # dim_features  - the desired number of dimensions
    # feature       - a state-feature mapping in an array-like object, for 1 state
    # description:
    #   initiates features as an empty array in a specified size or as the given state-feature mapping
    def __init__(self, policies, M, WW):

        self.WW = WW
        self.policies = policies
        self.M = M
    # end of function __init__
    ##################################################

#######################################################################################################################
# MDP_SOL class - contains the solution of MDP:
class MDP_SOL:

    ##################################################
    # function name: __init__
    # inputs:
    # dim_features  - the desired number of dimensions
    # feature       - a state-feature mapping in an array-like object, for 1 state
    # description:
    #   initiates features as an empty array in a specified size or as the given state-feature mapping
    def __init__(self, state_feature_exp, policy, M):

        self.state_feature_exp = state_feature_exp
        self.policy = policy
        self.M = M
    # end of function __init__
    ##################################################

#######################################################################################################################
# ICMDP class:
class ICMDP:
    # GLOBAL definitions, parameters etc. of ICMDP class

    ##################################################
    # function name: __init__
    # inputs:
    # THETA - transitions
    # F     - features
    # C     - contexts
    # w     - reward-feature expectations mapping
    # W     - context-w mapping
    # description:
    #   initiates an  ICMDP class
    def __init__(self):
        self.__THETA = 0
        self.__THETA_mat = empty()
        self.__F = 0
        self.__F_mat = empty()
        self.__C = 0
        self.__w = 0
        self.__W = 0
        self.__THETA_set = 0
        self.__F_set = 0
        self.__C_set = 0
        self.__w_set = 0
        self.__W_set = 0
        self.__dim_features = 0
        self.__num_states = 0
        self.__num_actions = 0
        self.__dim_context = 0
        self.__state_feature_exp = empty()
    # end of function __init__
    ##################################################

    ##################################################
    # function name: set_F
    # inputs:
    # features - an object of Features class
    # description:
    #   sets the ICMDP features, F
    def set_F(self, features):
        [num_states, dim_features] = features.size()
        assert(self.__num_states == 0 or self.__num_states == num_states), "number of states must match"
        assert(self.__dim_features == 0 or self.__dim_features == dim_features), "dimension of features must match"
        self.__dim_features = dim_features
        self.__num_states = num_states
        self.__F = features
        self.__F_set = 1
    # end of function set_F
    ##################################################

    ##################################################
    # function name: set_THETA
    # inputs:
    # transitions - an object of Transitions class
    # description:
    #   sets the ICMDP transitions, THETA
    def set_THETA(self, transitions):
        [num_states, num_actions] = transitions.size()
        assert(self.__num_states == 0 or self.__num_states == num_states), "number of states must match"
        assert(self.__num_actions == 0 or self.__num_actions == num_actions), "number of actions must match"
        self.__num_actions = num_actions
        self.__num_states = num_states
        self.__THETA = transitions
        self.__THETA_set = 1
    # end of function set_THETA
    ##################################################

    ##################################################
    # function name: set_W
    # inputs:
    # W - a matrix of the linear mapping between the context to w
    # description:
    #   sets the ICMDP W
    def set_W(self, W):
        [rows, columns] = W.shape
        assert(self.__dim_features == 0 or self.__dim_features == columns), "feature dimensions must match"
        assert(self.__dim_context == 0 or self.__dim_context == rows), "context dimensions must match"
        self.__dim_features = columns
        self.__dim_context = rows
        self.__W = W
        self.__W_set = 1
        # end of function set_W
    ##################################################

    ##################################################
    # function name: set_w
    # inputs:
    # w - a vector that linear maps the feature vector to reward
    # description:
    #   sets the ICMDP w
    def set_w(self, w):
        assert(self.__dim_features == 0 or self.__dim_features == len(w)), "feature dimensions must match"
        self.__w = w
        self.__w_set = 1
        # end of function set_w
    ##################################################

    ##################################################
    # function name: set_C
    # inputs:
    # context - a context
    # description:
    #   sets the ICMDP context, C
    def set_C(self, context):
        assert(self.__dim_context == 0 or self.__dim_context == len(context)), "context dimensions must match"
        self.__C = context
        self.__C_set = 1
    # end of function set_C
    ##################################################

    ##################################################
    # function name: solve_MDP
    # inputs:
    # w - a possibility to use an external w
    # description:
    #   finds the policy and the feature expectations for the MDP defined by F, THETA, w
    def solve_MDP(self, gamma, tol = 0.001, w=empty(), flag = 'init', state_feature_exp = empty()):

        # make sure the environment is set:
        assert (self.__THETA_set != 0), "THETA is not set"
        assert (self.__F_set != 0), "F is not set"
        
        # check if need to create matrix forms for THETA and F:
        if isinstance(self.__THETA_mat,empty):
            self.__THETA_mat = self.__THETA.mat_form()
            self.__THETA = 0
        if isinstance(self.__F_mat,empty):
            self.__F_mat = self.__F.mat_form(num_actions = self.__num_actions)
            self.__F = 0

        # check if need to solve with a given w or with the one defined in the object:
        if (isinstance(w,empty)):
            assert (self.__w_set != 0), "w is not set"
            w = self.__w

        # check if need to initialize a random per-state feature expectations or use a given/accumulated one:
        if (isinstance(state_feature_exp,empty)):
            new_state_feature_exp = np.random.rand(self.__num_states, self.__dim_features)
        # elif (isinstance(state_feature_exp,empty)):
        #     new_state_feature_exp = self.__state_feature_exp
        else:
            new_state_feature_exp = state_feature_exp

        # initiate policy and values:
        policy = np.zeros(self.__num_states,dtype='int')
        policy_ind = np.zeros(self.__num_states,dtype='int')
        current_V = np.transpose(np.matmul(new_state_feature_exp,w))
        next_V = np.zeros(current_V.shape)
        policy_gap = self.__num_actions * np.asarray(range(self.__num_states))
        delta = tol + 1

        # iterate until delta < tol:
        while (delta > tol):
            # calculate the reward for each state-action pair:
            Q = self.__F_mat + gamma *np.matmul(self.__THETA_mat , new_state_feature_exp)
            Q_w = np.matmul(Q,w) #.reshape(self.__num_actions,self.__num_states)

            # calculate the value of the current policy and the policy:
            next_V = np.reshape(np.amax(np.reshape(Q_w,[self.__num_states,self.__dim_features]),axis=1),next_V.shape)
            policy = np.reshape(np.argmax(np.reshape(Q_w,[self.__num_states,self.__dim_features]), axis=1),policy.shape)
            policy_ind = policy + policy_gap

            # update the per-state feature expectations:
            new_state_feature_exp = Q[policy_ind,:].copy()

            # update delta and current_V:
            delta = np.amax(np.absolute(current_V - next_V))
            current_V = next_V.copy()

        # return the feature expectations according to initial state distribution (uniform or a specific initial state):
        if flag == 'init':
            M = new_state_feature_exp[0,:]
        if flag == 'uniform':
            M = (1.0/float(self.__num_states))*np.sum(new_state_feature_exp,axis=0)

        # update the object's per-state feature expectations:
        self.__state_feature_exp = new_state_feature_exp

        return MDP_SOL(new_state_feature_exp,policy,M)

    # end of function solve_MDP
    ##################################################

    ##################################################
    # function name: feature_expectations_opt
    # inputs:
    # W     - a vector of inputs
    # description:
    #   for usage with optimization tools
    def feature_expectations_opt(self, W, gamma, contexts, expert_mus, mode, tol = 0.0001, flag = 'init',state_feature_exp = empty()):
        print('iter')
        # reconstruct W from the given vector:
        W_mat = np.zeros([self.__dim_context,self.__dim_features])
        for i in range(self.__dim_context):
            W_mat[i,:] = W[i*self.__dim_features : i*self.__dim_features + self.__dim_features]/sum(abs(W[i*self.__dim_features : i*self.__dim_features + self.__dim_features]))

        # check that the number of contexts matches the number of expert feature expectations:
        assert(len(contexts) == len(expert_mus))

        # return the summed difference between the values:
        if mode == 'value':
            diff = 0
            for i in range(len(contexts)):
                M = self.solve_MDP(gamma=gamma, tol=tol, w=np.matmul(contexts[i],W_mat),flag = flag,state_feature_exp=state_feature_exp).M
                diff = diff + (contexts[i] @ W_mat @ (expert_mus[i] - M))**2
            print(diff)
            return diff

        # return the summed differece between the feature expectations:
        elif mode == 'mu':
            mu_diff = 0 #np.zeros(self.__dim_features)
            for i in range(len(contexts)):
                M = self.solve_MDP(gamma=gamma, tol=tol, w=np.matmul(contexts[i],W_mat),flag = flag,state_feature_exp=state_feature_exp).M
                mu_diff = mu_diff + (expert_mus[i] - M).transpose() @ (expert_mus[i] - M)
            print(mu_diff)
            return mu_diff

    # end of function feature_expectations_opt
    ##################################################

    ##################################################
    # function name: solve_CMDP
    # inputs:
    # context - a possibility to use an external context
    # W       - a possibility to use an external W
    # description:
    #   finds the policy and the feature expectations for the MDP defined by F, THETA, w
    def solve_CMDP(self, gamma, tol = 0.001, context = empty(), W = empty(), flag = 'init',state_feature_exp = empty()):

        # check inputs:
        if (isinstance(W,empty)):
            assert (self.__W_set != 0), "W is not set"
            W = self.__W

        if (isinstance(context,empty)):
            assert (self.__C_set != 0), "C is not set"
            context = self.__C

        # use the MDP solver to calculate where the context an W used to calculate w:
        return self.solve_MDP(gamma=gamma, tol=tol, w=np.matmul(context,W),flag = flag,state_feature_exp=state_feature_exp)

    # end of function solve_CMDP
    ##################################################

    ##################################################
    # function name: solve_IMDP
    # inputs:
    # M - the feature expectations
    # description:
    #   implementation of MWAL
    #  i finds the policy and/or w for the MDP defined by F, THETA,
    #   using the trajectories/ feature expectations of an expert
    def solve_IMDP(self, gamma, T, E, tol=0.001, state_feature_exp=empty(), flag = 'init'):

        assert (self.__THETA_set != 0), "THETA is not set"
        assert (self.__F_set != 0), "F is not set"

        B = 1. / (1 + np.sqrt(2 * np.log(self.__dim_features) / T))

        W = np.ones([self.__dim_features, 1])
        w = W/ np.sum(W)

        if (isinstance(state_feature_exp,empty)):
            new_state_feature_exp = np.random.rand(self.__num_states, self.__dim_features)
        WW = np.zeros([T,self.__dim_features])
        M = np.zeros([T,self.__dim_features])
        PP = np.zeros([T,self.__num_states])
        for i in tqdm(range(T)):
            expert = self.solve_MDP(gamma,tol, w=w, flag=flag, state_feature_exp=new_state_feature_exp)
            X = (1.0/4.0)*(((1.0 - gamma) * (M[i] - E)) + 2.0*np.ones([1, self.__dim_features]))
            W = (np.multiply(W, np.power(B,np.transpose(X))))

            w = W/ np.sum(W)
            WW[i,:] = w.transpose()
            M[i,:] = expert.M

            PP[i,:] = expert.policy
        return IMDP_SOL(policies=PP,M=M,WW=WW)
    # end of function solve_IMDP
    ##################################################

    ##################################################
    # function name: solve_ICMDP
    # inputs:
    # M_n -  a list of feature expectations
    # C_n - a list of contexts in a corresponding order to M_n
    # description:
    #
    def solve_ICMDP(self, gamma, contexts, expert_feature_exp, weights, tol = 0.00001,flag = 'init',state_feature_exp = empty(),algo='norm2', iterations = 25, suffix = ""):  #(self, M_n,C_n, alpha):

        # define the function for the algorithm's input:
        feature_exp = lambda w : self.solve_MDP(gamma=gamma,tol=tol,w=w,flag=flag,state_feature_exp=state_feature_exp)

        # use constraint generation:
        # old ver (linear):
        #if (algo == 'linear'):
        #    return constraint_generation_solver(contexts=contexts,expert_feature_expectations=expert_feature_exp,weights=weights,feature_expectations=feature_exp)
        
        # new ver (norm2):
        #elif (algo == 'norm2'):
        if (algo == 'norm2'):
            return constraint_generation_solver_3(contexts=contexts,expert_feature_expectations=expert_feature_exp,feature_expectations=feature_exp, iterations = iterations, suffix=suffix)
        
        else:
            print('Algorithm ' + algo + 'is not defined.')
    # end of function solve_ICMDP
    ##################################################

# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.qValues     = {}
    #self.setEpsilon(args['epsilon'])
    #self.setLearningRate(args['alpha'])
    #self.setDiscount(args['gamma'])

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    if self.qValues.has_key(state) and len(self.qValues[state]) > 0:
        qValuesList = self.qValues[state]
        if qValuesList.has_key(action):
            return qValuesList[action]
        else:
            return 0.0 
    else:
        return 0.0


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """


    actionReward = float('-inf')
    for action in self.getLegalActions(state):
        expectedQVal = self.getQValue(state, action)
        if actionReward < expectedQVal:
            actionReward = expectedQVal
    
    if actionReward == float('-inf'):
        return 0.0
    
    return actionReward 

  def setQValue(self, state, action, updatedValue):
    if not self.qValues.has_key(state):
        self.qValues[state] = {}

    if not self.qValues[state].has_key(action):
        self.qValues[state][action] = 0
    
    self.qValues[state][action] = updatedValue
    return updatedValue 

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    if len(self.getLegalActions(state)) == 0:
        return None

    maxActionValue = float('-inf')
    maxAction = None
    for action in self.getLegalActions(state):
        val = self.getQValue(state, action)
        if maxActionValue == val:
            randPolicy = random.choice([maxAction, action])
            if randPolicy == action:
                maxAction = action
                maxActionValue = val
        elif maxActionValue < val:
            maxActionValue = val
            maxAction = action
    

    return maxAction

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    isHeads = util.flipCoin(self.epsilon)
    
    if len(legalActions) is 0:
        return None

    if isHeads:
        #print "Taking the known policy"
        return random.choice(legalActions)
    else:
        #print "Taking the random choice"
        return self.getPolicy(state)
        

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    alpha = self.alpha
    gamma = self.discount
    oldQValue = self.getQValue(state,action)
    valueSPrimeAPrime = self.getValue(nextState)

    newQValue = (1 - alpha)*oldQValue + alpha*(reward + gamma*valueSPrimeAPrime)
    self.setQValue(state, action, newQValue)
    foo = 5

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    self.weights = util.Counter()
    PacmanQAgent.__init__(self, **args) 

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    finalValue = 0
    for key in self.featExtractor.getFeatures(state, action).keys():
        finalValue += self.weights[key] * self.featExtractor.getFeatures(state, action)[key]

    return finalValue

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    correction = (reward + (self.discount*self.getValue(nextState))) - self.getQValue(state, action)
    for key in self.featExtractor.getFeatures(state, action).keys():
        self.weights[key] = self.weights[key] + self.alpha*correction*self.featExtractor.getFeatures(state, action)[key]

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      for weight in self.weights.keys():
          print "Weight: %s; Value %2.2f" % (str(weight), float(self.weights[weight]))
      pass

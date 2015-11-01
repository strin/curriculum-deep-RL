# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    
    for i in range(0, self.iterations):
        #import copy
        newValues = util.Counter()
        for state in self.mdp.getStates():  
            if self.mdp.isTerminal(state):
                newValues[state] = 0
                continue
            
            maxActionValue = -1*float('inf')
            maxAction = None
            possibleActions = self.mdp.getPossibleActions(state)
            if not possibleActions:
                newValues[state] = 0

            for action in possibleActions:
                actionSumSPrime = self.getQValue(state, action)
                            
                #Find the maximum action
                if maxActionValue < actionSumSPrime:
                    maxAction = action
                    maxActionValue = actionSumSPrime

            v_kPlus1 = maxActionValue
            newValues[state] = v_kPlus1
        self.values = newValues
                  
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    actionSumSPrime = 0
    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
        TransitionProb = transition[1]
        statePrime = transition[0]
        gamma = self.discount
        reward = self.mdp.getReward(state, action, statePrime) 
        actionSumSPrime += TransitionProb * (reward + (gamma * self.values[statePrime]))

    return actionSumSPrime

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    maxActionValue = -1*float('inf')
    maxAction = None
    possibleActions = self.mdp.getPossibleActions(state)

    if not possibleActions or self.mdp.isTerminal(state):
        return None

    for action in possibleActions:
        actionSumSPrime = self.getQValue(state, action)
                    
        #Find the maximum action
        if maxActionValue < actionSumSPrime:
            maxAction = action
            maxActionValue = actionSumSPrime

    return maxAction

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  

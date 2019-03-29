# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            states = self.mdp.getStates()
            value = util.Counter()
            for state in states:
                if not self.mdp.isTerminal(state):
                    #print('terminal')
                    actions = self.mdp.getPossibleActions(state)
                    value2 = util.Counter()
                    for action in actions:
                        value2[action] = self.computeQValueFromValues(state, action)
                    value[state] = value2[value2.argMax()]
            self.values = value


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        for nextStates in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val = q_val + nextStates[1]*(self.mdp.getReward(state, action, nextStates[0]) + self.discount*self.getValue(nextStates[0]))
        return q_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        q_val = util.Counter()
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            q_val[action] = self.computeQValueFromValues(state, action)
        return q_val.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            j = i % len(states)
            state = states[j]
            if not self.mdp.isTerminal(state):
                #print('terminal')
                actions = self.mdp.getPossibleActions(state)
                value2 = util.Counter()
                for action in actions:
                    value2[action] = self.computeQValueFromValues(state, action)
                self.values[state] = value2[value2.argMax()]

        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        priority_queue = util.PriorityQueue()
        states = self.mdp.getStates()
        list1 = {}
        
        for state in states:
            list1[state] = set()
        
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                q_val = []
                for action in actions:
                    nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
                    for nextState in nextStates:
                        if nextState[1] > 0:
                            list1[nextState[0]].add(state)
                    
                    q_val.append(self.computeQValueFromValues(state, action))
                diff = abs(self.values[state] - max(q_val))
                priority_queue.push(state, -diff)
                
                    
        for i in range(self.iterations):
            if priority_queue.isEmpty():
                break
            else:
                state = priority_queue.pop()
                
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    q_val = [self.computeQValueFromValues(state, action) for action in actions]
                    self.values[state] = max(q_val)
                
                for state2 in list1[state]:
                    actions = self.mdp.getPossibleActions(state2)
                    q_val = [self.computeQValueFromValues(state2, action) for action in actions]
                    diff = abs(self.values[state2] - max(q_val))
                    if (diff > self.theta):
                        priority_queue.update(state2, -diff)
    "*** YOUR CODE HERE ***"


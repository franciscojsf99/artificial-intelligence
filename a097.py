###Francisco Figueiredo, 89443, Grupo 097

import random
import numpy as np


LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.9

EPSILON = 0.8



# LearningAgent to implement
# no knowledeg about the environment can be used
# the code should work even with another environment
class LearningAgent:

        # init
        # nS maximum number of states
        # nA maximum number of action per state
        def __init__(self,nS,nA):

                # define this function
                self.nS = nS
                self.nA = nA
                # define this function

                #self.qTable = np.zeros((nS, nA))
                self.qTable = []
                for i in range(0 , nS):
                        self.qTable.append([0])

        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):
                # define this function
                # print("select one action to learn better")

                a = 0
                # define this function

                if self.qTable[st] == [0]:
                        for i in range(1, len(aa)):
                                self.qTable[st].append(0)
                
                if random.uniform(0, 1) < EPSILON:
                        aux = random.choice(aa)
                        a = aa.index(aux)
                else:
                        aux = []
                        if self.qTable[st] == [0]:
                            aux = random.choice(aa)
                            a = aa.index(aux)
                        else:
                            for i in range(0, len(aa)):
                                aux.append(self.qTable[st][i])
                            a = np.argmax(aux)

                return a

        # Select one action, used when evaluating
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontoexecute(self,st,aa):
                # define this function
                a = 0
                # print("select one action to see if I learned")

                aux = []
                if self.qTable[st] == [0]:
                        aux = random.choice(aa)
                        a = aa.index(aux)
                else:
                    for i in range(0, len(aa)):
                        aux.append(self.qTable[st][i])
                    a = np.argmax(aux)
                
                return a


        # this function is called after every action
        # ost - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                # define this function
                #print("learn something from this data")

                currentQ = self.qTable[ost][a]
                maxQ = np.max(self.qTable[nst][:])

                newQ = currentQ + LEARNING_RATE * (r + DISCOUNT_FACTOR * maxQ - currentQ)

                self.qTable[ost][a] = newQ

                return

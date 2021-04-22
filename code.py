from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import enchant
import collections
import string
import re
import time

import neat
from neat.activations import sigmoid_activation

TIME_CONSTANT = .1
RUNS = 5
TIME_PER_RUN =  10
INITIALREWARD = 10
INPUT = [ord('h'), ord('e'), ord('l'), ord('l'), ord('o'), 0,0,0,0,0]

def genNodes(num):
    nodes = []

    for i in range(num):
        connctions = []

        for j in range(num):
            connctions.append((j,random.uniform(-1,1)))
        
        nodes.append(connctions)
    
    return nodes


def genEvals(time_constant, activation, aggregation, biases, response, nodes):
    evals = {}

    for i in range(len(nodes)):
        evals[i] = neat.ctrnn.CTRNNNodeEval(time_constant, activation, aggregation, biases[i], response, nodes)

    return evals

def runNetwork(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, TIME_CONSTANT)

    output = ""
    time = TIME_CONSTANT
    action = net.advance(INPUT, TIME_CONSTANT, TIME_CONSTANT)
    output += floatToChar(action[0])
        
    while time < TIME_PER_RUN:
        action = net.advance([0,0,0,0,0,0,0,0,0,0], TIME_CONSTANT, TIME_CONSTANT)
        output += floatToChar(action[0])
        time+=TIME_CONSTANT

    return output

found = {}

def eval_genomes_for_words(genomes, config):
    d = enchant.Dict("en_US")
    for genome_id, genome in genomes:

        output = runNetwork(genome, config)

        words = re.split(' ',output)

        genome.fitness = 0
        seen = collections.defaultdict(int)
        chars = collections.defaultdict(int)

        for i in range(len(output)):
            if output[i].isalpha():
                chars[output[i]]+=1
                genome.fitness+=max(1,INITIALREWARD-chars[output[i]])
            if output[i] == ' ':
                if i>0 and output[i-1] != ' ':
                    genome.fitness+=75
        for word in words:
            if word != "" and word.isalpha() and d.check(word):
                if len(word) > 1 or word == 'a' or word == 'A' or word == 'I':
                    seen[word]+=500
                    if word not in found:
                        found[word] = True
                        time.sleep(3)
                    genome.fitness+=max(len(word)*1000-seen[word], 10)

def eval_genomes_for_char(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        
        output = runNetwork(genome, config)
        
        charCounts = collections.defaultdict(int)

        for i in output:
            if i.isalpha() or i == ' ':
                charCounts[i]+=1
                genome.fitness+=max(INITIALREWARD - charCounts[i],1)

def floatToChar(num):
    c = chr(int(num*1000)%127)

    if c.isalpha() or c == ' ':
        return c
    
    new = ord(c)%53

    if new < 26:
        c = chr(65+new)
    elif new < 52:
        c = chr(97+new-26)
    else:
        c = ' '
    return c

def main():
   # NUMBER_OF_NODES = 50
   # TIME_CONSTANT = .01
   # ACTIVIATION = sigmoid_activation
   # AGGREGATION = sum
   # RESPONSE = 1

   # nodes = genNodes(NUMBER_OF_NODES)

   # biases = [random.uniform(-1,1) for i in range(NUMBER_OF_NODES)]
   # evals = genEvals(TIME_CONSTANT, ACTIVIATION, AGGREGATION, biases, RESPONSE, nodes)

   # net = neat.ctrnn.CTRNN(range(10),range(NUMBER_OF_NODES-10,NUMBER_OF_NODES), evals)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
  
    winner = pop.run(eval_genomes_for_words, 300)

    for word in found:
        print(word)
    print(runNetwork(winner, config))

main()

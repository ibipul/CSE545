########################################
## Template Code for Big Data Analytics
## assignment 1 - part I, at Stony Brook Univeristy
## Fall 2017


import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from multiprocessing import Process, Manager
from pprint import pprint

import string
import re
import numpy as np
from random import random


##########################################################################
##########################################################################
# PART I. MapReduce

class MyMapReduce:  # [TODO]
    __metaclass__ = ABCMeta

    def __init__(self, data, num_map_tasks=5, num_reduce_tasks=3):  # [DONE]
        self.data = data  # the "file": list of all key value pairs
        self.num_map_tasks = num_map_tasks  # how many processes to spawn as map tasks
        self.num_reduce_tasks = num_reduce_tasks  # " " " as reduce tasks

    ###########################################################
    # programmer methods (to be overridden by inheriting class)

    @abstractmethod
    def map(self, k, v):  # [DONE]
        print("Need to override map")

    @abstractmethod
    def reduce(self, k, vs):  # [DONE]
        print("Need to override reduce")

    ###########################################################
    # System Code: What the map reduce backend handles

    def mapTask(self, data_chunk, namenode_m2r):  # [DONE]
        # runs the mappers and assigns each k,v to a reduce task
        for (k, v) in data_chunk:
            # run mappers:
            mapped_kvs = self.map(k, v)
            # assign each kv pair to a reducer task
            for (k, v) in mapped_kvs:
                namenode_m2r.append((self.partitionFunction(k), (k, v)))

    def partitionFunction(self, k):  # [TODO] --completed

        # given a key returns the reduce task to send it
        key_as_string = str(k)
        character_sum = sum([ ord(i) for i in key_as_string])
        reduce_task_number = character_sum % self.num_reduce_tasks
        return reduce_task_number



    def reduceTask(self, kvs, namenode_fromR):  # [TODO]

        # sort all values for each key (can use a list of dictionary)
        # call reducers on each key with a list of values
        # and append the result for each key to namenode_fromR
        # [TODO] --completed
        kvs_master_dict=defaultdict(list)
        for w in kvs:
            kvs_master_dict[w[0]].append(w[1])
            # except KeyError:
            #     kvs_master_dict[w[0]] = [w[1]]

        for (k, vs) in kvs_master_dict.items():
            reduced_kv = self.reduce(k, vs)
            namenode_fromR.append(reduced_kv)


    def runSystem(self):  # [TODO]--completed
        # runs the full map-reduce system processes on mrObject

        # the following two lists are shared by all processes
        # in order to simulate the communication
        # [DONE]
        namenode_m2r = Manager().list()  # stores the reducer task assignment and
        # each key-value pair returned from mappers
        # in the form: [(reduce_task_num, (k, v)), ...]
        namenode_fromR = Manager().list()  # stores key-value pairs returned from reducers
        # in the form [(k, v), ...]

        # divide up the data into chunks accord to num_map_tasks, launch a new process
        # for each map task, passing the chunk of data to it.
        # hint: if chunk contains the data going to a given maptask then the following
        #      starts a process
        #      p = Process(target=self.mapTask, args=(chunk,namenode_m2r))
        #      p.start()
        #  (it might be useful to keep the processes in a list)
        # [TODO] --completed
        chunk_list = []
        # List of data tuples: (a,b) grouped together by congruence modulo of num_map_tasks on 'a'
        for i in range(self.num_map_tasks):
            chunk_i = [ self.data[index]  for index in range(len(self.data)) if index%self.num_map_tasks == i]
            chunk_list.append(chunk_i)

        #Start mapper processes
        mapper_process_list = []
        for i in range(self.num_map_tasks):
            mproc = Process(target=self.mapTask, args=(chunk_list[i],namenode_m2r))
            mproc.start()
            mapper_process_list.append(mproc)

        # join map task processes back
        # [TODO] --completed
        for mproc in mapper_process_list:
            mproc.join()

        # print output from map tasks
        # [DONE]
        print("namenode_m2r after map tasks complete:")
        pprint(sorted(list(namenode_m2r)))

        # "send" each key-value pair to its assigned reducer by placing each
        # into a list of lists, where to_reduce_task[task_num] = [list of kv pairs]
        to_reduce_task = [[] for i in range(self.num_reduce_tasks)]
        # [TODO] --completed
        for kv_tuple in sorted(list(namenode_m2r)):
            to_reduce_task[kv_tuple[0]].append(kv_tuple[1])


        # launch the reduce tasks as a new process for each.
        # [TODO] --completed
        reducer_process_list=[]
        for i in range(self.num_reduce_tasks):
            rproc = Process(target=self.reduceTask, args=(to_reduce_task[i],namenode_fromR))
            rproc.start()
            reducer_process_list.append(rproc)

        # join the reduce tasks back
        # [TODO]--completed
        for rproc in reducer_process_list:
            rproc.join()


        # print output from reducer tasks
        # [DONE]
        print("namenode_m2r after reduce tasks complete:")
        pprint(sorted(list(namenode_fromR)))

        # return all key-value pairs:
        # [DONE]
        return namenode_fromR


##########################################################################
##########################################################################
##Map Reducers:

class WordCountMR(MyMapReduce):  # [DONE]
    # the mapper and reducer for word count
    def map(self, k, v):  # [DONE]
        counts = dict()
        for w in v.split():
            w = w.lower()  # makes this case-insensitive
            try:  # try/except KeyError is just a faster way to check if w is in counts:
                counts[w] += 1
            except KeyError:
                counts[w] = 1
        return counts.items()

    def reduce(self, k, vs):  # [DONE]
        return (k, np.sum(vs))


class SetDifferenceMR(MyMapReduce):  # [TODO] -- completed
    # contains the map and reduce function for set difference
    # Assume that the mapper receives the "set" as a list of any primitives or comparable objects
    # def map(self, k, v):
    #     set_elements = dict()
    #     for list_item in v:
    #         set_elements[list_item]=k
    #
    #     return set_elements.items()
    #
    # def reduce(self, k, vs):
    #     if len(vs) > 1:
    #         return (k,None)
    #     elif 'R' in vs:
    #         return (k,'R')
    #     else:
    #         return (k,None)

    key_name = 'SetDifference'
    #Introducing an additional key to trick the partition Function
    # to send all sets to the same reducer.
    # So that reducer can output the result as [(R,[...])]

    def map(self, k, v):
        set_elements = defaultdict(list)
        for list_item in v:
            set_elements[self.key_name] = (k,v)

        return set_elements.items()

    def reduce(self, k, vs):
        set_R = []
        set_S = []
        for (a,b) in vs:
            if a=='R':
                set_R = set_R + b
            else:
                set_S = set_S + b

        for element_from_S in set_S:
            if element_from_S in set_R:
                set_R.remove(element_from_S)

        return ('R',set_R)

##########################################################################
##########################################################################

def word_cleaner(line):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', line)

if __name__ == "__main__":  # [DONE: Uncomment peices to test]
    # ###################
    # ##run WordCount:
    data = [(1, "The horse raced past the barn fell"),
              (2, "The complex houses married and single soldiers and their families"),
              (3, "There is nothing either good or bad, but thinking makes it so"),
              (4, "I burn, I pine, I perish"),
              (5, "Come what come may, time and the hour runs through the roughest day"),
              (6, "Be a yardstick of quality."),
              (7, "A horse is the projection of peoples' dreams about themselves - strong, powerful, beautiful"),
              (8,
               "I believe that at the end of the century the use of words and general educated opinion will have altered so much that one will be able to speak of machines thinking without expecting to be contradicted."),
              (9, "The car raced past the finish line just in time."),
              (10, "Car engines purred and the tires burned.")]

    data = [(x[0], word_cleaner(x[1])) for x in data]
    mrObject = WordCountMR(data, 4, 3)
    mrObject.runSystem()

    ####################
    ##run SetDifference
    # (TODO: uncomment when ready to test)
    print("\n\n*****************\n Set Difference\n*****************\n")
    data1 = [('R', ['apple', 'orange', 'pear', 'blueberry']),
    		 ('S', ['pear', 'orange', 'strawberry', 'fig', 'tangerine'])]
    data2 = [('R', [x for x in range(50) if random() > 0.5]),
    		 ('S', [x for x in range(50) if random() > 0.75])]
    mrObject = SetDifferenceMR(data1, 2, 2)
    mrObject.runSystem()
    mrObject = SetDifferenceMR(data2, 2, 2)
    mrObject.runSystem()
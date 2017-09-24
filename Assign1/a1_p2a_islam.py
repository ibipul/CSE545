from pprint import pprint
from random import random
import re,string

from pyspark import SparkContext
sc = SparkContext('local', 'Problems 2a')

def word_cleaner(line):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', line)
#################################
# Word Count implementation here
#################################
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
lines = sc.parallelize(data)
# discarding the first part of the tuples and splitting the words
words = lines.flatMap(lambda x: word_cleaner(x[1]).split())
# Map operation to tokenize the words
tokens = words.map(lambda x: (x.lower(),1))
# Reduce operation to get word counts
word_counts =  tokens.reduceByKey(lambda a,b: a+b).collect()
pprint(sorted(list(word_counts)))


######################################
##   Set Difference implementation below
######################################

data = [('R', [x for x in range(50) if random() > 0.5]),
           ('S', [x for x in range(50) if random() > 0.75])
        ]

# Set difference implementation via filter function
def set_difference_operator(tuples):
    """
    Identifies elemets of R-S and returns a boolean
    :param tuples tuple:
    :return: True/False
    :rtype bool:
    """
    return True if 'R' in tuples[1] and len(tuples[1]) == 1 else False

rdd = sc.parallelize(data)
# mapping from (R, [x,...]), (S, [y,...]) to (x,R),...,(y,S),... form
# to treat elements as keys as there are no duplicates
set_elements = rdd.map(lambda x: [(i,x[0]) for i in x[1]]).flatMap(lambda xs: [x for x in xs])
# Effective reduce reduce step
set_diff = set_elements.groupByKey().mapValues(list).filter(set_difference_operator)
# Additional group by to flip the result to the desired output format
resultant_set = set_diff.flatMap(lambda x: [(y,x[0]) for y in x[1]])\
                .groupByKey().mapValues(list).collect()

pprint(sorted(list(resultant_set)))

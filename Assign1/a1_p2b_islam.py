from pprint import pprint
import re, os
from operator import add
from pyspark import SparkContext
sc = SparkContext('local', 'Problems 2b')

_LOCAL_DIR_PATH = "C:\\Users\\ibipul\\codes\\datasets\\blogs\\"
_LOCAL_FILES_REGEX = _LOCAL_DIR_PATH + '*'

def date_revarsal(orig_date):
    """
    Extracts yyyy-month from dd,month,YYYY strings
    :param orig_date string:
    :return: date-string
    :rtype string:
    """
    s = orig_date.split(',')
    return s[2]+'-'+s[1]

def industry_search(text_str):
    """
    Looks up a global set of industry names
    and searches the blog post strings and reports the matched words.
    Words are returned if they are in the word boundary
    :param text_str string:
    :return: List of matched words
    :rtype list:
    """
    match_list = []
    for word in industry_set_list.value:
        p = re.compile(r'\b(%s)\b' % word, re.I)
        match_list.append(p.findall(text_str))
    # match_list is a nested list, we flatten it before returning it
    flat_list = [item for sublist in match_list for item in sublist]
    return flat_list

def key_by_industry_date(date_industry_tuple):
    """
    This function sets up [((industry,date),1)...] list
    from [(date,[industry1,industry2,industyr3...], ...)
    :param date_industry_tuple:
    :return: list of industry date of post tuples
    :rtype list(tuple):
    """
    date_str = date_industry_tuple[0]
    industry_list = date_industry_tuple[1]
    key_reversal_list = []
    for industry in industry_list:
        key_reversal_list.append(((industry, date_str),1))
    return key_reversal_list


# Prob 2b part 1
files_list = os.listdir(_LOCAL_DIR_PATH)
file_names = sc.parallelize(files_list)
unique_industries_list = file_names.map(lambda x: (x.split('.')[3],1))\
    .reduceByKey(lambda a,b: a+b).map(lambda x: x[0])
# This a spark Broadcast variable
industry_set_list = sc.broadcast(set(unique_industries_list.collect()))

# Prob 2b part 2:
# Following is how the transformation works on RDD, and how they look in each stage
# data: [(file_name, contents as string) ...]
# T0: [(content_string), (content_string) ...]
# T1: [[(d-m-yyy,post1),(d-m-yyy),...],[(d-m-yyy,post1),(d-m-yyy,post),...]]
# T2: flat-list [ (yyyy-m,post),(yyyy-m,post),..., (yyyy-m,post)]
# T3: [ (yyyy-m, [ind1,ind2]), (yyyy-m, [ind_i]), ...]
# T4: [ ((ind1,yyyy-m),1),((ind2,yyyy-m),1) ...]
# final output T5:[(ind1,((yyyy-m,i),(yyyy-m,j)...)), (ind2,...]

# RDD containing all blogs
blogs = sc.wholeTextFiles(_LOCAL_FILES_REGEX)
# text extraction and white space compression
t0 = blogs.map(lambda x: re.sub('\s+',' ', x[1]))
# data tuple fetch by regex
t1 = t0.map(lambda x: re.compile('<date>(.*?)</date> <post>(.*?)</post>', re.IGNORECASE).findall(x))
# list flattening and date extraction
t2 = t1.flatMap(lambda xs: [x for x in xs]).map(lambda x: (date_revarsal(x[0]),x[1]))
# Search for industry mention list, by date and filter tuples with no industry mentions
t3 = t2.map(lambda x: (x[0], industry_search(x[1].lower()))).filter(lambda x: False if len(x[1])==0 else True)
# Set industry name (lower case) as key and flatten to list of tuples with key as (industry,date)
t4 = t3.map(lambda x: key_by_industry_date(x)).flatMap(lambda xs : [x for x in xs])
# Reduce by (industry,date) key to get counts, then set key as industry and group by key
t5 = t4.reduceByKey(add).map(lambda x: (x[0][0],(x[0][1],x[1]))).groupByKey().mapValues(tuple)
final_aggregate = t5.collect()

## Pretty print of the output
pprint(sorted(list(final_aggregate)))






#re.compile('<date>(.*?)</date>[\r\n]+<post>(.*?)</post', re.IGNORECASE).findall()
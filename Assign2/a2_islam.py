from tifffile import TiffFile
import io
import time
import zipfile
import hashlib
import numpy as np
from pprint import pprint
from scipy import linalg
from collections import defaultdict
from pyspark import SparkContext
sc = SparkContext(appName='Assignment_2: Bipul Islam')

_LOCAL_DIR_PATH = "C:\\Users\\ibipul\\codes\\datasets\\a2_small_sample\\"
_LOCAL_FILES_REGEX = _LOCAL_DIR_PATH + '*'
_CHUNK_SIZE = 38

###################
##
##  FUNCTION SECTION
##
####################
def getOrthoTif(zfBytes):
    # This function has been provided
    #given a zipfile as bytes (i.e. from reading from a binary file),
    # return a np array of rgbx values for each pixel
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")
    #find tif:
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':#found it, turn into array:
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
            return tif.asarray()

def getTiffAsMatrix(kv):
    """
    Given (file, binary contents)
    Function returns (filename, np.arrary)
    :param kv tuple:
    :return key value tuple:
    """
    file_name = kv[0].split('/')[-1]
    fbinary = kv[1]
    tiffmat = getOrthoTif(fbinary)
    return (file_name,tiffmat)

def tiffmatrixSplit(kv):
    """
    Given a (k,V) tuple with key as file name , value as 2500x2500x4 matrix
    We reduce it to component sub-images of size 500x500x4
    On receiving [(img, 2500x2500x4)]
    We return: [(img-0, 500x500x4),(img-1, 500x500x4)...(img-24, 500x500x4)] as a list.
    :param kv tupple:
    :return:
    :rtype list[]:
    """
    filename, tiffmat = kv[0], kv[1]
    # Each image is 500x500
    kv_list = []
    if len(tiffmat) == 2500:
        num_matrices = 5**2
        split_size = 5
    elif len(tiffmat) == 5000:
        num_matrices = 10**2
        split_size = 10
    else:
        raise ValueError("TIFF file has dimensions other than 2500x2500 or 5000x5000")
    all_matrices = []
    file_names = [filename + '-' + str(i) for i in np.arange(num_matrices)]
    big_rows = np.vsplit(tiffmat, split_size)
    for row in big_rows:
        all_matrices += np.hsplit(row, 5)
    return list(zip(file_names,all_matrices))

def pixelConverter(pixels):
    """
    This is the pixel smoothing function. It's applied to each 1x4 pixel element.
    :param pixels:
    :return:
    """
    intensity = int((sum(pixels[:3]) / 3) * (pixels[-1] / 100))
    return intensity

def tilePixelIntensityConverter(kv):
    """
    Given a (filename, 500x500 nd array)
    Does the intensity smoothing for each pixel to return
    (filename, 500x500 2d array)
    :param kv tuple:
    :return:
    :rtype tuple
    """
    file_name = kv[0]
    tile = kv[1]
    intensity_converted_img = np.apply_along_axis(pixelConverter, 2, tile)
    return (file_name, intensity_converted_img)

def downScaleResolution(kv, factor=10):
    """
    Given a (filename, 500x500 2d array)
    returns (filename, 500x500 2d array downscaled by factor)

    :param kv:
    :param factor:
    :return:
    """
    sub_img_name = kv[0]
    sub_image = kv[1]
    img_dimension = len(sub_image)
    big_image = sub_image
    Nbig = img_dimension
    Nsmall = Nbig//factor
    small_image = big_image.reshape([Nsmall, Nbig // Nsmall, Nsmall, Nbig // Nsmall]).mean(3).mean(1)
    return (sub_img_name,small_image)

def elementFilter(element):
    """
    Acts on each pixel of flattened arrays to threshold them to -1,0,1
    This function is called in a vectorizes fashion inside getRowColDiffFeatureVec
    :param element int: intensity of a pixel
    :return:
    :rtype int:
    """
    if element < -1:
        return -1
    elif element > 1:
        return 1
    else:
        return 0

def getRowColDiffFeatureVec(kv):
    """
    Given a (filename, scaled_down image array)
    Creates feature vectors for an image by computing row & column differences
    :param kv (tuple):
    :return:
    :rtype tuple:
    """
    img_name = kv[0]
    img = kv[1]
    row_diff = np.diff(img)
    col_diff = np.diff(img.transpose())
    feature_vector = np.concatenate((row_diff.flatten(),col_diff.flatten()))
    vfunc = np.vectorize(elementFilter)
    features = vfunc(feature_vector)
    return (img_name, features)

def applyMD5ToChunk(img_chunk,chunk_size):
    """
    This function applies md5 hash to a chunk of the feature vector.
    It is called in a vectorized fashion inside getImageSignature
    :param img_chunk:
    :param chunk_size:
    :return:
    """
    if len(img_chunk) < chunk_size:
        return ''
    else:
        chunk_signature = hashlib.md5(img_chunk.tostring()).hexdigest()
        return chunk_signature[int(len(chunk_signature)/2)]

def getImageSignature(kv,chunk_size = _CHUNK_SIZE):
    """
    Given a (filename, image_array)
    returns (filename, image_signature of length 128)
    :param kv:
    :param chunk_size:
    :return:
    :rtype (tuple)
    """
    img_name = kv[0]
    img = kv[1]
    vfunc = np.vectorize(applyMD5ToChunk)
    img_chunks = np.split(img, range(chunk_size, len(img), chunk_size))
    vf = vfunc(img_chunks, chunk_size)
    img_signature = ''.join(vf)
    return (img_name, img_signature)

def customHashFunc(str):
    """
    Given a img signature hex string, hash it to a bucket
    :param str string:
    :return: Returns has bucket of if
    :rtype int:
    """
    return sum(ord(chr) for chr in str)%128

def lshTransform(kv, band_count = 16):
    """
    Given an (filename, img_array)
    Resolves it into [((hash_bucket, band),img_name), ((hash_bucket, band),img_name),...]
    :param kv tuple:
    :param band_count int: Tune able band count
    :return : List of key value pairs
    :rtype list[]:
    """
    img_name = kv[0]
    img_sig = kv[1]
    band_hash_list = []
    band_size = int(128/band_count)
    img_sig_chunks = list(map(''.join, zip(*[iter(img_sig)]*band_size)))
    vfunc = np.vectorize(customHashFunc)
    band_hash_list = vfunc(img_sig_chunks)
    # Key: (band_id, bucket_id)
    zip_list_1 = list(zip(np.arange(band_count),band_hash_list))
    img_name_list = [[img_name]]*band_count
    return list(zip(zip_list_1,img_name_list))

def filter4Sample(kv):
    """
    Removes all key value pairs were the relevant images are not in  value list
    :param kv:
    :return:
    :rtype boolean:
    """
    val_list = kv[1]
    if '3677454_2025195.zip-0' in val_list or \
       '3677454_2025195.zip-1' in val_list or \
       '3677454_2025195.zip-18' in val_list or \
       '3677454_2025195.zip-19' in val_list:
        return True
    else:
        return False

def candidateExtraction(kv):
    """
    For key value pairs in which relevant images are present,
    flip the relevant images as key and return (k, v-list),
    where vlist is list of similar images.
    :param kv:
    :return:
    :rtype tuple:
    """
    val_list = kv[1]
    if '3677454_2025195.zip-0' in val_list:
        val_list.remove('3677454_2025195.zip-0')
        return ('3677454_2025195.zip-0', val_list)
    elif '3677454_2025195.zip-1' in val_list:
        val_list.remove('3677454_2025195.zip-1')
        return ('3677454_2025195.zip-1', val_list)
    elif '3677454_2025195.zip-18' in val_list:
        val_list.remove('3677454_2025195.zip-18')
        return ('3677454_2025195.zip-18', val_list)
    elif '3677454_2025195.zip-19' in val_list:
        val_list.remove('3677454_2025195.zip-19')
        return ('3677454_2025195.zip-19', val_list)

def aggregateSimilarityDict(collected_list):
    """
    Being given the collected list of similar images
    Runs throug it to compose a dictionary of images:
    '3677454_2025195.zip-0' ,'3677454_2025195.zip-1','3677454_2025195.zip-18','3677454_2025195.zip-19'
    as keys, value as list of similar images.
    :param collected_list list[]:
    :return:
    :rtype dict{}:
    """
    similarity_dict = defaultdict(list)
    for tup in collected_list:
        similarity_dict[tup[0]] += tup[1]
    for e in similarity_dict.keys():
        similarity_dict[e] = list(set(similarity_dict[e]))
    return dict(similarity_dict)

#--------------------------------------------
# Functions geared towards formatted display
# of 1.e, 2.f, 3.b
#--------------------------------------------
def display1e(kv):
    fil_name = kv[0]
    if fil_name in ['3677454_2025195.zip-0','3677454_2025195.zip-1','3677454_2025195.zip-18','3677454_2025195.zip-19']:
        return True
    else:
        return False

def display2f(kv):
    fil_name = kv[0]
    if fil_name in ['3677454_2025195.zip-1','3677454_2025195.zip-18']:
        return True
    else:
        return False

def display3b(similarity_dict):
    for e in similarity_dict.keys():
        if e in ['3677454_2025195.zip-1','3677454_2025195.zip-18']:
            print(e,' : ', similarity_dict[e])

###################################
##  Set of functions to compute SVD
###################################
"""
Idea behind the SVD implementation done here:
 - Here we are breaking the image into 500x 500 sub images, 
    each of which contributes to a feature vector
 - We have run a further intensity averaging, and down-sampling on each to create:
    (img_component_i, 1x4900[np.array]), essentially a Key-Value pair 
    where key is the image-component-name and value is the feature vector of shape 1x4900
    
 Broad Idea: 
 -Each Spark RDD partition is of the form: 
   [(img_comp-i, 1x4900 array),(img_comp-j, 1x4900 array),...,(img_comp-z, 1x4900 array)]
 - We run SVD to transform it to corresponding low-dimensional form:  
   [(img_comp-i, 1x10 array),(img_comp-j, 1x10 array),...,(img_comp-z, 1x10 array)]
   
 - We Use MapPartition function of pyspark to that end, to run our PartionedSVD
 - For each partition, 
        - partitionedSVD extracts the component-image-identifiers, and feature vectors
        - np.stack is used to create a nx4900 dimensional matrix for the partition -- in order.
        - We run normal SVD on this matrix, and reduce the dimension to nx10
        - We recompose the Key Value pairs to get our target RDD partitions  
"""

def f(iterator):
    for x in iterator:
            print(x)
    print("\n===\n")

def partionedSVD(iterator, low_dim_p=10):
    img_name_list = []
    row_array_list = []
    for x in iterator:
        img_name_list.append(x[0])
        row_array_list.append(x[1])
    # Assemble all rows in the partition
    # & create the Matrix
    img_mat = np.stack(row_array_list)
    # Normalizing - scaling & centering
    mu, std = np.mean(img_mat, axis=0), np.std(img_mat, axis=0)
    img_diffs_zs = (img_mat - mu) / std
    # run singular value decomposition on this .
    U, s, Vh = linalg.svd(img_diffs_zs, full_matrices=1)
    img_diffs_zs_lowdim = U[:, 0:low_dim_p]
    ## Getting ready for throw back
    transformed_partition = [(img_name_list[i],img_diffs_zs_lowdim[i]) for i in range(len(img_name_list))]
    return transformed_partition

def similarity_map(svd_collect):
    """
    When passed with collected object from Distributed SVD
    returns a dictionary with each  key as image and component block number,
    and value as the 1d array size 1x10.
    :param svd_collect list[]:
    :return:
    :rtype dict{}:
    """
    sim_map = defaultdict(list)
    for x in svd_collect:
        sim_map[x[0]] = x[1]
    return dict(sim_map)

############
##
## PY Spark Code Section
##
############
# Timing tracker
start_time = time.time()
## Read Files
rdd = sc.binaryFiles(_LOCAL_FILES_REGEX) #
#rdd = sc.binaryFiles('hdfs:/data/large_sample')
#rdd = sc.binaryFiles('hdfs:/data/small_sample')

## Obtain RDD as:[ (filename, tiffMatrix)...]
rdd2 = rdd.map(lambda kv: getTiffAsMatrix(kv))
## Split each matrix to 500x500x4 images RDD: [ (img-0, 500x500x4),...(img-n, 500x500x4)]
rdd3 = rdd2.flatMap(lambda kv: tiffmatrixSplit(kv))
## Collect operation for 1.E
data_for_print1e = rdd3.filter(lambda x:display1e(x)).collect()
## Smooth out pixels to get RDD: [(img-0,500x500), (img-1,500x500) ...]
rdd4 = rdd3.map(lambda kv: tilePixelIntensityConverter(kv))
## Call Persist on RDD at this stage
rdd4.persist()
## Call down-scale of resolution on each sub image, default factor=10
## Gives RDD[ (img-0,50x50),(img-1,50x50)...]
rdd5 = rdd4.map(lambda kv: downScaleResolution(kv))
## Computer Feature Vector for each image
rdd6 = rdd5.map(lambda kv: getRowColDiffFeatureVec(kv))

########################
##
##  Locality Sensitive Hashing section
##
########################
## Get Image signatures: [(img-0, 128char-0),(img-1, 128bsig-2)...]
rdd7 = rdd6.map(lambda kv: getImageSignature(kv))
## RUN LSH grouping
rdd8 = rdd7.flatMap(lambda kv: lshTransform(kv,16))
## reduce by (band, bucket) key
rdd9 = rdd8.reduceByKey(lambda x,y: x+y)
## Remove tuples which do not have target tuples.
rdd10 = rdd9.filter(lambda kv: filter4Sample(kv))
## Extract candidates for relevant images
rdd11 = rdd10.map(lambda kv: candidateExtraction(kv)).filter(lambda kv: True if len(kv[1])>0 else False)
print("Main Collect Step initiated for part 3...")
candidate_set = rdd11.collect()
end_time = time.time()
print("Time elapsed (upto LSH Completion): ", (end_time - start_time)/60,' mins \n')
## Collect from LSH similarity candidates
similarity_dict = aggregateSimilarityDict(candidate_set)

#########################
##
##  SVD Computation Section
##  (Idea and function description in Function section)
#########################
rx = rdd6.mapPartitions(partionedSVD)
print("SVD completion collect initiated")
d_collected = rx.collect()
similarity_map_dict = similarity_map(d_collected)

######################
##
##   Output Section
##
#######################
### 1E
print(" ----------------OUTPUT 1.E --------------\n\n")
for e in data_for_print1e:
    print('(',e[0],',',e[1][0][0],')')

print("\n\n")
### 2F
print(" ----------------OUTPUT 2.F --------------\n\n")
print("('3677454_2025195.zip-1'",rdd6.lookup('3677454_2025195.zip-1')[0],")")
print("('3677454_2025195.zip-18'",rdd6.lookup('3677454_2025195.zip-18')[0],")")
print("\n\n")
### 3B
print(" ----------------OUTPUT 3.B --------------\n\n")
display3b(similarity_dict)
print("\n\n")
### 3C
print(" ----------------OUTPUT 3.C --------------\n\n")
dist_tab = []
for image in ['3677454_2025195.zip-1', '3677454_2025195.zip-18']:
    similar_images = similarity_dict[image]
    for c in similar_images:
        dist = round(np.linalg.norm(similarity_map_dict[image] - similarity_map_dict[c]), 3)
        dist_tab.append(((image,c), dist))

pprint(sorted(dist_tab,key=lambda x: x[1]))
end_time = time.time()
print("Total Time elapsed before Extra Creds: ", (end_time - start_time)/60,' mins \n')

##  We need to run down scale with param=5, and getImageSignature with chunk_size=154
##  Feature vectors will be of size: 19800. Commenting out the bonus credit code, couldn't run on large sample.
##

###########################
##
##  Extra Credit Things
##
############################
# rdd5 = rdd4.map(lambda kv: downScaleResolution(kv,5)) # Shape is 100x100
# rdd6 = rdd5.map(lambda kv: getRowColDiffFeatureVec(kv)) # Feature Vector: 19800
# # Bonus LSH Section
# rdd7 = rdd6.map(lambda kv: getImageSignature(kv,154)) #chunk_size=154
# rdd8 = rdd7.flatMap(lambda kv: lshTransform(kv,4))
# rdd9 = rdd8.reduceByKey(lambda x,y: x+y)
# rdd10 = rdd9.filter(lambda kv: filter4Sample(kv))
# rdd11 = rdd10.map(lambda kv: candidateExtraction(kv)).filter(lambda kv: True if len(kv[1])>0 else False)
# print("Bonus Collect Step initiated for part 3...")
# candidate_set = rdd11.collect()
# end_time = time.time()
# print("Time elapsed (upto Bonus LSH Completion): ", (end_time - start_time)/60,' mins \n')
# ## Collect from LSH similarity candidates
# similarity_dict = aggregateSimilarityDict(candidate_set)
#
# ## SVD Computation Section
# rx = rdd6.mapPartitions(partionedSVD)
# print("SVD completion collect initiated")
# d_collected = rx.collect()
# similarity_map_dict = similarity_map(d_collected)
#
# ##   LSH & SVD Bonus Output Section
#
# print(" ----------------Bonus OUTPUT 3.B --------------\n\n")
# display3b(similarity_dict)
# print("\n\n")
# ### 3C
# print(" ----------------Bonus OUTPUT 3.C --------------\n\n")
# dist_tab = []
# for image in ['3677454_2025195.zip-1', '3677454_2025195.zip-18']:
#     similar_images = similarity_dict[image]
#     for c in similar_images:
#         dist = round(np.linalg.norm(similarity_map_dict[image] - similarity_map_dict[c]), 3)
#         dist_tab.append(((image,c), dist))
#
# print(sorted(dist_tab,key=lambda x: x[1]))
# end_time = time.time()
# print("Total Time elapsed since start: ", (end_time - start_time)/60,' mins \n')


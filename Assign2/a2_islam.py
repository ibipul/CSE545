from tifffile import TiffFile
import io
import time
import zipfile
import hashlib
import numpy as np
from pprint import pprint
import scipy.stats as ss
from scipy import linalg
from operator import add
from collections import defaultdict
from pyspark import SparkContext
sc = SparkContext('local', 'Assignment 2')

_LOCAL_DIR_PATH = "C:\\Users\\ibipul\\codes\\datasets\\a2_small_sample\\"
_LOCAL_FILES_REGEX = _LOCAL_DIR_PATH + '*'
_CHUNK_SIZE = 38


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

def splitTiffArray(kv):
    """
    Given a (key, image arrary) splits it into
    list of [(key, 500x500 nd array),(key, 500x500 nd array)...]
    :param kv tuple:
    :return:
    :rtype list[]
    """
    filename, tiffmat= kv[0], kv[1]
    # Each image is 500x500
    kv_list = []
    if len(tiffmat) == 2500:
        # 25 images case
        row_col_chunks = [(i,i+500) for i in range(0,2500,500)]
    elif len(tiffmat) == 5000:
        row_col_chunks = [(i, i + 500) for i in range(0, 5000, 500)]
    else:
        raise ValueError("TIFF file has dimensions other than 2500x2500 or 5000x5000")
    cell_counter = 0
    for x in row_col_chunks:
        for y in row_col_chunks:
            kv_list.append((filename+'-'+str(cell_counter),tiffmat[x[0]:x[1],y[0]:y[1]]))
            cell_counter +=1
    return kv_list

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
    intensity_converted_img = []
    for row in tile:
        row_intensities = []
        for pixel in row:
            intensity = int((sum(pixel[:3])/3) * (pixel[-1]/100))
            row_intensities.append(intensity)
        intensity_converted_img.append(row_intensities)
    return (file_name, np.array(intensity_converted_img))

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
    down_scaled_image = []
    row_col_chunks = [(i,i+factor) for i in range(0,img_dimension,factor)]
    for x in row_col_chunks:
        row_factxfact_val = []
        for y in row_col_chunks:
            img_window = sub_image[x[0]:x[1],y[0]:y[1]]
            row_factxfact_val.append(img_window.mean())
        down_scaled_image.append(row_factxfact_val)
    return (sub_img_name,np.array(down_scaled_image))

def elementFilter(element):
    """
    Acts on each pixel of flattened arrays to threshold them to -1,0,1
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
    row_diff_flattened = list(row_diff.flatten())
    col_diff_flattened = list(col_diff.flatten())
    feature_vector = row_diff_flattened + col_diff_flattened
    features = [elementFilter(x) for x in feature_vector]
    return (img_name, np.array(features))

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
    img_signature = ''
    counter = 0
    start_id = 0
    while(counter <= 128):
        img_chunk = img[start_id:(start_id + chunk_size)]
        if len(img_chunk) < chunk_size:
            break
        else:
            chunk_signature = hashlib.md5(img_chunk.tostring()).hexdigest()
            img_signature += chunk_signature[int(len(chunk_signature)/2)]
            counter += 1
            start_id += chunk_size
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
    row_size = int(128/band_count)
    for i in range(0,128,row_size):
        band_component = img_sig[i:(i+row_size)]
        band_hash_list.append(customHashFunc(band_component))
    # return something like ((file_name, band_id), bucket_number)
    return [((band_hash_list[id],id),img_name) for id in range(len(band_hash_list))]

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
            pprint(similarity_dict[e])

###################################
##  Set of function to compute SVD
###################################

def setupSVDdata(y, img_names):
    """
    Using the data set y = [(img_name-i,array[features]),(img_name-i,array[features]),...]
    We create partitions such that each partition corresponds to an image, and we have:
    [ (img_name1, array([[-1, -1,  1, ..., -1,  1, -1],...,[-1,  1,  1, ...,  1, -1, -1]]--> 2d array)),
      (img_name2, array([[-1, -1,  1, ..., -1,  1, -1],...,[-1,  1,  1, ...,  1, -1, -1]]--> 2d array)),
      ...,
      (img_namen, array([[-1, -1,  1, ..., -1,  1, -1],...,[-1,  1,  1, ...,  1, -1, -1]]--> 2d array))
    ]
    i.e here each batch is a tuple and comprises of all 500x500 sub components of original image
    stacked together row wise. The order of the rows is 0-24 as split up in the originally.

    This way we have a way to access low dimensional feature vecture by image name and component number.
    '3677454_2025195.zip-18' feature vector will be available as
    row 18 (starting from 0), of the batch which has key '3677454_2025195.zip'

    Where each tuple has an image name and a matrix composed of it's feature vectors from
    sub images as we broke them into 500x500 pieces in initial steps
    :param y:
    :param img_names:
    :return:
    """
    batches = []
    for img in img_names:
        # For each image we get all component sub image feature vectors
        batch_i = [i[0] for i in y if img in i[0]]
        array_i = [j[1] for j in y if img in j[0]]
        img_mat = np.stack(array_i)
        batches.append((img, img_mat))
    return batches

def partitionedSVD(kv, low_dim_p=10):
    """
    Partitioned SVD function.
    Recieves a (img_name, 2d array of feature vectors of component images)
    Runs SVD on it, returns (img_name, 2d array of nx10)
    :param kv:
    :param low_dim_p:
    :return:
    """
    img_name, mat = kv[0], kv[1]
    # in order for SVD to perform PCA, the columns must be standardized:
    # (also known as “zscore”: mean centers and divides by stdev)
    mu, std = np.mean(mat, axis=0), np.std(mat, axis=0)
    img_diffs_zs = (mat - mu) / std
    # run singular value decomposition on this .
    U, s, Vh = linalg.svd(img_diffs_zs, full_matrices=1)
    img_diffs_zs_lowdim = U[:, 0:low_dim_p]
    return (img_name, img_diffs_zs_lowdim)

def reattachSubImageNames(kv):
    """
    From (img_name, nd-array) tuples
    we get back [(img_name-0, row_0),(img_name-1, row_1),(img_name-2, row_2)...]
    :param kv tuple:
    :return: list of tupples
    :rtype list[]
    """
    img, mat = kv[0], kv[1]
    component_list = []
    for i in range(len(mat)):
        component_list.append((img + '-' + str(i), mat[i]))
    return component_list

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

# Timing tracker
start_time = time.time()
rdd = sc.binaryFiles(_LOCAL_FILES_REGEX) #
#rdd = sc.binaryFiles('hdfs:/data/large_sample')
#rdd = sc.binaryFiles('hdfs:/data/small_sample')
img_names = rdd.map(lambda x: x[0].split('/')[-1]).collect()
rdd2 = rdd.map(lambda kv: getTiffAsMatrix(kv))
rdd3 = rdd2.map(lambda kv: splitTiffArray(kv)).flatMap(lambda xs : [x for x in xs])
##### Print Section for 1.e #####
print1e = rdd3.filter(lambda x:display1e(x)).collect()
for e in print1e:
    print('(',e[0],',',e[1][0][0],')')
#################################
rdd4 = rdd3.map(lambda kv: tilePixelIntensityConverter(kv))
rdd4.persist()
rdd5 = rdd4.map(lambda kv: downScaleResolution(kv))
rdd6 = rdd5.map(lambda kv: getRowColDiffFeatureVec(kv))
## Print section for 2.f ########
print2f = rdd6.filter(lambda x: display2f(x)).collect()
print(print2f)
#################################
rdd7 = rdd6.map(lambda kv: getImageSignature(kv))
rdd8 = rdd7.map(lambda kv: lshTransform(kv,16)).flatMap(lambda xs : [x for x in xs])
rdd9 = rdd8.groupByKey().mapValues(list)
rdd10 = rdd9.filter(lambda kv: filter4Sample(kv))
rdd11 = rdd10.map(lambda kv: candidateExtraction(kv)).filter(lambda kv: True if len(kv[1])>0 else False)
candidate_set = rdd11.collect()
end_time = time.time()
print("Time elapsed: ", (end_time - start_time)/60,' mins \n')
## Collect from LSH similarity candidates
similarity_dict = aggregateSimilarityDict(candidate_set)
### Print 3b
display3b(similarity_dict)
###
### Fire SVD similarity computation
feature_vecs = rdd6.collect()
dat = setupSVDdata(feature_vecs,img_names)
data = sc.parallelize(dat)
data_svd = data.map(lambda kv: partitionedSVD(kv))
data_inv_map = data_svd.map(lambda kv: reattachSubImageNames(kv)).flatMap(lambda xs : [x for x in xs])
d_collected = data_inv_map.collect()
similarity_map_dict = similarity_map(d_collected)
## Distance
for image in ['3677454_2025195.zip-1', '3677454_2025195.zip-18']:
    similar_images = similarity_dict[image]
    for c in similar_images:
        print('dist(',image,',',c,'):= ', round(np.linalg.norm(similarity_map_dict[image] - similarity_map_dict[c]),3) )
end_time = time.time()
print("Time elapsed: ", (end_time - start_time)/60,' mins \n')
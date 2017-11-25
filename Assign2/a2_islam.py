from tifffile import TiffFile
import io
import time
import zipfile
import hashlib
import numpy as np
import scipy.stats as ss
from numpy.linalg import svd
from operator import add
from pyspark import SparkContext
sc = SparkContext('local/aws-cluster', 'Assignment 2')

_LOCAL_DIR_PATH = "C:\\Users\\ibipul\\codes\\datasets\\a2_small_sample\\"
_LOCAL_FILES_REGEX = _LOCAL_DIR_PATH + '*'
_CHUNK_SIZE = 38


def getOrthoTif(zfBytes):
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
    file_name = kv[0].split('/')[-1]
    fbinary = kv[1]
    tiffmat = getOrthoTif(fbinary)
    return (file_name,tiffmat)

def splitTiffArray(kv):
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
    if element < -1:
        return -1
    elif element > 1:
        return 1
    else:
        return 0

def getRowColDiffFeatureVec(kv):
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
    return sum(ord(chr) for chr in str)%128

def lshTransform(kv, band_count = 8):
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
    val_list = kv[1]
    if '3677454_2025195.zip-0' in val_list or \
       '3677454_2025195.zip-1' in val_list or \
       '3677454_2025195.zip-18' in val_list or \
       '3677454_2025195.zip-19' in val_list:
        return True
    else:
        return False

def candidateExtraction(kv):
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

# Timing tracker
start_time = time.time()
rdd = sc.binaryFiles(_LOCAL_FILES_REGEX) #
#rdd = sc.binaryFiles('hdfs:/data/small_sample')
rdd2 = rdd.map(lambda kv: getTiffAsMatrix(kv))
rdd3 = rdd2.map(lambda kv: splitTiffArray(kv)).flatMap(lambda xs : [x for x in xs])
rdd4 = rdd3.map(lambda kv: tilePixelIntensityConverter(kv))
rdd5 = rdd4.map(lambda kv: downScaleResolution(kv))
rdd6 = rdd5.map(lambda kv: getRowColDiffFeatureVec(kv))
rdd7 = rdd6.map(lambda kv: getImageSignature(kv))
rdd8 = rdd7.map(lambda kv: lshTransform(kv,16)).flatMap(lambda xs : [x for x in xs])
rdd9 = rdd8.groupByKey().mapValues(list)
rdd10 = rdd9.filter(lambda kv: filter4Sample(kv))
rdd11 = rdd10.map(lambda kv: candidateExtraction(kv)).filter(lambda kv: True if len(kv[1])>0 else False)
z= rdd11.collect()
end_time = time.time()
print("Time elapsed: ", (end_time - start_time)/60,' mins \n')
rdd12 = rdd11.reduceByKey(add).mapValues(list)
z = rdd12.collect()
end_time = time.time()
print("Time elapsed: ", (end_time - start_time)/60,' mins \n')
for i in z:
    print('Candidates similar to : ', i[0],' are: ', set(i[1]))

from tifffile import TiffFile
import io
import zipfile
import hashlib
import numpy as np
import scipy.stats as ss
from numpy.linalg import svd

_CHUNK_SIZE = 38

def readFileInBinary(filename):
    with open(filename, 'rb') as f:
        fbin = f.read()
    f.close()
    return fbin

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

def getTiffAsMatrix(file_path):
    fbinary = readFileInBinary(file_path)
    tiffmat = getOrthoTif(fbinary)
    return tiffmat

def splitTiffArray(tiffmat, filename='dummy.zip'):
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
    return [(band_hash_list[id],(id,img_name)) for id in range(len(band_hash_list))]

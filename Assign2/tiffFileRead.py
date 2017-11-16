from tifffile import TiffFile

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

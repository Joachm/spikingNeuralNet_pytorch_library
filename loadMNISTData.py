import numpy as np


def toHotEncoding(classification):
    hotEncoding = np.zeros([len(classification),np.max(classification)+1])
    hotEncoding[np.arange(len(hotEncoding)),classification]=1
    return hotEncoding

def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize
    
    data = np.fromfile(folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([nImages, width, height])

    labels = np.fromfile(folder + '/' + prefix + '-labels-idx1-ubyte',
            dtype = 'ubyte') [2 * intType.itemsize:]

    return data, labels

import os
import re

import cv2
import numpy

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                            count=int(width) * int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def read_mini_mias():
    images_tensor = numpy.zeros((322, 1024 * 1024))
    i = 0
    for dirName, subdirList, fileList in os.walk("data/all-mias/"):
        for fname in fileList:
            if fname.endswith(".pgm"):
                images_tensor[i] = numpy.ndarray.flatten(read_pgm("data/all-mias/" + fname, byteorder='<'))
                i += 1
    return images_tensor


if __name__ == "__main__":
    from matplotlib import pyplot

    x = read_mini_mias()
    # image = read_pgm("data/all-mias/mdb001.pgm", byteorder='<')
    images = numpy.zeros((322, 64, 64))
    for i in range(x.shape[0]):
        images[i] = cv2.resize(x[i].reshape(1024, 1024), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    res = cv2.resize(x[0].reshape(1024, 1024), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    print(images.shape)
    pyplot.imshow(x[0].reshape(1024, 1024), pyplot.cm.gray)
    pyplot.show()

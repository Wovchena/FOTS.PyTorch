import subprocess
import os
import numpy as np

if 'nt' == os.name:
    parent_dir = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(parent_dir, 'adaptor.lib')):
        raise RuntimeError('cl adaptor.cpp ./include/clipper/clipper.cpp /I ./include /I "C:\Programs\Python37\include" /LD /Fe:adaptor.pyd "C:\Programs\Python37\libs\python37.lib"')
else:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
        raise RuntimeError('Cannot compile lanms: {}'.format(BASE_DIR))


def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    from .adaptor import merge_quadrangle_n9 as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='float32')
    ret[:,:8] /= precision
    return ret

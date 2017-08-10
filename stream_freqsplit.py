from __future__ import print_function
from glob import glob
import multiprocessing
from os import path, makedirs
from tqdm import tqdm
import os
import numpy as np
import argparse


def freq_split(s, r=8, mask_type='boxed'):
    '''
    Split frequencies of a given 3D signal
    :param s: input signal matrix of shape (dim_0, dim_1, dim_2)
    :param r: number of modes to split
    :param mask_type: 'boxed' or 'circular'
    :return: s_lf, s_hf: low and high frequencies of the signal
    '''

    s = np.fft.fftn(s)
    dims = s.shape
    mids = [int(dims[i] / 2) for i in range(len(dims))]

    # frequency mask
    if mask_type == 'circular':
        U, V, W = np.mgrid[-mids[0]:mids[0], -mids[1]:mids[1], -mids[2]:mids[2]]
        D = np.sqrt(np.square(U) + np.square(V) + np.square(W))
        lf_mask = np.fft.ifftshift(np.less_equal(D, r * np.ones_like(D)).astype(np.complex_))
        hf_mask = np.ones_like(lf_mask, dtype=np.complex_) - lf_mask

    elif mask_type == 'boxed':
        U, V, W = np.mgrid[-mids[0]:mids[0], -mids[1]:mids[1], -mids[2]:mids[2]]
        mask = np.array([U < r, U >= -r, V < r, V >= -r, W < r, W >= -r])
        lf_mask = np.fft.ifftshift(np.all(mask, axis=0).astype(np.complex_))
        hf_mask = np.ones_like(lf_mask, dtype=np.complex_) - lf_mask

    else:
        raise Exception('mask_type undefined.')

    # apply mask
    s_lf = np.multiply(s, lf_mask)
    s_hf = np.multiply(s, hf_mask)

    s_lf = np.real(np.fft.ifftn(s_lf))
    s_hf = np.real(np.fft.ifftn(s_hf))

    return s_lf, s_hf


def process_mesh(read_path, write_path):
    # read mesh
    s = np.load(read_path)[0, :, :, :]
    s_lf, s_hf = freq_split(s, r=8)
    s_out = np.array([s_lf, s_hf])
    np.save(write_path, s_out)


def helper(args):
    return process_mesh(*args)


def base_name(str):
    """return the base name of a path, i.e folder name"""
    return os.path.basename(os.path.normpath(str))


def cond_mkdir(directory):
    """Conditionally make directory if it does not exist"""
    if not path.exists(directory):
        makedirs(directory)
    else:
        print('Directory ' + directory + ' existed. Did not create.')


if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser(description="process mesh to sdf using python multiprocessing module")
    parser.add_argument('datadir', type=str, help='Path to original dataset.')
    parser.add_argument('savedir', type=str, help='Path to saving processed dataset.')
    parser.add_argument('ncore', type=int, default=0, help='Number of cores. Enter 0 for maximum cpu cores available.')
    parser.add_argument('nmesh', type=int, default=0, help='Process only nmesh. Enter 0 to process all mesh.')
    argument = parser.parse_args()

    data_dir = argument.datadir
    save_dir = argument.savedir

    if argument.ncore == 0:
        num_cores = multiprocessing.cpu_count()
        print("Using max cpu cores = %d" % (num_cores))
    else:
        num_cores = argument.ncore

    cond_mkdir(save_dir)
    read_loc = glob(os.path.join(data_dir, "*"))
    write_loc = []

    for d_in in read_loc:
        d_out = path.join(save_dir, base_name(d_in))
        write_loc.append(d_out)

    if argument.nmesh != 0:
        read_loc = read_loc[:argument.nmesh]
        write_loc = write_loc[:argument.nmesh]

    args = []

    # retrieve mesh according to file_names with multiprocessing
    for counter, (read_path, write_path) in enumerate(zip(read_loc, write_loc)):
        args.append((read_path, write_path))

    print("Starting multithreaded processing with {0} cores.".format(multiprocessing.cpu_count()))

    pool = multiprocessing.Pool()

    with tqdm(total=len(read_loc)) as pbar:
        for _, _ in tqdm(enumerate(pool.imap_unordered(helper, args))):
            pbar.update()
    pbar.close()
    pool.close()
    pool.join()

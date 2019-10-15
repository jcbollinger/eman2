#! /usr/bin/env python

# Author: Adnan Ali, 17/06/2019 (adnan.ali@mpi-dortmund.mpg.de)
# Author: Markus Stabrin 17/06/2019 (markus.stabrin@mpi-dortmund.mpg.de)
# Author: Thorsten Wagner 17/06/2019 (thorsten.wagner@mpi-dortmund.mpg.de)
# Copyright (c) 2019 MPI Dortmund

# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

from __future__ import print_function
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy as sp
import pandas as pd
import cPickle as pickle
from EMAN2 import *
import EMAN2_cppwrap
from EMAN2 import EMNumPy

import sp_utilities
import sp_projection
import sp_statistics
import sp_filter
import mpi
import sp_applications
import argparse
import sp_fundamentals

import sp_filter

from scipy.optimize import curve_fit

location =os.getcwd()
RUNNING_UNDER_MPI = "OMPI_COMM_WORLD_SIZE" in os.environ

main_mpi_proc = 0
if RUNNING_UNDER_MPI:
    pass  # IMPORTIMPORTIMPORT from mpi import mpi_init
    pass  # IMPORTIMPORTIMPORT from mpi import MPI_COMM_WORLD, mpi_comm_rank, mpi_comm_size, mpi_barrier, mpi_reduce, MPI_INT, MPI_SUM

    mpi.mpi_init(0, [])
    my_mpi_proc_id = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
    n_mpi_procs = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
    shared_comm = mpi.mpi_comm_split_type(mpi.MPI_COMM_WORLD, mpi.MPI_COMM_TYPE_SHARED, 0, mpi.MPI_INFO_NULL)
    my_node_proc_id = mpi.mpi_comm_rank(shared_comm)
else:
    my_mpi_proc_id = 0
    n_mpi_procs = 1
    my_node_proc_id = 0


no_of_micrographs = 17
N_ITER =25
shift = 1

try:
    ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
    GLOBAL_PATH = os.path.abspath(os.path.join(__file__ ,"../../../"))
except:
    GLOBAL_PATH = os.getcwd()


def numpy2em_python(numpy_array, out=None):
    """
	Create an EMData object based on a numpy array by reference.
	The output EMData object will have the reversed order of dimensions.
	x,y,z -> z,y,x
	Arguments:
	numpy_array - Array to convert
	Return:
	EMData object
	"""
    if out is None:
        shape = numpy_array.shape[::-1]
        if len(shape) == 1:
            shape = (shape[0], 1)
        return_array = EMData(*shape)
    else:
        return_array = out
	return_view = EMNumPy.em2numpy(return_array)
	return_view[...] = numpy_array
	return_array.update()
    if out is None:
        return return_array

def return_movie_names(input_image_path):
    mic_pattern = input_image_path
    mic_basename_pattern = os.path.basename(mic_pattern)
    global_entry_dict = {}
    subkey_input_mic_path = "Input Micrograph Path"

    mic_basename_tokens = mic_basename_pattern.split('*')
    mic_id_substr_head_idx = len(mic_basename_tokens[0])

    input_mic_path_list = glob.glob(mic_pattern)

    for input_mic_path in input_mic_path_list:
        # Find tail index of  id substring and extract the substring from the  name
        input_mic_basename = os.path.basename(input_mic_path)
        mic_id_substr_tail_idx = input_mic_basename.index(mic_basename_tokens[1])
        mic_id_substr = input_mic_basename[mic_id_substr_head_idx:mic_id_substr_tail_idx]
        if not mic_id_substr in global_entry_dict:
            global_entry_dict[mic_id_substr] = {}
        global_entry_dict[mic_id_substr][subkey_input_mic_path] = input_mic_path

    print(" ")
    print("\n Summary of dataset consistency check...")
    print(("  Detected  IDs               : %6d" % (len(global_entry_dict))))
    print(("  Entries in input directory  : %6d" % (len(input_mic_path_list))))


    valid_mic_id_substr_list = []
    for mic_id_substr in global_entry_dict:
        mic_id_entry = global_entry_dict[mic_id_substr]
        valid_mic_id_substr_list.append(mic_id_substr)

    input_file_path_list = []
    for mic_id_substr in sorted(valid_mic_id_substr_list):
        mic_path = global_entry_dict[mic_id_substr][subkey_input_mic_path]
        input_file_path_list.append(mic_path)

    print(("Found %d micrographs in %s." % (len(input_mic_path_list), os.path.dirname(mic_pattern))))
    return input_file_path_list


"""
Reads the x and y shifts values per frame in a micrograph  
"""
def returns_values_in_file(f, mode = 'r'):
    """
    read a file and returns all its lines
    :param f: path to file
    :param mode: how open the file. Default: read file
    :return: contained values
    """
    if os.path.isfile(f):
        with open(f, mode) as f1:
            values_f1 = f1.readlines()
        return values_f1
    print ("ERROR> the given file '"+str(f)+"' is not present!")
    exit(-1)

def read_meta_shifts(f):
    x = []
    y  = []
    for row in returns_values_in_file(f):
        if "image #" in row:
            v=row.replace('\n','').split('=')[1].replace(' ', '').split(',')
            x.append(float(v[0]))
            y.append(float(v[1]))
        elif "Frame (" in row:
            v = row.split()
            x.append(float(v[-2]))
            y.append(float(v[-1]))
    return x,y



def read_all_attributes_from_stack(stack):
    no_of_imgs_once = EMUtil.get_image_count(stack)  # Counting how many are there in the stack
    # -------Extracting the information from the substack
    ptcl_source_images_once = EMUtil.get_all_attributes(stack, 'ptcl_source_image')
    project_params_all_once = EMUtil.get_all_attributes(stack, "xform.projection")
    particle_coordinates_all_once = EMUtil.get_all_attributes(stack, "ptcl_source_coord")
    ctf_params_all_once = EMUtil.get_all_attributes(stack, "ctf")

    nx_all_once = EMUtil.get_all_attributes(stack, 'nx')
    ny_all_once = EMUtil.get_all_attributes(stack, 'ny')
    nz_all_once = EMUtil.get_all_attributes(stack, 'nz')


    return no_of_imgs_once, ptcl_source_images_once, project_params_all_once, \
           particle_coordinates_all_once, ctf_params_all_once, nx_all_once, ny_all_once, nz_all_once

def find_particles_info_from_movie(stack, movie_name, no_of_imgs, ptcl_source_images, project_params_all,
                                   particle_coordinates_all, ctf_params_all, nx_all, ny_all, nz_all,
                                   source_n_all, chunkval, adnan_n_all, show_first = False, use_chunk = True):

    #-----------------------------------------------   CTF related attributes
    """
    defocus = defocus associated with the image, positive value corresponds to underfocus
    cs =  spherical aberration constant [mm].
    voltage = accelerating voltage of the microscope [kV]
    apix = angular pixel size
    bfactor = The parameter in Gaussian like envelope function, which roughly explains
                Fourier factor dumping of the image.
    ampcont = amplitude contrast
    dfdiff  = astigmatism amplitude
    dfang =  astigmatism angle
    """

    # -------------------------------------------  2D orientation / orientation attributes
    """
    phi =  Eulerian angle for 3D reconstruction (azimuthal) 
    theta = Eulerian angle for 3D reconstruction (tilt) 
    psi = Eulerian angle for 3D reconstruction (in-plane rotation of projection) 
    tx =  shift in x direction
    ty = shift in y direction
    """
    print("Number of images in the substack are %d" % len(ptcl_source_images))

    project_params_per_movie = []
    particle_coordinates_per_movie = []
    ctf_params_per_movie = []
    nx_per_movie = []
    ny_per_movie = []
    nz_per_movie = []
    source_n_per_movie = []
    adnan_per_movie = []

    if str(os.path.basename(movie_name)).split('.')[-1] == 'mrcs':
        for i in range(no_of_imgs):
            if (
                    str(os.path.basename(movie_name)) ==
                    str(os.path.basename(ptcl_source_images[i])) or
                    str(os.path.basename(movie_name)) ==
                    str(os.path.basename(ptcl_source_images[i]))+'s'
                    ):
                project_params_per_movie.append(project_params_all[i])
                particle_coordinates_per_movie.append(particle_coordinates_all[i])
                ctf_params_per_movie.append(ctf_params_all[i])
                nx_per_movie.append(nx_all[i])
                ny_per_movie.append(ny_all[i])
                nz_per_movie.append(nz_all[i])
                source_n_per_movie.append(source_n_all[i])
                adnan_per_movie.append(adnan_n_all[i])

    elif str(os.path.basename(movie_name)).split('.')[-1] == 'tiff':
        for i in range(no_of_imgs):
            if (
                    str(os.path.basename(movie_name)).split('.tiff')[0] ==
                    str(os.path.basename(ptcl_source_images[i])).split('.mrc')[0] or
                    str(os.path.basename(movie_name)).split('.tiff')[0] ==
                    str(os.path.basename(ptcl_source_images[i])).split('.mrcs')[0]
            ):
                if i in chunkval and use_chunk == True:
                    project_params_per_movie.append(project_params_all[i])
                    particle_coordinates_per_movie.append(particle_coordinates_all[i])
                    ctf_params_per_movie.append(ctf_params_all[i])
                    nx_per_movie.append(nx_all[i])
                    ny_per_movie.append(ny_all[i])
                    nz_per_movie.append(nz_all[i])
                    source_n_per_movie.append(source_n_all[i])
                    adnan_per_movie.append(adnan_n_all[i])
                elif use_chunk == False:
                    project_params_per_movie.append(project_params_all[i])
                    particle_coordinates_per_movie.append(particle_coordinates_all[i])
                    ctf_params_per_movie.append(ctf_params_all[i])
                    nx_per_movie.append(nx_all[i])
                    ny_per_movie.append(ny_all[i])
                    nz_per_movie.append(nz_all[i])
                    source_n_per_movie.append(source_n_all[i])
                    adnan_per_movie.append(adnan_n_all[i])

                else:
                    continue
    if np.array(ctf_params_per_movie).size == 0:
        return False

    if str(os.path.basename(movie_name)).split('.')[-1] == 'mrcs':
        print("Number of particles detected in %s are %d" % (str(os.path.basename(movie_name)),
                                                             len(project_params_per_movie)))
        print("Ctf estimation parameters for 1st particle in the stack are ", ctf_params_per_movie[0].to_dict())
        print("Projection parameters for 1st particle in the stack are ", project_params_per_movie[0].get_params('spider'))
        print("Dimensions x for all particles are ", nx_per_movie)
        print("Dimensions y for all particles are ", ny_per_movie)
        print("Dimensions z for all particles are ", nz_per_movie)
    elif str(os.path.basename(movie_name)).split('.')[-1] == 'tiff':

        print('Ctf shape',np.array(ctf_params_per_movie).shape)


        print("Number of particles detected in %s are %d" % (str(os.path.basename(movie_name)),
                                                             len(project_params_per_movie)))
        print("Ctf estimation parameters for 1st particle in the stack are ", ctf_params_per_movie[0].to_dict())
        print("Projection parameters for 1st particle in the stack are ", project_params_per_movie[0].get_params('spider'))
        print("Dimensions x for all particles are ", nx_per_movie[0])
        print("Dimensions y for all particles are ", ny_per_movie[0])
        print("Dimensions z for all particles are ", nz_per_movie[0])

    if show_first:
        ima = EMAN2_cppwrap.EMData()
        ima.read_image(stack, 0, False)
        plt.ion()
        plt.figure()
        plt.imshow(ima.get_2dview(), cmap=plt.get_cmap('Greys'))
        plt.colorbar()
        plt.show()

    return project_params_per_movie, particle_coordinates_per_movie, ctf_params_per_movie, \
           nx_per_movie, ny_per_movie, nz_per_movie, source_n_per_movie, adnan_per_movie

"""
Reading a reference map
"""
def get_2D_project_for_all_ptcl_from_reference(volume_ft ,
                                               kb_fu,
                                               project_params_in_stack,
                                               frames_length,
                                               shift_in_x,
                                               shift_in_y,
                                               show = False):

    project_2D_per_movie = []
    for j in range (frames_length):
        project_2D_per_frame = []
        for i in range(len(project_params_in_stack)):
            params_substack = project_params_in_stack[i].get_params('spider')
            params_for_each_image = [params_substack['phi'], params_substack['theta'], params_substack['psi'],
                                     params_substack['tx'] - shift_in_x[j], params_substack['ty'] - shift_in_y[j] ]
            project_2D_per_frame.append(sp_projection.prgs(volume_ft, kb_fu, params_for_each_image))
        project_2D_per_movie.append(project_2D_per_frame)

    if show:
        plt.ion()
        plt.figure()
        plt.imshow(project_2D_per_movie[0][0].get_2dview(), cmap = plt.get_cmap('Greys'))
        plt.colorbar()
        plt.show()
    return project_2D_per_movie


"""
Extracting particle image from the movie data. First getting the particle cordinates from the dictionary and then 
creating a window around to extract the same particle from each frame
"""

#----------------------- Particle cordinate
def get_all_polish_ptcl_imgs( movie_name,
                              nxx, nyy,
                              part_cord,
                              zsize,
                              shift_x,
                              shift_y,
                              weights,
                              gainrefname,
                              ctf_par,
                              newshifts):
    """Getting dimension information"""
    current_frame = EMData()
    current_frame.read_image(movie_name, 0)
    xsize = current_frame.get_xsize()
    ysize = current_frame.get_ysize()

    cen_xx = xsize // 2
    cen_yy = ysize // 2

    gainref = EMData()
    gainref.read_image(gainrefname,0)

    Util.mul_scalar(gainref, -1.0)

    print("Gain shape", gainref.get_3dview().shape)
    print("Image shape", current_frame.get_3dview().shape)

    particle_imgs_in_movie = []

    # w = current_frame.get_2dview().shape[0]
    # h = current_frame.get_2dview().shape[1]
    # wt = 2 * (w - 1)
    # var = 0.0
    # cnt = 0.0
    # rr = (w - 2) * (w - 2)
    # scale = 0.0

    # movielist = []
    # for i in range (zsize):
    #     current_frame.read_image(movie_name, i)
    #     current_frame = Util.muln_img(current_frame, gainref)
    #     current_frame =  sp_fundamentals.fshift(current_frame, int(round(shift_x[i])), int(round(shift_y[i])))
    #
    #
    #     four_img = np.fft.fft2(current_frame.get_2dview())
    #     four_img2 = np.fft.fftshift(four_img)
    #
    #     movielist.append(four_img2.real)
    #
    #     imgreal = four_img2.real
    #     imgimag = four_img2.imag
    #
    #     imgnorm = np.multiply(imgreal, imgreal) + np.multiply(imgimag, imgimag)
    #
    #     del four_img
    #     del four_img2
    #     del imgreal
    #     del imgimag


    #     for y in range(h):
    #         for x in range(w):
    #             if x==0 and y==0:
    #                 continue
    #
    #             if x > 0 :
    #                 scale = 2.0
    #             else:
    #                 scale = 1.0
    #
    #             var += scale * imgnorm[x,y]
    #             cnt += scale
    #
    # scale = np.sqrt(np.divide(wt * h * var, cnt * zsize))
    # del var
    # del cnt
    # del imgnorm

    mask = sp_utilities.model_circle((nxx / 2) + 32, nxx, nxx)

    for i in range(zsize):
        current_frame.read_image(movie_name, i)
        current_frame = Util.muln_img(current_frame, gainref)

        shift_x[i] *= 0.8849999904632568
        shift_y[i] *= 0.8849999904632568

        # shift_x[i] *= 1.244531000
        # shift_y[i] *= 1.244531000

        # current_frame =  sp_fundamentals.fshift(current_frame, int(round(shift_x[i])),
        #                                         int(round(shift_y[i])))


        # np.divide(movielist[i], scale, out=movielist[i])
        # doseweight_framesi = np.fft.ifftshift(movielist[i])
        # doseweight_frames = np.fft.ifft2( doseweight_framesi ).real

        # numpy2em_python(doseweight_frames, out=current_frame)

        # del doseweight_framesi
        # del doseweight_frames

        line = weights[i, :, 0].tolist()
        crop_imgs = []
        new_coord = []
        bad_particles = []
        for j in range(len(part_cord)):
            try:
                box_img = Util.window(current_frame, nxx, nyy, 1, part_cord[j][0] - cen_xx,
                                             part_cord[j][1] - cen_yy ,0)

                ptclmask = copy.deepcopy(mask)
                # ptclmask = sp_fundamentals.fshift(ptclmask,
                #                                     int(round(shift_x[i])),
                #                                    int(round(shift_y[i])))

                # box_img = Util.muln_img(box_img, ptclmask)

                st = Util.infomask(box_img, ptclmask, False)
                box_img +=  2 * st[0]

                st = Util.infomask(box_img, ptclmask, False)
                box_img -= st[0]


                # newshifts[j][i, 0] *= 0.8849999904632568
                # newshifts[j][i, 1] *= 0.8849999904632568
                #
                box_img = sp_fundamentals.fshift(box_img, -int(round(newshifts[j][i,0])),
                                                  -int(round(newshifts[j][i,1])))

                box_img = sp_filter.filt_table(box_img, line)

                crop_imgs.append(box_img)
                new_coord.append(j)
                del box_img
            except RuntimeError:
                bad_particles.append(j)
                continue

        particle_imgs_in_movie.append(crop_imgs)
        del crop_imgs
        del line
    del current_frame

    return particle_imgs_in_movie , new_coord, bad_particles



def givemerealbfactors():
    bfile = np.loadtxt('/home/adnan/PycharmProjects/Dose_Fresh/sphire/bin/bfactors.star')
    slope = bfile[:,1]
    intercept = bfile[:,2]
    bfacy = np.exp(intercept)
    # angpix_ref = 1.244531000
    # angpix_out = 1.244531000
    angpix_ref = 0.8849999904632568
    angpix_out = 0.8849999904632568
    sh_ref = 129
    s_ref = 256
    fc = 24
    s_out = 256
    cf = 8.0 * ((angpix_ref)**2)  * ((sh_ref)**2)
    bfacx = np.sqrt(np.divide(-cf, slope))

    return bfacx, bfacy

def computeweights(bfax, bfay):
    fc = 24
    kc2 = 256
    kc = 129
    output = np.zeros((fc, kc, kc2))
    yy =0
    for f in range (fc):
        for y in range (0, kc2):
            for x in range(0, kc):
                if y < kc :
                    yy = y
                else:
                    yy == y - kc2
                r2 = (x*x) + (yy*yy)
                output[f,x,y] =  bfay[f] * np.exp(  np.divide((-0.5 * r2), bfax[f] * bfax[f] ))
    return output

bfx , bfy = givemerealbfactors()
freqweights = computeweights(bfx, bfy)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'movies_folder',
    type=str,
    help='Movies folder'
)

parser.add_argument(
    'log_folder',
    type=str,
    help='Log files folder'
)

parser.add_argument(
    'stack_file',
    type=str,
    help='Substack file'
)

parser.add_argument(
    'ref_volume_0',
    type=str,
    help='Reference volume 0 of the particle'
)

parser.add_argument(
    'ref_volume_1',
    type=str,
    help='Reference volume 1 of the particle'
)

parser.add_argument(
    'mask_0',
    type=str,
    help = 'Mask for reference volume 0'
)

parser.add_argument(
    'mask_1',
    type=str,
    help ='Mask for reference volume 1'
)

parser.add_argument(
    'chunck_vol_0',
    type=str,
    help = 'Chunck file for reference volume 0'
)

parser.add_argument(
    'chunck_vol_1',
    type=str,
    help = 'Chunck file for reference volume 1'
)


parser.add_argument(
    'saved_folder',
    type=str,
    help='Location of folder where new mrcs files and bdb will be stored'
)

parser.add_argument(
    'no_of_mic',
    type=int,
    help='Number of micrographs to be processed'
)

args = parser.parse_args()
no_of_micrographs = args.no_of_mic

Gainreffile = os.path.join( os.path.abspath(os.path.join(str(args.movies_folder), os.pardir)), 'gain.mrc')
movie_names = return_movie_names(str(args.movies_folder))

#-------------------------------Reading corrected sums log files for reading the global shifts applied

log_movie_path = args.log_folder
shift_movie_files = return_movie_names(args.log_folder)
stackfilename = "bdb:" + args.stack_file
stack_absolute_path = args.saved_folder
# ----------------------------  Loading the reference volume
# ref_volume = sp_utilities.pad(ref_volume, pad_size, pad_size, pad_size)
# ----------------------------- Preparing the volume for calculation of projections
# transforms the volume in fourier space and expands the volume with npad and centers the volume and returns the volume
print("Reading all the parameters from stack")
source_n_ind_all = EMUtil.get_all_attributes(stackfilename, "source_n")
adnan_all = EMUtil.get_all_attributes(stackfilename, 'adnan_n')
no_of_imgs, ptcl_source_images, project_params_all, particle_coordinates_all, \
            ctf_params_all, nx_all, ny_all, nz_all = read_all_attributes_from_stack(stackfilename)

print("Finding Non existing .mrc files in substack with respect to the movies")

movie_name_x = []
shift_movie_files_x = []
ptcl_source_images_xx =[]

for i in range(no_of_imgs):
    ptcl_source_images_xx.append(str(os.path.basename(ptcl_source_images[i])).split('.mrc')[0])

count = 0
for i in range (no_of_micrographs) :
    temp = np.where(np.array(ptcl_source_images_xx) == str(os.path.basename(movie_names[i])).split('.tiff')[0])
    if len(temp[0]) != 0:
        movie_name_x.append(movie_names[i])
        shift_movie_files_x.append(shift_movie_files[i])
        count+=1


movie_names = movie_name_x
shift_movie_files = shift_movie_files_x
no_of_micrographs = count


refnames = []
refnames.append(args.ref_volume_0)
refnames.append(args.ref_volume_1)

masknames = []
masknames.append(args.mask_0)
masknames.append(args.mask_1)

chunck0 = np.loadtxt(args.chunck_vol_0)
chunck1 = np.loadtxt(args.chunck_vol_1)


ima_start, ima_end = sp_applications.MPI_start_end(no_of_micrographs, n_mpi_procs, my_mpi_proc_id)


for micro in enumerate(movie_names[ima_start:ima_end]):

    print("Applying GLOBAL shifts")
    if str(os.path.basename(micro[1])).split('.')[-1] == 'mrcs':
        logfile = os.path.join(os.path.abspath(os.path.join(log_movie_path, os.pardir)),
                               micro[1].split('.')[0].split('/')[-1] + '.log')
        print(logfile)
    elif str(os.path.basename(micro[1])).split('.')[-1] == 'tiff':
        logfile = os.path.join(os.path.abspath(os.path.join(log_movie_path, os.pardir)),
                           micro[1].split('.')[0].split('/')[-1]  + '.star')
        print(logfile)
    # shift_x, shift_y = read_meta_shifts(logfile)

    shiftfile = np.loadtxt(logfile)
    shift_x = shiftfile[:, 1]
    shift_y = shiftfile[:, 2]

    zsize = EMUtil.get_image_count(micro[1])

    trackfile = os.path.join(os.path.abspath(os.path.join(log_movie_path, os.pardir)),
                             micro[1].split('.')[0].split('/')[-1] + '_tracks' + '.star')

    df = pd.read_csv(trackfile)
    particle_number = int(df['data_general'].get_values()[0].rsplit()[1])
    frames = 24
    shiftperptcl = np.zeros((particle_number, frames, 2))

    for i in range(particle_number):
        for j in range(frames):
            if i == 0:
                k = 5 + j
            if i > 0:
                k = i * 24 + ((4 * i) + j) + 5
            shiftperptcl[i, j, 0] = float(df['data_general'].get_values()[k].rsplit()[0])
            shiftperptcl[i, j, 1] = float(df['data_general'].get_values()[k].rsplit()[1])


    print("Sorted the particles from the stack ")
    project_params, particle_coordinates, ctf_params, \
                                    nx, ny, nz, source_n_ind, adnan_n = find_particles_info_from_movie( stackfilename,
                                                                                               micro[1],
                                                                                               no_of_imgs,
                                                                                               ptcl_source_images,
                                                                                               project_params_all,
                                                                                               particle_coordinates_all,
                                                                                               ctf_params_all,
                                                                                               nx_all,
                                                                                               ny_all,
                                                                                               nz_all,
                                                                                               source_n_ind_all,
                                                                                               chunck0,
                                                                                               adnan_all,
                                                                                               show_first=False,
                                                                                               use_chunk=False)

    del project_params

    print("Applying dose weighting")



    particle_imgs_dosed, old_ind_coord, bad_pp = get_all_polish_ptcl_imgs(micro[1],
                                                                  nx[0],
                                                                  ny[0],
                                                                  particle_coordinates,
                                                                  zsize,
                                                                  shift_x,
                                                                  shift_y,
                                                                  freqweights,
                                                                  Gainreffile,
                                                                          ctf_params,
                                                                  shiftperptcl)

    particle_imgs_dosed = np.array(particle_imgs_dosed).swapaxes(0, 1)

    print("Dose weighting done, summing starts")
    print(particle_imgs_dosed.shape)

    print('Length of list', micro[1], len(particle_imgs_dosed), len(adnan_n), len(bad_pp))

    mask = sp_utilities.model_circle(nx[0] / 2, nx[0], nx[0])
    ave_particle_dosed = []
    for i in range(len(particle_imgs_dosed)):
        ave_particle_dosed.append(sum(particle_imgs_dosed[i]) )

        # st = Util.infomask(ave_particle_dosed[i], mask, False)
        # Util.mul_scalar(ave_particle_dosed[i], -1.0)
        # ave_particle_dosed[i] += 2 * st[0]

        # st = Util.infomask(ave_particle_dosed[i], mask, False)
        # ave_particle_dosed[i] -= st[0]
        # ave_particle_dosed[i] /= st[1]

    del particle_imgs_dosed

    print("Writing into mrcs files", len(ave_particle_dosed))
    local_stack_path = "bdb:%s" % stack_absolute_path + micro[1].split('.')[0].split('/')[-1] + "_ptcls"

    local_mrc_path = stack_absolute_path + micro[1].split('.')[0].split('/')[-1] + "_ptcls.mrcs"

    local_bdb_stack = db_open_dict(local_stack_path)
    old_stack = db_open_dict(stackfilename, ro=True)

    print(local_mrc_path)

    for i in range(len(ave_particle_dosed)):
        index_old = adnan_n[old_ind_coord[i]]
        old_dict = old_stack.get(index_old, nodata=True).get_attr_dict()
        old_dict['data_path'] = local_mrc_path
        old_dict['data_n'] = int(i)
        old_dict['source_n'] = int(i)
        old_dict['ptcl_source_coord_id'] = i
        local_bdb_stack[i] = old_dict
        ave_particle_dosed[i].append_image(local_mrc_path)

    db_close_dict(local_stack_path)
    db_close_dict(stackfilename)

    del local_bdb_stack
    del old_stack
    del ave_particle_dosed
    del source_n_ind

mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
mpi.mpi_finalize()



# from EMAN2 import *
# import EMAN2_cppwrap
# from EMAN2 import EMNumPy
#
# new_stack_path = 'bdb:/home/adnan/PycharmProjects/DoseWeighting/particle_rel'
# local_bdb_stack = db_open_dict(new_stack_path)
#
# no_of_imgs= EMUtil.get_image_count(new_stack_path)
#
# for i in range(no_of_imgs):
#     old_dict = local_bdb_stack.get(i, nodata=True).get_attr_dict()
#     old_dict['adnan_n'] = i
#     local_bdb_stack[i] = old_dict
#
# db_close_dict(new_stack_path)





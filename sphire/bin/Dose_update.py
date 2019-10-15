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
                                   source_n_all, chunkval, show_first = False, use_chunk = True):

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
                elif use_chunk == False:
                    project_params_per_movie.append(project_params_all[i])
                    particle_coordinates_per_movie.append(particle_coordinates_all[i])
                    ctf_params_per_movie.append(ctf_params_all[i])
                    nx_per_movie.append(nx_all[i])
                    ny_per_movie.append(ny_all[i])
                    nz_per_movie.append(nz_all[i])
                    source_n_per_movie.append(source_n_all[i])

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
           nx_per_movie, ny_per_movie, nz_per_movie, source_n_per_movie

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
def create_mask (xlen, ylen):
    row, col = np.ogrid[0:xlen, 0:ylen]
    length = xlen/2
    edge_norm = length**2
    cosine_falloff = 0.5
    kernel_mask = np.sqrt(((row - length) ** 2 + (col - length) ** 2) /
                          float(edge_norm)) * cosine_falloff
    return kernel_mask

def get_weight_values(b_list, c_list, freq_k):
    a = np.multiply.outer(4*np.array(b_list), freq_k ** 2)
    np.add(a.T, np.array(c_list), out=a.T)
    np.exp(a.T, out=a.T)
    return a

#----------------------- Particle cordinate
def get_all_reduce_ptcl_imgs_modified(movie_name,
                                      nxx, nyy,
                                      part_cord,
                                      b_list_i,
                                      c_list_i,
                                      sum_k,
                                      zsize,
                                      shift_x,
                                      shift_y,
                                      gainrefname):
    """Getting dimension information"""
    current_frame = EMData()
    current_frame.read_image(movie_name, 0)
    xsize = current_frame.get_xsize()
    ysize = current_frame.get_ysize()

    cen_xx = xsize // 2
    cen_yy = ysize // 2
    cen_zz = zsize // 2

    shape = (ysize, xsize)
    mask_applied = create_mask(*shape)

    gainref = EMData()
    gainref.read_image(gainrefname,0)

    print("Mask shape", mask_applied.shape )
    print("Gain shape", gainref.get_3dview().shape)
    print("Image shape", current_frame.get_3dview().shape)
    print("xsize , yszie are", xsize, ysize)

    particle_imgs_in_movie = []

    # mask = sp_utilities.model_circle(112, nxx, nxx)

    for i in range(zsize):
        current_frame.read_image(movie_name, i)
        current_frame = Util.muln_img(current_frame, gainref)
        current_frame =  sp_fundamentals.fshift(current_frame, int(round(shift_x[i])), int(round(shift_y[i])))
        weights_mask = get_weight_values([b_list_i[i]], [c_list_i[i]], mask_applied)
        np.divide(weights_mask, sum_k, out=weights_mask)

        print("Weights shape",   weights_mask.shape)

        # weights_mask = weights_mask.swapaxes(1,2)
        """Apply Fourier Transform"""
        four_img = np.fft.fft2(current_frame.get_2dview())
        four_img2 = np.fft.fftshift(four_img)
        np.multiply(four_img2, weights_mask[0], out=four_img2)
        doseweight_framesi = np.fft.ifftshift(four_img2)
        doseweight_framesi = np.fft.ifft2( doseweight_framesi ).real

        numpy2em_python(doseweight_framesi, out=current_frame)

        print("After applying masking the dimension of the current frame changed to",
              current_frame.get_3dview().shape)

        del four_img
        del four_img2
        del doseweight_framesi

        # boxmask = copy.deepcopy(mask)
        # boxmask = sp_fundamentals.fshift(boxmask, int(round(shift_x[i])), int(round(shift_y[i])))

        crop_imgs = []
        new_coord = []
        for j in range(len(part_cord)):
            try:
                box_img = Util.window(current_frame, nxx, nyy, 1, part_cord[j][0] - cen_xx,
                                             part_cord[j][1] - cen_yy ,0)

                # box_img = Util.muln_img(box_img, boxmask)
                crop_imgs.append(box_img)
                new_coord.append(j)
                del box_img
            except RuntimeError:
                continue

        particle_imgs_in_movie.append(crop_imgs)
        del crop_imgs
    del current_frame

    return particle_imgs_in_movie , new_coord



def get_fscs_all_particles_modified( movie_name,
                                     refer,
                                     nxx,
                                     nyy,
                                     part_cord,
                                     zsize,
                                     gainrefname,
                                     shift_x,
                                     shift_y,
                                     proj_param):
    fsc_vali = []
    fsc_freqi = []
    current_frame = EMData()
    current_frame.read_image(movie_name, 0)
    xsize = current_frame.get_xsize()
    ysize = current_frame.get_ysize()

    cen_xx = xsize // 2
    cen_yy = ysize // 2
    cen_zz = zsize // 2

    gainref = EMData()
    gainref.read_image(gainrefname,0)

    # mask = sp_utilities.model_circle(112, nxx, nyy)

    print("Gain shape", gainref.get_3dview().shape)
    print("Image shape", current_frame.get_3dview().shape)
    avg_ptcl_list = []
    for i in range(zsize):
        current_frame.read_image(movie_name,i)
        current_frame = Util.muln_img(current_frame, gainref)
        fsc_frames_vali = []
        fsc_frames_freqi = []

        for j in range(len(part_cord)):
            try:
                params_substack = proj_param[j].get_params('spider')
                # ptclmask = copy.deepcopy(mask)
                # ptclmask = sp_fundamentals.fshift(ptclmask,
                #                                   params_substack['tx'] - int(round(shift_x[i])),
                #                                   params_substack['ty'] - int(round(shift_y[i])))

                ptcl = Util.window(current_frame, nxx, nyy, 1, part_cord[j][0] - cen_xx,
                                             part_cord[j][1] - cen_yy ,0)

                # ptcl = Util.muln_img(ptcl, ptclmask)
                # refer[i][j] = Util.muln_img(refer[i][j], ptclmask)

                fsc_frames_vali.append(sp_statistics.fsc(ptcl, refer[i][j])[1])
                fsc_frames_freqi.append(sp_statistics.fsc(ptcl, refer[i][j])[0])

                del ptcl
                # del ptclmask

            except RuntimeError:
                continue

        fsc_vali.append(fsc_frames_vali)
        fsc_freqi.append(fsc_frames_freqi)

    del current_frame
    return fsc_vali, fsc_freqi


def fitfunc(x, a, b,c ):
    return  a * np.exp(c + (16*b * (np.array(x)*np.array(x))))

def calculate_bfactor(fsc_array,freq_per_micrograph):

    frames_range = np.array(fsc_array).shape[0]
    freq_range = len(freq_per_micrograph[0])

    print("No of frames use for fitting", frames_range)
    print("Frequency range", freq_range)

    fsc_final_orig = fsc_array[:,shift:]
    d_list = np.average(fsc_final_orig, axis=0)

    myk = freq_per_micrograph[0][shift:]
    b_list = np.zeros(frames_range)
    c_list = np.zeros(frames_range)

    c0 = -0.5
    b0 = -50
    d0 = 1.50
    print("Shifted Frequency range", myk.shape)

    for iteration in range(N_ITER):

        for f in range(frames_range):
            fcc_per_f = fsc_final_orig[f, :]
            f_c_b = lambda u, c, b: np.multiply(d_list[u], np.exp(c + (4*b * (myk ** 2))))
            fcc_per_f[fcc_per_f <= 0] = 0.01
            popt, pconv = curve_fit(f_c_b, range(len(myk)), fcc_per_f, p0=(c0, b0),
                                    bounds=((-50, -800), (50, 0)))
            c_list[f] = popt[0]
            b_list[f] = popt[1]
           # c0 = popt[0]
          #  b0 = popt[1]

        for k_index, k in enumerate(myk):
            fcc_per_k = fsc_final_orig[:, k_index]
            fcc_per_k[fcc_per_k <= 0] = 0.01
            f_d = lambda u, d: d * np.exp(c_list[u] +  np.multiply(4*b_list[u], k ** 2))
            popt, pconv = curve_fit(f_d, np.arange(frames_range).tolist(),
                                    fcc_per_k, p0=(d0), bounds=(1, 2))
            d_list[k_index] = popt[0]

           # d0 = popt[0]
    return b_list, c_list, d_list

def givemerealbfactors():
    bfile = np.loadtxt('/home/adnan/PycharmProjects/Dose_Fresh/sphire/bin/bfactors.star')

    slope = bfile[:,1]
    intercept = bfile[:,2]

    bfacy = np.exp(intercept)

    angpix_ref = 1.244531000
    angpix_out = 1.244531000
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
        for y in range (kc2):
            for x in range(kc):
                if y < kc :
                    yy = y
                else:
                    yy == y - kc2

                r2 = (x*x) + (yy*yy)

                output[f,x,y] =  bfay[f] * np.exp(  np.divide((-0.5 * r2), bfax[f] * bfax[f] ))

    return output






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

# count = 0
# for i in range (no_of_micrographs) :
#     temp = np.where(np.array(ptcl_source_images_xx) == str(os.path.basename(movie_names[i])).split('.mrcs')[0])
#     if len(temp[0]) != 0:
#         movie_name_x.append(movie_names[i])
#         shift_movie_files_x.append(shift_movie_files[i])
#         count+=1

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

fsc_values_per_micrograph = []
fsc_avgs_per_micrograph = []
freq_per_micrograph = []
fsc_raw = []

fsc_values_per_micrograph_i = []
fsc_avgs_per_micrograph_i = []
freq_per_micrograph_i = []
fsc_raw_i = []

for ref_num in range(2):
    fsc_values = []
    fsc_avgs = []
    frequencies = []
    fsc_raw_all = []

    ref_volume = sp_utilities.get_im(refnames[ref_num])
    mask_volume = sp_utilities.get_im(masknames[ref_num])

    new_volume = Util.muln_img(ref_volume, mask_volume)

    del ref_volume
    del mask_volume

    volft, kb = sp_projection.prep_vol(new_volume, npad=2, interpolation_method=-1)

    for micro in enumerate(movie_names[ima_start:ima_end]):
        print("Applying GLOBAL shifts")
        if str(os.path.basename(micro[1])).split('.')[-1] == 'mrcs':
            logfile = os.path.join(os.path.abspath(os.path.join(log_movie_path, os.pardir)),
                                   micro[1].split('.')[0].split('/')[-1] + '.log')
        elif str(os.path.basename(micro[1])).split('.')[-1] == 'tiff':
            logfile = os.path.join(os.path.abspath(os.path.join(log_movie_path, os.pardir)),
                               micro[1].split('.')[0].split('/')[-1]  + '.star')
            print(logfile)
        # shift_x, shift_y = read_meta_shifts(logfile)

        zsize = EMUtil.get_image_count(micro[1])

        print("Number of frames is", zsize)

        shiftfile = np.loadtxt(logfile)
        shift_x = shiftfile[:, 1]
        shift_y = shiftfile[:, 2]

        if ref_num == 0 :
            chun = chunck0
        else:
            chun = chunck1

        print("Sorted the particles from the stack ")
        return_value = find_particles_info_from_movie(  stackfilename,
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
                                                                                chun,
                                                                                show_first=False,
                                                                                use_chunk=True)
        if not return_value:
            continue
        project_params, particle_coordinates, ctf_params, nx, ny, nz, source_n_ind = return_value

        ref_project_2D_ptcl_all = get_2D_project_for_all_ptcl_from_reference(volft,
                                                                             kb,
                                                                             project_params,
                                                                             zsize,
                                                                             shift_x,
                                                                             shift_y,
                                                                             show = False)


        for j in range (zsize):
            for i in range(len(ref_project_2D_ptcl_all[j])):
                ref_project_2D_ptcl_all[j][i] = sp_filter.filt_ctf(ref_project_2D_ptcl_all[j][i],
                                                                   ctf_params[i],
                                                                   sign=-1,
                                                                   binary=False)

        """
        Extracting particle image from the movie data. First getting the particle cordinates from the dictionary and then
        creating a window around to extract the same particle from each frame
        """

        fsc_val, fsc_freq = get_fscs_all_particles_modified( micro[1],
                                                             ref_project_2D_ptcl_all,
                                                             nx[0],
                                                             ny[0],
                                                             particle_coordinates,
                                                             zsize,
                                                             Gainreffile,
                                                             shift_x,
                                                             shift_y,
                                                             project_params)
        fsc_val = np.array(fsc_val)
        fsc_freq = np.array(fsc_freq)


        print("Fsc values shape",fsc_val.shape)
        print("Frequency values shape", fsc_freq.shape)

        fsc_final = np.average(fsc_val, axis=1)

        fsc_final_avg = []
        for idx in range(0, len(fsc_final) - 3):
            avv = []
            for p in range(len(fsc_final[idx])):
                avv.append((fsc_final[idx][p] +
                            fsc_final[idx + 1][p] +
                            fsc_final[idx + 2][p] +
                            fsc_final[idx + 3][p]) / 4)
            fsc_final_avg.append(avv)

        for idx in range(len(fsc_final) - 3, len(fsc_final)):
            avv = []
            for p in range(len(fsc_final[idx])):
                avv.append((fsc_final[idx][p] +
                            fsc_final[idx - 1][p] +
                            fsc_final[idx - 2][p] +
                            fsc_final[idx - 3][p]) / 4)
            fsc_final_avg.append(avv)

        fsc_values.append(fsc_final)
        fsc_avgs.append(fsc_final_avg)
        frequencies.append(fsc_freq[0][0])
        fsc_raw_all.append(np.array(fsc_val))

        del fsc_final
        del fsc_final_avg
        del fsc_freq
        del fsc_val
        del ref_project_2D_ptcl_all
        # del mask

    print("Fsc part is completed, removing data from cache")

    fsc_values_per_micrograph_i.extend(fsc_values)
    fsc_avgs_per_micrograph_i.extend(fsc_avgs)
    freq_per_micrograph_i.extend(frequencies)
    fsc_raw_i.extend(fsc_raw_all)

    # del fsc_values
    # del fsc_avgs
    # del frequencies
    # del fsc_raw_all

fsc_values_per_micrograph= sp_utilities.wrap_mpi_gatherv(fsc_values_per_micrograph_i, 0, mpi.MPI_COMM_WORLD)
fsc_avgs_per_micrograph = sp_utilities.wrap_mpi_gatherv(fsc_avgs_per_micrograph_i, 0, mpi.MPI_COMM_WORLD)
freq_per_micrograph = sp_utilities.wrap_mpi_gatherv(freq_per_micrograph_i, 0, mpi.MPI_COMM_WORLD)
fsc_raw =  sp_utilities.wrap_mpi_gatherv(fsc_raw_i, 0 , mpi.MPI_COMM_WORLD)

print('shape of array',np.array(fsc_values_per_micrograph).shape)
mpi.mpi_barrier(mpi.MPI_COMM_WORLD)


if main_mpi_proc == my_mpi_proc_id :

    fsc_values_per_micrograph = np.array(fsc_values_per_micrograph)
    fsc_avgs_per_micrograph = np.array(fsc_avgs_per_micrograph)
    freq_per_micrograph = np.array(freq_per_micrograph)


    print("Shape = ", fsc_values_per_micrograph.shape)

    fsc_sum_per_frame = np.average(np.array(fsc_values_per_micrograph) , axis = 0)

    b_list, c_list, d_list = calculate_bfactor(fsc_sum_per_frame,freq_per_micrograph)

    np.savetxt("fscv.txt", fsc_sum_per_frame)
    np.savetxt("frequencies.txt", freq_per_micrograph)
    np.savetxt("blist.txt", b_list)
    np.savetxt("clist.txt", c_list)
    np.savetxt("dlist.txt", d_list)


    b_list = [float(val) for val in b_list]
    c_list = [float(val) for val in c_list]
    d_list = [float(val) for val in d_list]

    del fsc_sum_per_frame

else:
    b_list = []
    c_list = []
    d_list = []


print("MPI barrier before broadcasting")

b_list = sp_utilities.bcast_list_to_all(b_list, my_mpi_proc_id, main_mpi_proc, mpi.MPI_COMM_WORLD)
c_list = sp_utilities.bcast_list_to_all(c_list, my_mpi_proc_id, main_mpi_proc, mpi.MPI_COMM_WORLD)
d_list = sp_utilities.bcast_list_to_all(d_list, my_mpi_proc_id, main_mpi_proc, mpi.MPI_COMM_WORLD)
print("MPI barrier after broadcasting")

initial_frame = EMData()
initial_frame.read_image(movie_names[0], 0)
xdim = initial_frame.get_xsize()
ydim = initial_frame.get_ysize()
print("Creating mask for all images, Start")
shape = (ydim, xdim)
mask_applied = create_mask(*shape)
sum_k = get_weight_values(b_list, c_list, mask_applied).sum(axis=0)
del initial_frame
del mask_applied
print("Creating mask for all images, Finish")


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

    print("Sorted the particles from the stack ")
    project_params, particle_coordinates, ctf_params, \
                                    nx, ny, nz, source_n_ind = find_particles_info_from_movie( stackfilename,
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
                                                                                               show_first=False,
                                                                                               use_chunk=False)

    del project_params

    print("Applying dose weighting")

    particle_imgs_dosed, old_ind_coord = get_all_reduce_ptcl_imgs_modified(micro[1],
                                                                           nx[0],
                                                                           ny[0],
                                                                           particle_coordinates,
                                                                           b_list,
                                                                           c_list,
                                                                           sum_k,
                                                                           zsize,
                                                                           shift_x,
                                                                           shift_y,
                                                                           Gainreffile)

    particle_imgs_dosed = np.array(particle_imgs_dosed).swapaxes(0,1)

    print("Dose weighting done, summing starts")
    print(particle_imgs_dosed.shape)

    mask = sp_utilities.model_circle(nx[0] / 2, nx[0], nx[0])
    ave_particle_dosed = []
    for i in range(len(particle_imgs_dosed)):
        ave_particle_dosed.append(sum(particle_imgs_dosed[i]))

        st = Util.infomask(ave_particle_dosed[i], mask, False)
        Util.mul_scalar(ave_particle_dosed[i], -1.0)
        ave_particle_dosed[i] += 2 * st[0]

        st = Util.infomask(ave_particle_dosed[i], mask, False)
        ave_particle_dosed[i] -= st[0]
        ave_particle_dosed[i] /= st[1]

    del particle_imgs_dosed

    print("Writing into mrcs files", len(ave_particle_dosed))
    local_stack_path = "bdb:%s" % stack_absolute_path + micro[1].split('.')[0].split('/')[-1] + "_ptcls"

    local_mrc_path = stack_absolute_path + micro[1].split('.')[0].split('/')[-1] + "_ptcls.mrcs"

    local_bdb_stack = db_open_dict(local_stack_path)
    old_stack = db_open_dict(stackfilename, ro=True)

    for i in range(len(ave_particle_dosed)):
        index_old = source_n_ind[old_ind_coord[i]]
        old_dict = old_stack.get(index_old, nodata=True).get_attr_dict()
        old_dict['data_path'] = local_mrc_path
        old_dict['ptcl_source_coord_id'] = i
        local_bdb_stack[i] = old_dic
        ave_particle_dosed[i].append_image(local_mrc_path)

    db_close_dict(local_stack_path)
    db_close_dict(stackfilename)

    del local_bdb_stack
    del old_stack
    del ave_particle_dosed
    del source_n_ind

mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
mpi.mpi_finalize()

# def check_cross_corre(movie_name, shift_in_x, shift_in_y, gainfile):
#     first_case = EMData(100,100)
#     whiteframe = 255 * np.ones((20, 20), np.uint8)
#     blackimage = np.zeros((100, 100), np.uint8)
#     blackimage[40:whiteframe.shape[0] + 40, 40:whiteframe.shape[1] + 40] = whiteframe
#
#     numpy2em_python(blackimage, out=first_case)
#
#     for i in range (5):
#         print(i)
#         first_case.append_image("check_shifting_plusround.mrcs")
#         first_case = sp_fundamentals.fshift(first_case, round(5.3), round(5.7))
#
#     del first_case
#
#     return 1,1


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

blist = np.loadtxt('/home/adnan/PycharmProjects/Dose_Fresh/sphire/bin/bfactors.star')


plt.figure()
plt.plot(blist[:,1])
plt.show()


plt.figure()
plt.plot(blist[:,2])
plt.show()


def givemerealbfactors():
    bfile = np.loadtxt('/home/adnan/PycharmProjects/Dose_Fresh/sphire/bin/bfactors.star')

    slope = bfile[:,1]
    intercept = bfile[:,2]

    bfacy = np.exp(intercept)
    0.8849999904632568
    angpix_ref = 1.244531000
    angpix_out = 1.244531000
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





import sp_filter as spfil

bfx , bfy = givemerealbfactors()
freqweights = computeweights(bfx, bfy)

current_frame = EMData(256,256)

randimg = np.arange(0, 65536).reshape(256,256)

numpy2em_python(randimg, out=current_frame)


line = freqweights[0,:,0].tolist()

checkimg = spfil.filt_table(current_frame, line)
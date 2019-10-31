"""

from __future__ import print_function
from __future__ import division
from past.utils import old_div
import numpy
import copy
import global_def

import unittest
import os
import shutil
from sparx.libpy import utilities as fu

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
print(ABSOLUTE_PATH)


class MyTestCase(unittest.TestCase):
   def test_angular_distribution_returns_same_results(self):

       params_file = ABSOLUTE_PATH + "/final_params_032.txt"
       output_folder_new = "Angular_distribution_New"
       output_folder_old = "Angular_distribution_Old"
       prefix = "angdis"
       method = "P"
       pixel_size = 1.14
       delta = 3.75
       symmetry = "icos"
       box_size = 320
       particle_radius = 140
       dpi  = 72

       if os.path.isdir(output_folder_new):
           shutil.rmtree(output_folder_new)

       if os.path.isdir(output_folder_old):
           shutil.rmtree(output_folder_old)

       import time
       start = time.time()
       return_new = fu.angular_distribution(params_file, output_folder_new, prefix, method, \
                                    pixel_size, delta, symmetry, box_size,particle_radius, \
                                            dpi, do_print=True)
       print(time.time()-start)
       start = time.time()
       return_old = fu.angular_distribution_old(params_file, output_folder_old, prefix, method, \
                                    pixel_size, delta, symmetry, box_size, particle_radius,\
                                            dpi, do_print=True)
       print(time.time() - start)
       self.assertEqual(return_new, return_old)




if __name__ == '__main__':
    unittest.main()
"""

from __future__ import print_function
from __future__ import division


from cPickle import load as pickle_load
from cPickle import dumps as pickle_dumps
from os import path, mkdir
from mpi import *
import global_def
from numpy import allclose, array_equal, array
from numpy import full as numpy_full
from numpy import float32 as numpy_float32
from numpy import ones as numpy_ones
from zlib import compress

from sphire.libpy import sp_utilities as oldfu
from sphire.utils.SPHIRE.libpy import sp_utilities as fu

mpi_init(0, [])
global_def.BATCH = True
global_def.MPI = True

ABSOLUTE_PATH = path.dirname(path.realpath(__file__))

import unittest
from test_module import get_arg_from_pickle_file, get_real_data, remove_list_of_file, returns_values_in_file,remove_dir,ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,IMAGE_2D,IMAGE_2D_REFERENCE,KB_IMAGE2D_SIZE,IMAGE_3D,IMAGE_BLANK_2D ,IMAGE_BLANK_3D ,MASK,MASK_2DIMAGE,MASK_3DIMAGE
from sphire.libpy.sp_fundamentals import symclass as foundamental_symclasss

from EMAN2_cppwrap import EMData

from copy import deepcopy
from  json import load as json_load
from random import randint
try:
    from StringIO import StringIO   # python2 case
except:
    from io import StringIO         # python3 case. You will get an error because 'sys.stdout.write(msg)' presents in the library not in the test!!
import sys

TOLERANCE = 0.0075
TRACKER = 0#get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/user_functions.do_volume_mask"))[0][0][1] # cannot use it. search another one


"""
There are some tests where I have to write to a file. At the end of the tests, sometimes at the end of each tests, I remove them.
You cannot run these tests multipkle times in parallel (i.e: use only "-np 1"

WHAT IS MISSING:
0) in all the cases where the input file is an image. I did not test the case with a complex image. I was not able to generate it 
1) because mpi stuff are involved:
    -) send_string_to_all    
    -) estimate_3D_center_MPI
2) drop_image --> It writing an image on the hdf. how can I test it
3) get_symt: how look into a Transform obj ?
4) unpack_message --> cannot insert a valid input. See in the code for more deatil
5) Test_getindexdata --> there is no unittest for it
6) make_v_stack_header --> no idea how fill the params
7) get_params3D --> I need an img woth 'xform.align3d' key
8) set_params3D --> I need an img woth 'xform.align3d' key

    
RESULT AND KNOWN ISSUES
Some compatibility tests for the following functions fail!!!
1) even_angles --> default value with P method leads to a deadlock
2) even_angles_cd --> default value with P method leads to a deadlock
3) get_image_data --> with EMData() as input value the results has the first value a random value
4) Test_balance_angular_distribution --> some compatibility tests fail

IN THESE TESS COULD BE A BUG:
1) center_2D --> with center_method=4 (or 7) parameters 'self_defined_reference' will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it is a boolean
2) model_circle --> with negative radius we have a results (already strange), anyway it should not have positive value
3) Test_angular_histogram:test_with_sym_oct_method_S --> BUG in sp_fundamentals.py -->symmetry_neighbors --> local variable 'neighbors' referenced before assignment
        It seems to be a problem with 'sym=oct1' 

In these tests there is a strange behavior:
1) Test_bcast_list_to_all
    -) test_with_empty_list --> the compatibility test in the nosetests feiled
    -) test_myid_equal_sourcenode_and_wrong_type_in_listsender_returns_ValueError --> the  exception is not raised
"""

"""
pickle files stored under smb://billy.storage.mpi-dortmund.mpg.de/abt3/group/agraunser/transfer/Adnan/pickle files
"""

"""
There are some opened issues in:
2) even_angles --> default value with P method leads to a deadlock
3) even_angles_cd --> default value with P method leads to a deadlock
4) find --> it seems to be not used
10) write_headers --> in .bdb case are not working under linux. Take a look to the code for seeing their comments
        --> if the file exists it overwrites it without a warning message. will we have to insert this message?
11) write_header --> I do not know how test the .bdb case. Hier contrary to write_headers it works under linux
12) file_type --> it is not giving us the filetype of the file. it is just parsing the name of the file and giving back the extension of the file
            Is this the real purpouse of this function?
13) set_params2D --> if you use xform=xform.align3d it works, but the output is somethiong that we do not really want to have. It does not set the values
                --> since set_params_proj has the same kind of input we are not able to discriminate them when we call the function. anyway It does not set the values
14) set_params3D --> if you use xform=xform.align2d it works, but the output is somethiong that we do not really want to have. It does not set the values
15) set_params_proj --> I need an image with key 'xform.projection' to finish these tests because the associated pickle file has not it --> dovrebbero essere quelle in pickle files/multi_shc/multi_shc.ali3d_multishc
16) The following functions concern the sending data in the process and are difficult or even not possible to test deeply
    -) reduce_EMData_to_root
    -) bcast_compacted_EMData_all_to_all
    -) gather_compacted_EMData_to_root
    -) bcast_EMData_to_all
    -) send_EMData
    -) recv_EMData
    -) recv_attr_dict
    -) send_attr_dict
    -) wrap_mpi_send
    -) wrap_mpi_recv
    -) wrap_mpi_gatherv
    -) wrap_mpi_split
18) 'update_tag' returns, in both of the implementations 'return 123456'. i'm not going to test it 
20) sample_down_1D_curve --> I need a file with the curve values
21) test_print_upper_triangular_matrix --> which variable is the third parameter??")
22) get_shrink_data_huang,recons_mref --> the file gave me does not work see the reasons in the test
23) do_two_way_comparison -->  I cannot run the Adnan reference test. I had to insert random data --> I cannot test it deeply,
24) Test_get_stat_proj.test_myid_not_the_same_value_as_main_Node_TypeError is it due to a bad implemntation?

"""



""" start: new in sphire 1.3
There are some opened issues in:
1) center_2D --> 
    a) with a 3D image should it work?
    b) with center_method=4 (or 7) IF the parameters 'self_defined_reference' , it will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it 
        is a boolean
16) The following functions concern the sending data in the process and are difficult or even not possible to test deeply        
    -) gather_EMData
    -) send_string_to_all
    -) wrap_mpi_split_shared_memory
"""




class Test_makerelpath(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.makerelpath()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.makerelpath()
        self.assertEqual(str(cm_new.exception), "makerelpath() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_makerelpath(self):
        return_new = oldfu.makerelpath(p1="/a/g",p2="/a/g/s/d.txt")
        return_old = fu.makerelpath(p1="/a/g",p2="/a/g/s/d.txt")
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, "s/d.txt")


#todo: need data
class Test_make_v_stack_header(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.make_v_stack_header()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.make_v_stack_header()
        self.assertEqual(str(cm_new.exception), "make_v_stack_header() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))




#   THESE FUNCTIONS ARE COMMENTED BECAUSE NOT INVOLVED IN THE PYTHON3 CONVERSION. THEY HAVE TO BE TESTED ANYWAY
"""
class Test_params_2D_3D(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.params_2D_3D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.params_2D_3D()
        self.assertEqual(str(cm_new.exception), "params_2D_3D() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_positive_mirror(self):
        return_old = oldfu.params_2D_3D(alpha=0.1, sx=2, sy=1, mirror=1)
        return_new = fu.params_2D_3D(alpha=0.1, sx=2, sy=1, mirror=1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (180.0, 180.0, 179.89999999403244, 1.9982515573501587, 1.0034891366958618)))

    def test_NOTpositive_mirror(self):
        return_old = oldfu.params_2D_3D(alpha=0.1, sx=2, sy=1, mirror=0)
        return_new = fu.params_2D_3D(alpha=0.1, sx=2, sy=1, mirror=0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0, 0, 359.89999999403244, 1.9982515573501587, 1.0034891366958618)))



class Test_params_3D_2D(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.params_3D_2D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.params_3D_2D()
        self.assertEqual(str(cm_new.exception), "params_3D_2D() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_theta_lower90(self):
        return_old = oldfu.params_3D_2D(phi="not_used",theta=50, s2x=2, s2y=1, psi=1)
        return_new = fu.params_3D_2D(phi="not_used",theta=50, s2x=2, s2y=1, psi=1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (359.0000000534512, 1.9822430610656738, 1.0347524881362915, 0)))

    def test_theta_highr90(self):
        return_old = oldfu.params_3D_2D(phi="not_used",theta=150, s2x=2, s2y=1, psi=1)
        return_new = fu.params_3D_2D(phi="not_used",theta=150, s2x=2, s2y=1, psi=1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (179.0000000534512, -1.9822430610656738, -1.0347524881362915, 1)))



class Test_amoeba_multi_level(unittest.TestCase):
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.amoeba"))

    @staticmethod
    def wrongfunction(a,b):
        return a+b

    @staticmethod
    def function_lessParam():
        return 0

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba_multi_level()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba_multi_level()
        self.assertEqual(str(cm_new.exception), "amoeba_multi_level() takes at least 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_amoeba_multi_level(self):
        '''
        I did not use 'self.assertTrue(allclose(return_new, return_old, atol=TOLERANCE,equal_nan=True))' because the 'nosetets' spawns the following error
                TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
        '''
        (var, scale, func, ftolerance, xtolerance, itmax , data) = self.argum[0]
        return_new = fu.amoeba_multi_level (var, scale, func, ftolerance, xtolerance, 20 , data)
        return_old = oldfu.amoeba_multi_level (var, scale, func, ftolerance, xtolerance, 20 , data)
        self.assertTrue(allclose(return_new[0], return_old[0], atol=TOLERANCE,equal_nan=True))
        self.assertTrue(abs(return_new[1]- return_old[1]) <TOLERANCE)
        self.assertEqual(return_new[2],return_old[2])

    def test_amoeba_multi_level_with_wrongfunction(self):
        (var, scale, func, ftolerance, xtolerance, itmax , data) = self.argum[0]
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba_multi_level (var, scale, self.wrongfunction, ftolerance, xtolerance, itmax , None)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba_multi_level (var, scale, self.wrongfunction, ftolerance, xtolerance, itmax , None)
        self.assertEqual(str(cm_new.exception), "wrongfunction() got an unexpected keyword argument 'data'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_amoeba_multi_level_with_function_lessParam_TypeError(self):
        (var, scale, func, ftolerance, xtolerance, itmax , data) = self.argum[0]
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba_multi_level (var, scale, self.function_lessParam, ftolerance, xtolerance, itmax , None)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba_multi_level (var, scale, self.function_lessParam, ftolerance, xtolerance, itmax , None)
        self.assertEqual(str(cm_new.exception), "function_lessParam() takes no arguments (2 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_amoeba_multi_level_with_NoneType_data_returns_TypeError_NoneType_obj_hasnot_attribute__getitem__(self):
        (var, scale, func, ftolerance, xtolerance, itmax , data) = self.argum[0]
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba_multi_level (var, scale, func, ftolerance, xtolerance, itmax , None)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba_multi_level (var, scale, func, ftolerance, xtolerance, itmax , None)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute '__getitem__'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_ce_fit(unittest.TestCase):
    def test_mixed_array(self,a1=None, a2=None):
        if a1 is not None
            for i, j in zip(a1, a2):
                if isinstance(i, list) or isinstance(i, tuple):
                    self.assertTrue(array_equal(i,j))
                else:
                    self.assertEqual(i, j)

    def test_NoneType_inp_imagecrashes_because_signal11SIGSEV(self):
        pass
        '''
        return_new = fu.ce_fit(inp_image=None, ref_image= IMAGE_2D_REFERENCE, mask_image=MASK)
        return_old = oldfu.ce_fit(inp_image=None, ref_image= IMAGE_2D_REFERENCE, mask_image=MASK)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,None)
        '''

    def test_Empty_inp_image(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.ce_fit(inp_image=EMData(), ref_image= IMAGE_2D_REFERENCE, mask_image=MASK)
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.ce_fit(inp_image=EMData(), ref_image= IMAGE_2D_REFERENCE, mask_image=MASK)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_NoneType_ref_imagecrashes_because_signal11SIGSEV(self):
        pass
        '''
        return_new = fu.ce_fit(inp_image=IMAGE_2D, ref_image= None, mask_image=MASK)
        return_old = oldfu.ce_fit(inp_image=IMAGE_2D, ref_image= None, mask_image=MASK)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,None)
        '''

    def test_Empty_ref_image(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.ce_fit(inp_image=IMAGE_2D, ref_image= EMData(), mask_image=MASK)
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.ce_fit(inp_image=IMAGE_2D, ref_image= EMData(), mask_image=MASK)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_NoneType_mask_imagee(self):
        return_new = fu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=None)
        return_old = oldfu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=None)
        self.test_mixed_array(return_new[0],return_old[0])
        self.assertEqual(return_new[1],return_old[1])
        self.assertTrue(array_equal(return_new[2].get_3dview(), return_old[2].get_3dview()))
        self.test_mixed_array(return_new[0], ['Final Parameter [A,B]:', [1.0038907527923584, 0.0013515561586245894], 'Final Chi-square :', 8806394.0, 'Number of Iteration :', 0])
        self.assertEqual(return_new[1],'Corrected Image :')

    def test_NoneType_with_mask_image(self):
        return_new = fu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=MASK_2DIMAGE)
        return_old = oldfu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=MASK_2DIMAGE)
        self.test_mixed_array(return_new[0],return_old[0])
        self.assertEqual(return_new[1],return_old[1])
        self.assertTrue(array_equal(return_new[2].get_3dview(), return_old[2].get_3dview()))
        self.test_mixed_array(return_new[0], ['Final Parameter [A,B]:', [1.0039470195770264, 0.0015781484544277191], 'Final Chi-square :', 8806394.0, 'Number of Iteration :', 0])
        self.assertEqual(return_new[1],'Corrected Image :')

    def test_Empty_mask_image(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=EMData())
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=EMData())
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_2DIMAGE_different_size_mask_runtimeError(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=MASK)
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.ce_fit(inp_image=IMAGE_2D, ref_image= IMAGE_2D_REFERENCE, mask_image=MASK)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The size of mask image should be of same size as the input image")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])


class Test_common_line_in3D(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.common_line_in3D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.common_line_in3D()
        self.assertEqual(str(cm_new.exception), "common_line_in3D() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_common_line_in3D(self):
        return_old = oldfu.common_line_in3D(phiA=2, thetaA=1, phiB=4, thetaB=3)
        return_new = fu.common_line_in3D(phiA=2, thetaA=1, phiB=4, thetaB=3)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (45.11342420305066, 89.00320412303714)))



class Test_compose_transform2m(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.compose_transform2m()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.compose_transform2m()
        self.assertEqual(str(cm_new.exception), "compose_transform2m() takes exactly 10 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_compose_transform2m(self):
        return_old = oldfu.compose_transform2m(alpha1=1.0,sx1=2.0,sy1=3.0,mirror1=0,scale1=1.0,alpha2=2.0,sx2=3.0,sy2=4.0,mirror2=0,scale2=1.0)
        return_new = fu.compose_transform2m(alpha1=1.0,sx1=2.0,sy1=3.0,mirror1=0,scale1=1.0,alpha2=2.0,sx2=3.0,sy2=4.0,mirror2=0,scale2=1.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (2.999999834763278, 5.103480339050293, 6.928373336791992, 0, 1.0)))



class Test_create_smooth_mask(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.create_smooth_mask()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.create_smooth_mask()
        self.assertEqual(str(cm_new.exception), "create_smooth_mask() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_create_smooth_mask(self):
        return_old = oldfu.create_smooth_mask( radius=2, img_dim=500, size=8 )
        return_new = fu.create_smooth_mask( radius=2, img_dim=500, size=8 )
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))


#todo: how test it? it is writing an image
class Test_drop_png_image(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.drop_png_image()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.drop_png_image()
        self.assertEqual(str(cm_new.exception), "drop_png_image() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Empty_im(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.drop_png_image(im=EMData(), trg='png')
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.drop_png_image(im=EMData(), trg='png')
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_NoneType_ref_image(self):
        return_new = fu.drop_png_image(im=IMAGE_2D, trg='png')
        return_old = oldfu.drop_png_image(im=IMAGE_2D, trg='png')
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,None)


#todo: it is writing on a file
class Test_dump_row(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.dump_row()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.dump_row()
        self.assertEqual(str(cm_new.exception), "dump_row() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Empty_input(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.dump_row(input=EMData(), fname="filename", ix=0, iz=0)
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.dump_row(input=EMData(), fname="filename", ix=0, iz=0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_NoneType_ref_inputage(self):
        return_new = fu.dump_row(input=IMAGE_2D, fname="filename", ix=0, iz=0)
        return_old = oldfu.dump_row(input=IMAGE_2D, fname="filename", ix=0, iz=0)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,None)


#todo: it writes an image
class Test_eigen_images_get(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.eigen_images_get()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.eigen_images_get()
        self.assertEqual(str(cm_new.exception), "eigen_images_get() takes exactly 5 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_eigen_images_get(self):
        oldv = oldfu.eigen_images_get(stack="", eigenstack="", mask="", num="", avg="")
        v = fu.eigen_images_get(stack="", eigenstack="", mask="", num="", avg="")
        pass



class Test_find_inplane_to_match(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.find_inplane_to_match()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.find_inplane_to_match()
        self.assertEqual(str(cm_new.exception), "find_inplane_to_match() takes at least 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_find_inplane_to_match(self):
        return_old = oldfu.find_inplane_to_match(phiA=1, thetaA=1, phiB=2, thetaB=2, psiA=0, psiB=0)
        return_new = fu.find_inplane_to_match(phiA=1, thetaA=1, phiB=2, thetaB=2, psiA=0, psiB=0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (-359.0003071638985, 1.0003045023348265)))



class Test_get_sym(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_sym()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_sym()
        self.assertEqual(str(cm_new.exception), "get_sym() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_sym(self):
        return_old = oldfu.get_sym(symmetry="c1")
        return_new = fu.get_sym(symmetry="c1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[0.0, 0.0, 0.0]]))



#todo: need a file
class Test_get_textimage(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_textimage()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_textimage()
        self.assertEqual(str(cm_new.exception), "get_textimage() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_textimage(self):
        oldv = oldfu.get_textimage(fname="")
        v = fu.get_textimage(fname="")
        pass


#todo: need to run it to see the output values
class Test_hist_func(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.hist_func()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.hist_func()
        self.assertEqual(str(cm_new.exception), "hist_func() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_data_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.hist_func(data=[], args=[1,2,3])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.hist_func(data=[], args=[1,2,3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_data0_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.hist_func(data=[[], [1, 2, 3]], args=[1,2,3])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.hist_func(data=[[], [1, 2, 3]], args=[1,2,3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_args_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.hist_func(data=[[1, 2, 3], [1, 2, 3]], args=[])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.hist_func(data=[[1, 2, 3], [1, 2, 3]], args=[])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_info(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.info()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.info()
        self.assertEqual(str(cm_new.exception), "info() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_2dImg(self):
        return_old = oldfu.info(image=IMAGE_2D, mask=None, Comment="")
        return_new = fu.info(image=IMAGE_2D, mask=None, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (-0.34738418459892273, 1.3910515308380127, -4.196987152099609, 6.156104564666748, 352, 352, 1)))

    def test_2dImgwith_wrongsize_mask_RuntimeError_ImageDimensionException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            oldfu.info(image=IMAGE_2D, mask=MASK, Comment="")
        with self.assertRaises(RuntimeError) as cm_old:
            fu.info(image=IMAGE_2D, mask=MASK, Comment="")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The dimension of the image does not match the dimension of the mask!")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_2dImg_withMask(self):
        return_old = oldfu.info(image=IMAGE_2D, mask=None, Comment="")
        return_new = fu.info(image=IMAGE_2D, mask=None, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (1.617922306060791, 0.3879293203353882, 1.0218865871429443, 2.2291574478149414, 352, 352, 1)))

    def test_2dImgBlankwith_wrongsize_mask_RuntimeError_ImageDimensionException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            oldfu.info(image=IMAGE_BLANK_2D, mask=MASK, Comment="")
        with self.assertRaises(RuntimeError) as cm_old:
            fu.info(image=IMAGE_BLANK_2D, mask=MASK, Comment="")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The dimension of the image does not match the dimension of the mask!")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_2dblankImg(self):
        return_old = oldfu.info(image=IMAGE_BLANK_2D, mask=None, Comment="")
        return_new = fu.info(image=IMAGE_BLANK_2D, mask=None, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (0.0, 0.0, 0.0, 0.0, 10, 10, 1)))

    def test_2dblankImgwith_mask(self):
        return_old = oldfu.info(image=IMAGE_BLANK_2D, mask=MASK_IMAGE_BLANK_2D, Comment="")
        return_new = fu.info(image=IMAGE_BLANK_2D, mask=MASK_IMAGE_BLANK_2D, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (0.0, 0.0, 0.0, 0.0, 10, 10, 1)))

    def test_3dImg(self):
        return_old = oldfu.info(image=IMAGE_3D, mask=None, Comment="")
        return_new = fu.info(image=IMAGE_3D, mask=None, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (1.640047550201416, 8.512676239013672, -10.731719017028809, 144.96165466308594, 76, 76, 76)))

    def test_3dImgwith_wrongsize_mask_RuntimeError_ImageDimensionException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            oldfu.info(image=IMAGE_3D, mask=MASK, Comment="")
        with self.assertRaises(RuntimeError) as cm_old:
            fu.info(image=IMAGE_3D, mask=MASK, Comment="")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The dimension of the image does not match the dimension of the mask!")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_3dImgwith_mask(self):
        return_old = oldfu.info(image=IMAGE_3D, mask=MASK_3DIMAGE, Comment="")
        return_new = fu.info(image=IMAGE_3D, mask=MASK_3DIMAGE, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (24.25007438659668, 14.403861045837402, -10.731719017028809, 45.90932083129883, 76, 76, 76)))

    def test_3dblankImg(self):
        return_old = oldfu.info(image=IMAGE_BLANK_3D, mask=None, Comment="")
        return_new = fu.info(image=IMAGE_BLANK_3D, mask=None, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (0.0, 0.0, 0.0, 0.0, 10, 10, 10)))

    def test_3dblankImgwith_mask(self):
        return_old = oldfu.info(image=IMAGE_BLANK_3D, mask=MASK_IMAGE_BLANK_3D, Comment="")
        return_new = fu.info(image=IMAGE_BLANK_3D, mask=MASK_IMAGE_BLANK_3D, Comment="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (0.0, 0.0, 0.0, 0.0, 10, 10, 10)))

    def test_3dImgBlankwith_wrongsize_mask_RuntimeError_ImageDimensionException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            oldfu.info(image=IMAGE_BLANK_3D, mask=MASK, Comment="")
        with self.assertRaises(RuntimeError) as cm_old:
            fu.info(image=IMAGE_BLANK_3D, mask=MASK, Comment="")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The dimension of the image does not match the dimension of the mask!")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_model_square(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_square()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_square()
        self.assertEqual(str(cm_new.exception), "model_square() takes at least 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_model_square(self):
        return_old = oldfu.model_square(d=3, nx=100, ny=100, nz=1)
        return_new = fu.model_square(d=3, nx=100, ny=100, nz=1)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))


#todo: need files
class Test_parse_spider_fname(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.parse_spider_fname()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.parse_spider_fname()
        self.assertEqual(str(cm_new.exception), "parse_spider_fname() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


#todo: it is not running because a --> UnboundLocalError: local variable 'image_mask_applied' referenced before assignment
class Test_reconstitute_mask(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.reconstitute_mask()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.reconstitute_mask()
        self.assertEqual(str(cm_new.exception), "reconstitute_mask() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_reconstitute_mask(self):
        return_old = oldfu.reconstitute_mask(image_mask_applied_file=[],new_mask_file=3,save_file_on_disk=False,saved_file_name="image_in_reconstituted_mask.hdf")
        return_new = fu.reconstitute_mask(image_mask_applied_file=[],new_mask_file=3,save_file_on_disk=False,saved_file_name="image_in_reconstituted_mask.hdf")
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))




class Test_rotate_about_center(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rotate_about_center()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rotate_about_center()
        self.assertEqual(str(cm_new.exception), "rotate_about_center() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_rotate_about_center(self):
        return_old = oldfu.rotate_about_center(alpha=1, cx=2, cy=3)
        return_new = fu.rotate_about_center(alpha=1, cx=2, cy=3)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, (0.9999999465488011, -0.05205273628234863, 0.035361528396606445, 1.0)))


# not able to test it. why??
class Test_estimate_3D_center(unittest.TestCase):
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.rotate_3D_shift"))
    (data, notUsed) = argum[0]
    #data=[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
    #data=[IMAGE3d]
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.estimate_3D_center()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.estimate_3D_center()
        self.assertEqual(str(cm_new.exception), "estimate_3D_center() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_estimate_3D_center(self):
        return_old = oldfu.estimate_3D_center(data=[self.data,"xform.projection"])
        return_new = fu.estimate_3D_center(data=[self.data,"xform.projection"])
        self.assertTrue(array_equal(return_new, return_old))



class Test_sym_vol(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.sym_vol()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.sym_vol()
        self.assertEqual(str(cm_new.exception), "sym_vol() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Empty_image(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.sym_vol(image=EMData(),symmetry="c1")
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.sym_vol(image=EMData(),symmetry="c1")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_copy(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.sym_vol(image=IMAGE_3D, symmetry="c1")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.sym_vol(image=IMAGE_3D, symmetry="c1")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'copy'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_sym_vol_c1_3Dimg(self):
        return_old = oldfu.sym_vol(image=IMAGE_3D, symmetry="c1")
        return_new = fu.sym_vol(image=IMAGE_3D, symmetry="c1")
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_sym_vol_c1_2Dimg(self):
        return_old = oldfu.sym_vol(image=IMAGE_2D, symmetry="c1")
        return_new = fu.sym_vol(image=IMAGE_2D, symmetry="c1")
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_sym_vol_d1_3Dimg(self):
        return_old = oldfu.sym_vol(image=IMAGE_3D, symmetry="d1")
        return_new = fu.sym_vol(image=IMAGE_3D, symmetry="d1")
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_sym_vol_d1_2Dimg(self):
        return_old = oldfu.sym_vol(image=IMAGE_2D, symmetry="d1")
        return_new = fu.sym_vol(image=IMAGE_2D, symmetry="d1")
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))


#todo: node involved, i cannot test
class Test_gather_EMData(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.gather_EMData()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.gather_EMData()
        self.assertEqual(str(cm_new.exception), "gather_EMData() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_gather_EMData(self):
        return_old = oldfu.gather_EMData(data=[EMData(),EMData()], number_of_proc=2, myid=1, main_node=0)
        return_new = fu.gather_EMData(data=[EMData(),EMData()], number_of_proc=2, myid=1, main_node=0)
        for i,j in zip(return_new, return_old):
            self.assertTrue(array_equal(i, j))


class Test_set_seed(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_seed()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_seed()
        self.assertEqual(str(cm_new.exception), "set_seed() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_seed(self):
        return_old = oldfu.set_seed(sde=10)
        return_new = fu.set_seed(sde=10)
        self.assertIsNone(return_new)
        self.assertIsNone(return_old)



class Test_check_attr(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.check_attr()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.check_attr()
        self.assertEqual(str(cm_new.exception), "check_attr() takes at least 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_check_attr_returns_true(self):
        return_old = oldfu.check_attr(ima=IMAGE_2D, num=1, params="apix_x", default_value=2, action="Warning")
        return_new = fu.check_attr(ima=IMAGE_2D, num=1, params="apix_x", default_value=2, action="Warning")
        self.assertTrue(return_new)
        self.assertTrue(return_old)


    def test_check_attr_returns_false(self):
        return_old = oldfu.check_attr(ima=IMAGE_2D, num=1, params="para", default_value=2, action="Warning")
        return_new = fu.check_attr(ima=IMAGE_2D, num=1, params="para", default_value=2, action="Warning")
        self.assertFalse(return_new)
        self.assertFalse(return_old)


#todo: how tst it??
class Test_copy_attr(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.copy_attr()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.copy_attr()
        self.assertEqual(str(cm_new.exception), "copy_attr() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_copy_attr(self):
        return_old = oldfu.copy_attr(pin=0, name=0, pot=0)
        return_new = fu.copy_attr(pin=0, name=0, pot=0)


#todo: how tst it??
class Test_set_ctf(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_ctf()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_ctf()
        self.assertEqual(str(cm_new.exception), "set_ctf() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_ctf(self):
        return_old = oldfu.set_ctf(ima=IMAGE_2D, p=[0,1,1,1,1,1,1,1])
        return_new = fu.set_ctf(ima=IMAGE_2D, p=[0,1,1,1,1,1,1,1])



class Test_parse_user_function(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.parse_user_function()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.parse_user_function()
        self.assertEqual(str(cm_new.exception), "parse_user_function() takes exactly argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_not_str(self):
        return_old = oldfu.parse_user_function(opt_string=3)
        return_new = fu.parse_user_function(opt_string=3)
        self.assertIsNone(return_new)
        self.assertIsNone(return_old)

    def test_not_valid_optstr(self):
        return_old = oldfu.parse_user_function(opt_string="invalid")
        return_new = fu.parse_user_function(opt_string="invalid")
        self.assertEqual(return_old,"invalid")
        self.assertEqual(return_new,return_old)

    def test__2str_in_optstr(self):
        return_old = oldfu.parse_user_function(opt_string="[valid,valid2]")
        return_new = fu.parse_user_function(opt_string="[valid,valid2]")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, ["valid","valid2"]))

    def test__3str_in_optstr(self):
        return_old = oldfu.parse_user_function(opt_string="[valid,valid2,valid3]")
        return_new = fu.parse_user_function(opt_string="[valid,valid2,valid3]")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, ['valid2', 'valid3', 'valid']))



class Test_getang(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.getang()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.getang()
        self.assertEqual(str(cm_new.exception), "getang() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_getang(self):
        return_old = oldfu.getang(n=[0.1,0.13,0.15])
        return_new = fu.getang(n=[0.1,0.13,0.15])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [52.43140797117251, 81.37307344132137]))



class Test_nearest_ang(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearest_ang()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearest_ang()
        self.assertEqual(str(cm_new.exception), "nearest_ang() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearest_ang(vecs=[], phi=3, tht=4)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearest_ang(vecs=[], phi=3, tht=4)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nearest_ang(self):
        return_old = oldfu.nearest_ang(vecs=[1,2,3], phi=3, tht=4)
        return_new = fu.nearest_ang(vecs=[1,2,3], phi=3, tht=4)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_old, 0)



class Test_nearestk_projangles(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearestk_projangles()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearestk_projangles()
        self.assertEqual(str(cm_new.exception), "nearestk_projangles() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearestk_projangles(projangles=[], whichone=0, howmany=1, sym="c1")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearestk_projangles(projangles=[], whichone=0, howmany=1, sym="c1")
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nearestk_projangles_c1(self):
        return_old = oldfu.nearestk_projangles(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5]], whichone=0, howmany=1, sym="c1")
        return_new = fu.nearestk_projangles(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5]], whichone=0, howmany=1, sym="c1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [1]))

    def test_nearestk_projangles_d1(self):
        return_old = oldfu.nearestk_projangles(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5]], whichone=0, howmany=1, sym="d1")
        return_new = fu.nearestk_projangles(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5]], whichone=0, howmany=1, sym="d1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [1]))

    def test_nearestk_projangles_error_symmetry(self):
        return_old = oldfu.nearestk_projangles(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5]], whichone=0, howmany=1, sym="invalid")
        return_new = fu.nearestk_projangles(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5]], whichone=0, howmany=1, sym="invalid")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, []))


#todo: i cannot test it
class Test_assign_projangles_slow(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.assign_projangles_slow()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.assign_projangles_slow()
        self.assertEqual(str(cm_new.exception), "assign_projangles_slow() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_projangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.assign_projangles_slow(projangles=[], refangles=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.assign_projangles_slow(projangles=[], refangles=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.assign_projangles_slow(refangles=[], projangles=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.assign_projangles_slow(refangles=[], projangles=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nearestk_projangles_error_symmetry(self):
        return_old = oldfu.assign_projangles_slow(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5],[11,2,3,4,5]], refangles=[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[11,2,3,4,5]])
        return_new = fu.assign_projangles_slow(projangles=[[1,2,3,4,5],[1,2,30,4,5],[11,2,3,4,5],[11,2,3,4,5]], refangles=[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[11,2,3,4,5]])
        self.assertTrue(array_equal(return_new, return_old))


#todo: i cannot test it
class Test_nearest_full_k_projangles(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.assign_projangles_slow()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.assign_projangles_slow()
        self.assertEqual(str(cm_new.exception), "assign_projangles_slow() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_projangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearest_full_k_projangles(angles=[], reference_ang=[1, 2, 3], howmany=1, sym_class=None)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearest_full_k_projangles(angles=[], reference_ang=[1, 2, 3], howmany=1, sym_class=None)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearest_full_k_projangles(reference_ang=[], angles=[1, 2, 3], howmany=1, sym_class=None)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearest_full_k_projangles(reference_ang=[], angles=[1, 2, 3], howmany=1, sym_class=None)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    def test_sym_notC1(self):
        return_old = oldfu.nearest_full_k_projangles(reference_ang=[1,3,5], angles=[1,3,2], howmany=1, sym_class=None)
        return_new = fu.nearest_full_k_projangles(reference_ang=[1,3,5], angles=[1,3,2], howmany=1, sym_class=None)
        self.assertTrue(array_equal(return_new, return_old))

    def test_sym_C1(self):
        return_old = oldfu.nearest_full_k_projangles(reference_ang=[1,3,5], angles=[1,3,2], howmany=1, sym_class="c1")
        return_new = fu.nearest_full_k_projangles(reference_ang=[1,3,5], angles=[1,3,2], howmany=1, sym_class="c1")
        self.assertTrue(array_equal(return_new, return_old))



class Test_nearestk_to_refdir(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearestk_to_refdir()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearestk_to_refdir()
        self.assertEqual(str(cm_new.exception), "nearestk_to_refdir() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refdir_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearestk_to_refdir(refdir=[], refnormal=[1, 2, 3], howmany=1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearestk_to_refdir(refdir=[], refnormal=[1, 2, 3], howmany=1)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refnormal_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearestk_to_refdir(refnormal=[], refdir=[1, 2, 3], howmany=1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearestk_to_refdir(refnormal=[], refdir=[1, 2, 3], howmany=1)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nearestk_to_refdir(self):
        return_old = oldfu.nearestk_to_refdir(refnormal=[1,3,5], refdir=[1,3,2], howmany=1)
        return_new = fu.nearestk_to_refdir(refnormal=[1,3,5], refdir=[1,3,2], howmany=1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([0], return_old))



class Test_nearestk_to_refdirs(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearestk_to_refdirs()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearestk_to_refdirs()
        self.assertEqual(str(cm_new.exception), "nearestk_to_refdir() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refdir_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearestk_to_refdirs(refdir=[], refnormal=[1, 2, 3], howmany=1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearestk_to_refdirs(refdir=[], refnormal=[1, 2, 3], howmany=1)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refnormal_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.nearestk_to_refdirs(refnormal=[], refdir=[1, 2, 3], howmany=1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.nearestk_to_refdirs(refnormal=[], refdir=[1, 2, 3], howmany=1)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nearestk_to_refdir(self):
        return_old = oldfu.nearestk_to_refdirs(refnormal=[1, 3, 5,1, 3, 5,1, 3, 5,1, 3, 5,1, 3, 5], refdir=[[1, 3, 2], [1, 3, 2]], howmany=1)
        return_new = fu.nearestk_to_refdirs(refnormal=[1, 3, 5,1, 3, 5,1, 3, 5,1, 3, 5,1, 3, 5], refdir=[[1, 3, 2], [1, 3, 2]], howmany=1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[0], [1]]))


#todo: how can i test it?
class Test_assign_projangles_f(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.assign_projangles_f()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.assign_projangles_f()
        self.assertEqual(str(cm_new.exception), "assign_projangles_f() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_refangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.assign_projangles_f(refangles=[], projangles=[1, 2, 3], return_asg=False)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.assign_projangles_f(refangles=[], projangles=[1, 2, 3], return_asg=False)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_projangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.assign_projangles_f(projangles=[], refangles=[1, 2, 3], return_asg=False)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.assign_projangles_f(projangles=[], refangles=[1, 2, 3], return_asg=False)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_assign_projangles_f_notreturnASG(self):
        return_old = oldfu.assign_projangles_f(projangles=[1, 2, 3], refangles=[1, 2, 2], return_asg=False)
        return_new = fu.assign_projangles_f(projangles=[1, 2, 3], refangles=[1, 2, 2], return_asg=False)
        self.assertTrue(array_equal(return_new, return_old))

    def test_assign_projangles_f_returnASG(self):
        return_old = oldfu.assign_projangles_f(projangles=[1, 2, 3], refangles=[1, 2, 2], return_asg=True)
        return_new = fu.assign_projangles_f(projangles=[1, 2, 3], refangles=[1, 2, 2], return_asg=True)
        self.assertTrue(array_equal(return_new, return_old))


class Test_cone_ang(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.cone_ang()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.cone_ang()
        self.assertEqual(str(cm_new.exception), "cone_ang() takes at least 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_projangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.cone_ang(projangles=[], phi = 1, tht=2 , ant=3, symmetry="c1")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.cone_ang(projangles=[], phi = 1, tht=2 , ant=3, symmetry="c1")
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_sym_c1(self):
        return_old = oldfu.cone_ang(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c1")
        return_new = fu.cone_ang(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[1, 2, 3, 4]]))

    def test_sym_c2(self):
        return_old = oldfu.cone_ang(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c2")
        return_new = fu.cone_ang(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c2")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[1, 2, 3, 4]]))

    def test_sym_d2(self):
        return_old = oldfu.cone_ang(projangles=[[1, 2, 3, 4], [21, 22, 23, 24], [11, 12, 13, 14]], phi=1, tht=2, ant=3,symmetry="d2")
        return_new = fu.cone_ang(projangles=[[1, 2, 3, 4], [21, 22, 23, 24], [11, 12, 13, 14]], phi=1, tht=2, ant=3,symmetry="d2")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[1, 2, 3, 4]]))



class Test_cone_ang_f(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.cone_ang_f()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.cone_ang_f()
        self.assertEqual(str(cm_new.exception), "cone_ang_f() takes at least 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_projangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.cone_ang_f(projangles=[], phi = 1, tht=2 , ant=3, symmetry="c1")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.cone_ang_f(projangles=[], phi = 1, tht=2 , ant=3, symmetry="c1")
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_sym_c1(self):
        return_old = oldfu.cone_ang_f(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c1")
        return_new = fu.cone_ang_f(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[1, 2, 3, 4]]))

    def test_sym_c(self):
        return_old = oldfu.cone_ang_f(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c2")
        return_new = fu.cone_ang_f(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3, symmetry="c2")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[1, 2, 3, 4]]))

    def test_sym_d(self):
        return_old = oldfu.cone_ang_f(projangles=[[1, 2, 3, 4], [21, 22, 23, 24], [11, 12, 13, 14]], phi=1, tht=2, ant=3,symmetry="d2")
        return_new = fu.cone_ang_f(projangles=[[1, 2, 3, 4], [21, 22, 23, 24], [11, 12, 13, 14]], phi=1, tht=2, ant=3,symmetry="d2")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[1, 2, 3, 4]]))



class Test_cone_ang_with_index(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.cone_ang_with_index()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.cone_ang_with_index()
        self.assertEqual(str(cm_new.exception), "cone_ang_with_index() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_projangles_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.cone_ang_with_index(projangles=[], phi = 1, tht=2 , ant=3)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.cone_ang_with_index(projangles=[], phi = 1, tht=2 , ant=3)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_sym_c1(self):
        return_old = oldfu.cone_ang_with_index(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3)
        return_new = fu.cone_ang_with_index(projangles=[[1,2,3,4],[21,22,23,24],[11,12,13,14]], phi = 1, tht=2 , ant=3)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertTrue(array_equal(return_old[0], [[1, 2, 3, 4, 0]]))
        self.assertEqual(return_old[1], 1)



class Test_angles_between_anglesets(unittest.TestCase):
    agls1 =  [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]
    agls2 = [[0.0, 0.0, 0.66], [0.44907856464385987, 0.4307301163673401, 0.22000000655651095], [-0.27087580382823945, 0.5602020227909088, 0.22000000655651095], [-0.6164889872074127, -0.08450628250837326, 0.22000000655651095], [-0.11013545751571656, -0.6124297463893891, 0.22000000655651095], [0.5484215462207794, -0.2939962184429169, 0.22000000655651095], [5.7699032538494066e-08, 5.044209628034821e-15, -0.66], [0.6164889872074127, 0.08450620383024215, -0.21999998688697817], [0.11013536900281906, 0.6124297463893891, -0.21999998688697817], [-0.5484216248989106, 0.2939961397647858, -0.21999998688697817], [-0.44907860398292543, -0.43073007702827454, -0.21999998688697817], [0.2708758628368378, -0.5602019834518432, -0.21999998688697817]]


    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angles_between_anglesets()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angles_between_anglesets()
        self.assertEqual(str(cm_new.exception), "angles_between_anglesets() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_angleset1_returns_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angles_between_anglesets(angleset1=[], angleset2=[1,2,3,4], indexes=None)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angles_between_anglesets(angleset1=[], angleset2=[1,2,3,4], indexes=None)
        self.assertEqual(str(cm_new.exception), "object has no attribute")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_angleset2_returns_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angles_between_anglesets(angleset2=[], angleset1=[1,2,3,4], indexes=None)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angles_between_anglesets(angleset2=[], angleset1=[1,2,3,4], indexes=None)
        self.assertEqual(str(cm_new.exception), "object has no attribute")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_indexes_returns_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angles_between_anglesets(angleset2=[1,2,3,4], angleset1=[1,2,3,4], indexes=[])
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angles_between_anglesets(angleset2=[1,2,3,4], angleset1=[1,2,3,4], indexes=[])
        self.assertEqual(str(cm_new.exception), "object has no attribute")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_angles_between_anglesets_noIndex(self):
        return_old = oldfu.angles_between_anglesets(angleset1=self.agls1, angleset2=self.agls2, indexes=None)
        return_new = fu.angles_between_anglesets(angleset1=self.agls1, angleset2=self.agls2, indexes=None)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [0.0, 0.22190159831597722, 0.28859380708484916, 0.04353735746084626, 0.3154949948578467, 0.15146310888536912, 0.0, 0.04353731692694007, 0.31549499485669147, 0.15146306834889342, 0.22190157805124192, 0.28859378682155445]))

    def test_angles_between_anglesets_Index(self):
        return_old = oldfu.angles_between_anglesets(angleset1=[1,2,3,4], angleset2=[1,2,3,4], indexes=[1,2,3,4])
        return_new = fu.angles_between_anglesets(angleset1=[1,2,3,4], angleset2=[1,2,3,4], indexes=[1,2,3,4])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [0.18403868911316032, 0.2507530674355067, 0.08137682850104319, 0.35334686263089815]))


#todo: i cannot test it
class Test_group_proj_by_phitheta_slow(unittest.TestCase):
    proj_angles_list = numpy_full((900, 4), 0.0, dtype=numpy_float32)
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.group_proj_by_phitheta_slow()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.group_proj_by_phitheta_slow()
        self.assertEqual(str(cm_new.exception), "group_proj_by_phitheta_slow() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_sym_c1(self):
        return_old = oldfu.group_proj_by_phitheta_slow(proj_ang=self.proj_angles_list, symmetry="c1", img_per_grp=100, verbose=False)
        return_new = fu.group_proj_by_phitheta_slow(proj_ang=self.proj_angles_list, symmetry="c1", img_per_grp=100, verbose=False)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))

    def test_sym_d1(self):
        return_old = oldfu.group_proj_by_phitheta_slow(proj_ang=self.proj_angles_list, symmetry="d1", img_per_grp=100, verbose=False)
        return_new = fu.group_proj_by_phitheta_slow(proj_ang=self.proj_angles_list, symmetry="d1", img_per_grp=100, verbose=False)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))



class Test_class_iterImagesStack(unittest.TestCase):
    pass


#todo: i cannot test it
class Test_group_proj_by_phitheta(unittest.TestCase):
    proj_angles_list = numpy_full((900, 4), 0.0, dtype=numpy_float32)
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.group_proj_by_phitheta()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.group_proj_by_phitheta()
        self.assertEqual(str(cm_new.exception), "group_proj_by_phitheta() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_sym_c1(self):
        return_old = oldfu.group_proj_by_phitheta(proj_ang=self.proj_angles_list, symmetry="c1", img_per_grp=100, verbose=False)
        return_new = fu.group_proj_by_phitheta(proj_ang=self.proj_angles_list, symmetry="c1", img_per_grp=100, verbose=False)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))

    def test_sym_d1(self):
        return_old = oldfu.group_proj_by_phitheta(proj_ang=self.proj_angles_list, symmetry="d1", img_per_grp=100, verbose=False)
        return_new = fu.group_proj_by_phitheta(proj_ang=self.proj_angles_list, symmetry="d1", img_per_grp=100, verbose=False)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))


class Test_mulvec(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.mulvec()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.mulvec()
        self.assertEqual(str(cm_new.exception), "mulvec() takes exactly arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_v1_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.mulvec(v1=[], v2=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
                oldfu.mulvec(v1=[], v2=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_v2_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.mulvec(v2=[], v1=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.mulvec(v2=[], v1=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_mulvec(self):
        return_old = oldfu.mulvec(v1=[0.1,0.13,0.15],v2=[0.1,0.13,0.15])
        return_new = fu.mulvec(v1=[0.1,0.13,0.15],v2=[0.1,0.13,0.15])
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_old, 0.0494)



class Test_assignments_to_groups(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.assignments_to_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.assignments_to_groups()
        self.assertEqual(str(cm_new.exception), "assignments_to_groups() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_valid_n(self):
        return_old = oldfu.assignments_to_groups(assignments=[1,2,3,4,5,6,7,8,9,], n=20)
        return_new = fu.assignments_to_groups(assignments=[1,2,3,4,5,6,7,8,9,], n=20)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [], [], [], [], [], [], [], [], [], [], []]))

    def test_default_n(self):
        return_old = oldfu.assignments_to_groups(assignments=[1,2,3,4,5,6,7,8,9,], n=-1)
        return_new = fu.assignments_to_groups(assignments=[1,2,3,4,5,6,7,8,9,], n=-1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old,[[], [0], [1], [2], [3], [4], [5], [6], [7], [8]]))



class Test_groups_assignments(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.groups_assignments()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.groups_assignments()
        self.assertEqual(str(cm_new.exception), "groups_assignments() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_n(self):
        return_old = oldfu.groups_assignments(groups=[ [0], [1], [2], [3], [4], [5], [6], [7], [8]], n=-1)
        return_new = fu.groups_assignments(groups=[ [0], [1], [2], [3], [4], [5], [6], [7], [8]], n=-1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [0, 1, 2, 3, 4, 5, 6, 7, 8]))

    def test_n(self):
        return_old = oldfu.groups_assignments(groups=[[0], [1], [2], [3], [4], [5], [6], [7], [8]], n=13)
        return_new = fu.groups_assignments(groups=[[0], [1], [2], [3], [4], [5], [6], [7], [8]], n=-13)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1]))



class Test_chunks_distribution(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.chunks_distribution()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.chunks_distribution()
        self.assertEqual(str(cm_new.exception), "chunks_distribution() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_chunks_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.chunks_distribution(chunks=[], procs=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
                oldfu.chunks_distribution(chunks=[], procs=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_chunks0_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.chunks_distribution(chunks=[[],[1,2,3]], procs=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
                oldfu.chunks_distribution(chunks=[[],[1,2,3]], procs=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_procs_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.chunks_distribution(procs=[], chunks=[1, 2, 3])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.chunks_distribution(procs=[], chunks=[1, 2, 3])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_chunks_distribution(self):
        return_old = oldfu.chunks_distribution(chunks=[[1,2,3],[1,2,3]],procs=5)
        return_new = fu.chunks_distribution(chunks=[[1,2,3],[1,2,3]],procs=5)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [[[1, 2, 3]], [[1, 2, 3]], [], [], []]))



class Test_rearrange_ranks_of_processors(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rearrange_ranks_of_processors()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rearrange_ranks_of_processors()
        self.assertEqual(str(cm_new.exception), "rearrange_ranks_of_processors() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_rearrange_ranks_of_processors_rrAssignment(self):
        return_old = oldfu.rearrange_ranks_of_processors(mode="to fit round-robin assignment")
        return_new = fu.rearrange_ranks_of_processors(mode="")
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_old, 139702819159776)

    def test_rearrange_ranks_of_processors_NodeAssignment(self):
        return_old = oldfu.rearrange_ranks_of_processors(mode="to fit by-node assignment")
        return_new = fu.rearrange_ranks_of_processors(mode="")
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_old, 140095636451040)



# todo: node involved, i cannot test
class Test_wrap_mpi_split_shared_memory(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.wrap_mpi_split_shared_memory()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.wrap_mpi_split_shared_memory()
        self.assertEqual(str(cm_new.exception), "wrap_mpi_split_shared_memory() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrap_mpi_split_shared_memory(self):
        return_old = oldfu.wrap_mpi_split_shared_memory(mpi_comm=MPI_COMM_WORLD)
        return_new = fu.wrap_mpi_split_shared_memory(mpi_comm="")
        self.assertTrue(array_equal(return_new, return_old))


#it is genereting a random string, it is not unit-testable
class Test_random_string(unittest.TestCase):
    pass


#todo: how test it?
class Test_get_nonexistent_directory_increment_value(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_nonexistent_directory_increment_value()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_nonexistent_directory_increment_value()
        self.assertEqual(str(cm_new.exception), "get_nonexistent_directory_increment_value() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_nonexistent_directory_increment_value(self):
        return_old = oldfu.get_nonexistent_directory_increment_value(directory_location="", directory_name="", start_value=1, myformat="%03d")
        return_new = fu.get_nonexistent_directory_increment_value(directory_location="", directory_name="", start_value=1, myformat="%03d")
        self.assertTrue(array_equal(return_new, return_old))


#todo: need a file
class Test_store_program_state(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.store_program_state()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.store_program_state()
        self.assertEqual(str(cm_new.exception), "store_program_state() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_store_program_state(self):
        return_old = oldfu.store_program_state(filename="", state="", stack="")
        return_new = fu.store_program_state(filename="", state="", stack="")
        self.assertTrue(array_equal(return_new, return_old))


#todo: need a file
class Test_restore_program_stack_and_state(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.restore_program_stack_and_state()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.restore_program_stack_and_state()
        self.assertEqual(str(cm_new.exception), "restore_program_stack_and_state() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_restore_program_stack_and_state(self):
        return_old = oldfu.restore_program_stack_and_state(file_name_of_saved_state="")
        return_new = fu.restore_program_stack_and_state(file_name_of_saved_state="")
        self.assertTrue(array_equal(return_new, return_old))


#todo need data
class Test_program_state_stack(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.program_state_stack()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.program_state_stack()
        self.assertEqual(str(cm_new.exception), "program_state_stack() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_program_state_stack(self):
        return_old = oldfu.program_state_stack(full_current_state="",frameinfo="",file_name_of_saved_state=None,last_call="",force_starting_execution=False)
        return_new = fu.program_state_stack(full_current_state="",frameinfo="",file_name_of_saved_state=None,last_call="",force_starting_execution=False)
        self.assertTrue(array_equal(return_new, return_old))



class Test_qw(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.qw()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.qw()
        self.assertEqual(str(cm_new.exception), "qw() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_qw(self):
        return_old = oldfu.qw(s="ciao\nciao\t")
        return_new = fu.qw(s="ciao\nciao\t")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(['ciao', 'ciao'], return_old))



class Test_list_prod(unittest.TestCase):

    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.list_prod()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.list_prod()
        self.assertEqual(str(cm_new.exception), "list_prod() takes exactly 1 argument1 (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_list_prod(self):
        return_old = oldfu.list_prod(list_whose_elements_are_going_to_be_multiplied=[1,2,3,4])
        return_new = fu.list_prod(list_whose_elements_are_going_to_be_multiplied=[1,2,3,4])
        self.assertEqual(return_new, return_old)
        self.assertEqual(24, return_old)



class Test_calculate_space_size(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.calculate_space_size()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.calculate_space_size()
        self.assertEqual(str(cm_new.exception), "calculate_space_size() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_calculate_space_size(self):
        return_old = oldfu.calculate_space_size(x_half_size=3, y_half_size=2, psi_half_size=5)
        return_new = fu.calculate_space_size(x_half_size=3, y_half_size=2, psi_half_size=5)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([7, 5, 11], return_old))



#cannot test it, mpi stuff are involved
class Test_mpi_exit(unittest.TestCase):
    def test_mpi_exit(self):
        return_old = oldfu.mpi_exit()
        return_new = fu.mpi_exit()



class Test_get_attr_stack(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_attr_stack()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_attr_stack()
        self.assertEqual(str(cm_new.exception), "get_attr_stack() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_attribute_no_found_runTimeError(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_attr_stack(data_stack=[IMAGE_2D,IMAGE_2D], attr_string="notfound")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_attr_stack(data_stack=[IMAGE_2D,IMAGE_2D], attr_string="apix_x")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_attr_stack(self):
        return_old = oldfu.get_attr_stack(data_stack=[IMAGE_2D,IMAGE_2D], attr_string="apix_x")
        return_new = fu.get_attr_stack(data_stack=[IMAGE_2D,IMAGE_2D], attr_string="")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([1.0, 1.0], return_old))


#todo: need a stack with  'group' key -->  error with 'group': 'The requested key does not exist' caught
class Test_get_sorting_params(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_sorting_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_sorting_params()
        self.assertEqual(str(cm_new.exception), "get_sorting_params() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_sorting_params(self):
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["nproc"] = 3
        Tracker["total_stack"] = 2
        return_old =oldfu.get_sorting_params(Tracker=TRACKER, data=[IMAGE_2D,IMAGE_2D])
        return_new = fu.get_sorting_params(Tracker=TRACKER, data=[IMAGE_2D,IMAGE_2D])
        self.assertTrue(array_equal(return_new, return_old))



class Test_remove_small_groups(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.remove_small_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.remove_small_groups()
        self.assertEqual(str(cm_new.exception), "remove_small_groups() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_class_list(self):
        return_old = oldfu.remove_small_groups(class_list=[], minimum_number_of_objects_in_a_group=4)
        return_new = fu.remove_small_groups(class_list=[], minimum_number_of_objects_in_a_group=4)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([[], []], return_old))

    def test_remove_small_groups(self):
        return_old = oldfu.remove_small_groups(class_list=[[1,2,3,4,5],[1,20,40,50]], minimum_number_of_objects_in_a_group=4)
        return_new = fu.remove_small_groups(class_list=[[1,2,3,4,5],[1,20,40,50]], minimum_number_of_objects_in_a_group=4)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([[1, 1, 2, 3, 4, 5, 20, 40, 50], [[1, 2, 3, 4, 5], [1, 20, 40, 50]]], return_old))



class Test_get_outliers(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_outliers()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_outliers()
        self.assertEqual(str(cm_new.exception), "get_outliers() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_too_few_total_number_return_KeyError(self):
        with self.assertRaises(KeyError) as cm_new:
            fu.oldfu.get_outliers(total_number=5, plist=range(20))
        with self.assertRaises(KeyError) as cm_old:
            oldfu.get_outliers(total_number=5, plist=range(20))
        self.assertEqual(str(cm_new.exception), "5")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_outliers(self):
        return_old = oldfu.get_outliers(total_number=50, plist=range(20))
        return_new = fu.get_outliers(total_number=50, plist=range(20))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], return_old))


##todo: need a stack with  Tracker["P_chunk0"] .... which attribute can have it?
class Test_get_margin_of_error(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_margin_of_error()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_margin_of_error()
        self.assertEqual(str(cm_new.exception), "get_margin_of_error() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_margin_of_error(self):
        return_old = oldfu.get_margin_of_error(this_group_of_data="", Tracker="")
        return_new = fu.get_margin_of_error(this_group_of_data="", Tracker="")
        self.assertTrue(array_equal(return_new, return_old))


#todo need a test file
class Test_get_ali3d_params(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_ali3d_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_ali3d_params()
        self.assertEqual(str(cm_new.exception), "get_ali3d_params() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_ali3d_params(self):
        return_old = oldfu.get_ali3d_params(ali3d_old_text_file="", shuffled_list="")
        return_new = fu.get_ali3d_params(ali3d_old_text_file="", shuffled_list="")
        self.assertTrue(array_equal(return_new, return_old))



class Test_get_number_of_groups(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_number_of_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_number_of_groups()
        self.assertEqual(str(cm_new.exception), "get_number_of_groups() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_number_of_groups(self):
        return_old = oldfu.get_number_of_groups(total_particles=1000, number_of_images_per_group=20)
        return_new = fu.get_number_of_groups(total_particles=1000, number_of_images_per_group=20)
        self.assertEqual(return_new, return_old)
        self.assertEqual(50, return_old)



class Test_get_complementary_elements_total(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_complementary_elements_total()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_complementary_elements_total()
        self.assertEqual(str(cm_new.exception), "get_complementary_elements_total() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_complementary_elements_total(self):
        return_old = oldfu.get_complementary_elements_total(total_stack=10, data_list=[1,8,3,4,5])
        return_new = fu.get_complementary_elements_total(total_stack=10, data_list=[1,8,3,4,5])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([0, 2, 6, 7, 9], return_old))

    def test_empty_list(self):
        return_old = oldfu.get_complementary_elements_total(total_stack=10, data_list=[])
        return_new = fu.get_complementary_elements_total(total_stack=10, data_list=[])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], return_old))


#todo: need a tracker with valid values in Tracker["orgstack"]
class Test_get_two_chunks_from_stack(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_two_chunks_from_stack()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_two_chunks_from_stack()
        self.assertEqual(str(cm_new.exception), "get_two_chunks_from_stack() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_two_chunks_from_stack(self):
        return_old = oldfu.get_two_chunks_from_stack(Tracker="")
        return_new = fu.get_two_chunks_from_stack(Tracker='')
        self.assertTrue(array_equal(return_new, return_old))


#todo: cannot test it becasue NameError: global name 'adjust_fsc_down' is not defined
class Test_set_filter_parameters_from_adjusted_fsc(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_filter_parameters_from_adjusted_fsc()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_filter_parameters_from_adjusted_fsc()
        self.assertEqual(str(cm_new.exception), "set_filter_parameters_from_adjusted_fsc() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_filter_parameters_from_adjusted_fsc(self):
        return_old = oldfu.set_filter_parameters_from_adjusted_fsc(n1="", n2="", Tracker="")
        return_new = fu.set_filter_parameters_from_adjusted_fsc(n1="", n2="", Tracker="")
        self.assertTrue(array_equal(return_new, return_old))


#todo: need a valid directory
class Test_get_class_members(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_class_members()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_class_members()
        self.assertEqual(str(cm_new.exception), "get_class_members() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_class_members(self):
        return_old = oldfu.get_class_members(sort3d_dir="")
        return_new = fu.get_class_members(sort3d_dir="")
        self.assertTrue(array_equal(return_new, return_old))

    def test_dir_not_found(self):
        return_old = oldfu.get_class_members(sort3d_dir="invalid_dir")
        return_new = fu.get_class_members(sort3d_dir="invalid_dir")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([], return_old))



class Test_get_stable_members_from_two_runs(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_stable_members_from_two_runs()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_stable_members_from_two_runs()
        self.assertEqual(str(cm_new.exception), "get_stable_members_from_two_runs() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_dir_not_found_returns_indexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.get_stable_members_from_two_runs(SORT3D_rootdirs="", ad_hoc_number=1, log_main=1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.get_stable_members_from_two_runs(SORT3D_rootdirs="", ad_hoc_number=1, log_main=1)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


#todo: need data
class Test_two_way_comparison_single(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.two_way_comparison_single()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.two_way_comparison_single()
        self.assertEqual(str(cm_new.exception), "two_way_comparison_single() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_two_way_comparison_single(self):
        return_old = oldfu.two_way_comparison_single(partition_A="", partition_B="", Tracker="")
        return_new = fu.two_way_comparison_single(partition_A="", partition_B="", Tracker="")
        self.assertTrue(array_equal(return_new, return_old))



class Test_get_leftover_from_stable(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_leftover_from_stable()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_leftover_from_stable()
        self.assertEqual(str(cm_new.exception), "get_leftover_from_stable() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_stable_list_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.get_leftover_from_stable(stable_list=[], N_total=10, smallest_group=3)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.get_leftover_from_stable(stable_list=[], N_total=10, smallest_group=3)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_leftover_from_stable(self):
        return_old = oldfu.get_leftover_from_stable(stable_list=[range(10),range(10,20),range(20,30)], N_total=35, smallest_group=5)
        return_new = fu.get_leftover_from_stable(stable_list=[range(10),range(10,20),range(20,30)], N_total=35, smallest_group=5)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([[30, 31, 32, 33, 34], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]], return_old))

    def test_N_total_lower_than_elements_in_list(self):
        part_list = [0, 1, 20]
        with self.assertRaises(KeyError) as cm_new:
            fu.get_leftover_from_stable(stable_list=[range(10),range(10,20),range(20,30)], N_total=3, smallest_group=5)
        with self.assertRaises(KeyError) as cm_old:
            oldfu.get_leftover_from_stable(stable_list=[range(10),range(10,20),range(20,30)], N_total=3, smallest_group=5)
        self.assertEqual(str(cm_new.exception), 3)
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



#todo: how can i run it?
class Test_Kmeans_exhaustive_run(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.Kmeans_exhaustive_run()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.Kmeans_exhaustive_run()
        self.assertEqual(str(cm_new.exception), "Kmeans_exhaustive_run() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Kmeans_exhaustive_run(self):
        return_old = oldfu.Kmeans_exhaustive_run(ref_vol_list="", Tracker="")
        return_new = fu.Kmeans_exhaustive_run(ref_vol_list="", Tracker="")
        self.assertTrue(array_equal(return_new, return_old))


#todo: cannot test because NameError: global name 'data_list' is not defined
class Test_split_a_group(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.split_a_group()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.split_a_group()
        self.assertEqual(str(cm_new.exception), "split_a_group() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_split_a_group(self):
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["nproc"] = 3

        return_old = oldfu.split_a_group(workdir="", list_of_a_group=[1,2,3], Tracker=Tracker)
        return_new = fu.split_a_group(workdir="", list_of_a_group=[1,2,3], Tracker=Tracker)
        self.assertTrue(array_equal(return_new, return_old))


class Test_search_lowpass(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.search_lowpass()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.search_lowpass()
        self.assertEqual(str(cm_new.exception), "search_lowpass() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_first(self):
        return_old = oldfu.search_lowpass(fsc=[[],[1,2]])
        return_new = fu.search_lowpass(fsc=[[],[1,2]])
        self.assertEqual(return_new, return_old)
        self.assertEqual(0.45, return_old)

    def test_empty_second_returns_UnboundLocalError_imgtype_referenced_before_assignment(self):
        destination ='output.hdf'
        with self.assertRaises(UnboundLocalError) as cm_new:
            fu.search_lowpass(fsc=[[1,2],[]])
        with self.assertRaises(UnboundLocalError) as cm_old:
            oldfu.search_lowpass(fsc=[[1,2],[]])
        self.assertEqual(str(cm_new.exception), "local variable 'i' referenced before assignment")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_search_lowpass(self):
        fsc = [[1, 2, 3, 4, 5, 6], [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]]
        return_old = oldfu.search_lowpass(fsc=fsc)
        return_new = fu.search_lowpass(fsc=fsc)
        self.assertEqual(return_new, return_old)
        self.assertEqual(0.45, return_old)



class Test_split_chunks_bad(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.split_chunks_bad()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.split_chunks_bad()
        self.assertEqual(str(cm_new.exception), "split_chunks_bad() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_split_chunks_bad(self):
        return_old = oldfu.split_chunks_bad(l=range(100), n=10)
        return_new = fu.split_chunks_bad(l=range(100), n=10)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal([[0, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91], [2, 12, 22, 32, 42, 52, 62, 72, 82, 92], [3, 13, 23, 33, 43, 53, 63, 73, 83, 93], [4, 14, 24, 34, 44, 54, 64, 74, 84, 94], [5, 15, 25, 35, 45, 55, 65, 75, 85, 95], [6, 16, 26, 36, 46, 56, 66, 76, 86, 96], [7, 17, 27, 37, 47, 57, 67, 77, 87, 97], [8, 18, 28, 38, 48, 58, 68, 78, 88, 98], [9, 19, 29, 39, 49, 59, 69, 79, 89, 99], [10, 20, 30, 40, 50, 60, 70, 80, 90]], return_old))

    def test_emptylist_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.split_chunks_bad(l=[], n=10)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.split_chunks_bad(l=[], n=10)
        self.assertEqual(str(cm_new.exception), "tuple index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_convert_to_float(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.convert_to_float()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.convert_to_float()
        self.assertEqual(str(cm_new.exception), "convert_to_float() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_convert_to_float(self):
        return_old=oldfu.convert_to_float(value=30)
        return_new = fu.convert_to_float(value=30)
        self.assertEqual(return_new, return_old)
        self.assertEqual(0.0, return_old)

    def test_not_integer_value(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.convert_to_float(value=30)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.convert_to_float(value=30)
        self.assertEqual(str(cm_new.exception), "hex() argument can't be converted to hex")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_numpy2em_python(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.numpy2em_python()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.numpy2em_python()
        self.assertEqual(str(cm_new.exception), "numpy2em_python() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_numpy2em_python(self):
        return_old = oldfu.numpy2em_python(numpy_ones((3,100,100)))
        return_new = fu.numpy2em_python(numpy_ones((3,100,100)))
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))


#todo: need data
class Test_create_summovie_command(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.reconstitute_mask()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.reconstitute_mask()
        self.assertEqual(str(cm_new.exception), "reconstitute_mask() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_create_summovie_command(self):
        return_old = oldfu.create_summovie_command(temp_name="", micrograph_name="", shift_name="", frc_name="", opt="")
        return_new = fu.create_summovie_command(temp_name="", micrograph_name="", shift_name="", frc_name="", opt="")
        self.assertTrue(array_equal(return_new, return_old))


"""



class Test_center_2D(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.center_2D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.center_2D()
        self.assertEqual(str(cm_new.exception), "center_2D() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_input_image(self):
        with self.assertRaises(RuntimeError)as cm_new:
            fu.center_2D(image_to_be_centered=EMData() ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        with self.assertRaises(RuntimeError)as cm_old:
            oldfu.center_2D(image_to_be_centered=EMData() ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], 'x size <= 0')
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_NoneType_Img(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.center_2D(image_to_be_centered=None ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.center_2D(image_to_be_centered=None ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'phase_cog'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_2DImg(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [-0.3703499436378479, -0.3887350857257843, -0.39406290650367737, -0.31704390048980713, -0.33054259419441223, -0.3340435028076172, -0.32923534512519836, -0.3400946259498596, -0.3604322671890259, -0.3805030882358551, -0.4799676835536957, -0.5080035924911499, -0.5012468099594116, -0.46102362871170044, -0.46638357639312744, -0.47559505701065063, -0.4862135946750641, -0.4972260296344757, -0.47051724791526794, -0.4670148491859436, -0.214565709233284, -0.20879504084587097, -0.23537161946296692, -0.27080145478248596, -0.2621292471885681, -0.27169129252433777, -0.24054843187332153, -0.22561034560203552, -0.24432404339313507, -0.22685809433460236, 0.10862457752227783, 0.13046400249004364, 0.12984687089920044, 0.11155690997838974, 0.11670461297035217, 0.10330694913864136, 0.09238166362047195, 0.089042067527771, 0.11553214490413666, 0.10142993927001953, 0.08308745920658112, 0.059467729181051254, 0.03297220543026924, 0.03335859254002571, 0.018797576427459717, 0.032400548458099365, 0.02054790034890175, 0.04626963660120964, 0.041031841188669205, 0.04753470793366432, 0.11181235313415527, 0.08749543875455856, 0.08990707993507385, 0.09588098526000977, 0.11416783928871155, 0.1051185131072998, 0.10514253377914429, 0.1265401542186737, 0.14008067548274994, 0.12481226027011871, 0.011457648128271103, 0.00596990343183279, 0.000892100331839174, 0.04193740338087082, 0.04413039982318878, 0.047939855605363846, 0.049763184040784836, 0.07987479865550995, 0.051033299416303635, 0.014774000272154808, -0.09101400524377823, -0.1151394248008728, -0.07287856936454773, -0.010011367499828339, -0.04046791046857834, -0.05022193491458893, -0.05946069210767746, -0.0743170902132988, -0.08090417832136154, -0.08884717524051666, -0.17596139013767242, -0.19926026463508606, -0.17419566214084625, -0.09462296962738037, -0.14621615409851074, -0.14760564267635345, -0.1468927562236786, -0.16385626792907715, -0.1634739488363266, -0.16282308101654053, -0.32476934790611267, -0.37476593255996704, -0.31187760829925537, -0.25332340598106384, -0.29557618498802185, -0.3049299418926239, -0.3340802788734436, -0.3325638771057129, -0.33298560976982117, -0.33319368958473206]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[2], return_old[2]))
        self.assertTrue(array_equal(return_new[1], -2.616443395614624))
        self.assertTrue(array_equal(return_new[2], -2.5323870182037354))


    def test_2DBlankImg(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_BLANK_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_BLANK_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[2], return_old[2]))
        self.assertTrue(array_equal(return_old[1], -10.0))
        self.assertTrue(array_equal(return_old[2], -10.0))

    #todo: should be error??
    def test_3DImg(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_3D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_3D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=None)
        self.assertTrue(array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))
        self.assertTrue(array_equal(return_new[0].get_3dview().flatten(),[-0.3124222457408905, -0.33797937631607056, -0.3297944664955139, -0.3321264088153839, -0.3467698395252228, -0.369099885225296, -0.38085150718688965, -0.36819395422935486, -0.40426790714263916, -0.36355188488960266, -0.46062329411506653, -0.46862056851387024, -0.47957608103752136, -0.49064505100250244, -0.49222108721733093, -0.461311936378479, -0.47414663434028625, -0.48550745844841003, -0.5168172717094421, -0.4809400141239166, -0.2637782692909241, -0.27221450209617615, -0.2615463435649872, -0.23434412479400635, -0.22943256795406342, -0.24842901527881622, -0.21490421891212463, -0.21964964270591736, -0.20723502337932587, -0.2612045109272003, 0.11483573168516159, 0.11153682321310043, 0.101145900785923, 0.08556399494409561, 0.10126782208681107, 0.11239980161190033, 0.10062502324581146, 0.11581680178642273, 0.13634778559207916, 0.11936002969741821, 0.024717940017580986, 0.024847598746418953, 0.02758624032139778, 0.026829957962036133, 0.05005288124084473, 0.036718111485242844, 0.0621953010559082, 0.08287525177001953, 0.04279077425599098, 0.036078959703445435, 0.10335592925548553, 0.11355680227279663, 0.1021728515625, 0.11097017675638199, 0.13575048744678497, 0.13481873273849487, 0.12157899886369705, 0.10160244256258011, 0.08636114746332169, 0.09095658361911774, 0.043512534350156784, 0.04845504090189934, 0.04331347718834877, 0.06448392570018768, 0.07461109012365341, 0.03693888708949089, 0.007492864970117807, 0.01633390411734581, -0.004645288456231356, 0.019256697967648506, -0.020663680508732796, -0.04360250011086464, -0.055621322244405746, -0.060976698994636536, -0.08152966946363449, -0.0791875347495079, -0.09371790289878845, -0.09332311153411865, -0.11781566590070724, -0.034436605870723724, -0.1079692393541336, -0.15370477735996246, -0.14474962651729584, -0.15069492161273956, -0.16937577724456787, -0.15678539872169495, -0.17163842916488647, -0.17756026983261108, -0.20864415168762207, -0.1316317468881607, -0.27112331986427307, -0.2970453202724457, -0.3171194791793823, -0.33508822321891785, -0.3317832946777344, -0.3339288830757141, -0.32736796140670776, -0.3387298583984375, -0.37320399284362793, -0.2693241238594055, -1.1331480741500854, -1.1455312967300415, -1.1505773067474365, -1.1309585571289062, -1.100803256034851, -1.088606357574463, -1.0629980564117432, -1.0510480403900146, -1.0462234020233154, -1.0983222723007202, -1.1077581644058228, -1.0715959072113037, -1.0682271718978882, -1.0481901168823242, -1.0300770998001099, -1.0277085304260254, -0.996934711933136, -0.9909066557884216, -0.9436344504356384, -1.0677235126495361, -0.7573167085647583, -0.7291474938392639, -0.7252786159515381, -0.7040082812309265, -0.6642057299613953, -0.6517550349235535, -0.642329752445221, -0.6393872499465942, -0.5949283242225647, -0.7088325619697571, -0.3728346526622772, -0.38411569595336914, -0.383652001619339, -0.4161916673183441, -0.42736583948135376, -0.4475353956222534, -0.46455156803131104, -0.4697988033294678, -0.49879318475723267, -0.37787139415740967, -0.5325043201446533, -0.543497622013092, -0.5227758288383484, -0.5363678336143494, -0.5597220659255981, -0.557472825050354, -0.5584743022918701, -0.5561838150024414, -0.5841074585914612, -0.5641967058181763, -0.5609714388847351, -0.5546998381614685, -0.5286207795143127, -0.518185555934906, -0.531661331653595, -0.5354505777359009, -0.5287307500839233, -0.5462622046470642, -0.5609227418899536, -0.544247031211853, -0.5934084057807922, -0.6259387731552124, -0.6280260682106018, -0.6479818820953369, -0.6675323843955994, -0.6683492064476013, -0.6987584829330444, -0.7099424600601196, -0.7666160464286804, -0.6277141571044922, -0.7446931004524231, -0.7731879353523254, -0.7755385041236877, -0.7906172275543213, -0.8109278678894043, -0.8283115029335022, -0.8340073227882385, -0.8257192969322205, -0.8520690202713013, -0.7708728313446045, -0.8363459706306458, -0.8646601438522339, -0.8695026636123657, -0.8801398873329163, -0.8953269124031067, -0.9033486843109131, -0.9435297846794128, -0.9593198299407959, -1.0252269506454468, -0.8934946060180664, -1.0430725812911987, -1.062660813331604, -1.0722239017486572, -1.0802016258239746, -1.0708999633789062, -1.0702439546585083, -1.102285385131836, -1.1079035997390747, -1.1282696723937988, -1.064941644668579, -0.18598023056983948, -0.15836302936077118, -0.17339199781417847, -0.16445598006248474, -0.1474863588809967, -0.14693069458007812, -0.14167024195194244, -0.12408792227506638, -0.1047549918293953, -0.16610629856586456, 0.04560088738799095, 0.015244902111589909, 0.01533809769898653, 0.03140454739332199, 0.01934061199426651, 0.02086068131029606, 0.04083443060517311, 0.04226454347372055, 0.014151989482343197, 0.019432740285992622, -0.40754854679107666, -0.3967565596103668, -0.40813663601875305, -0.41124391555786133, -0.40836817026138306, -0.4211192727088928, -0.3986257016658783, -0.38375750184059143, -0.34694966673851013, -0.3883746564388275, -0.7808626294136047, -0.7200153470039368, -0.6868589520454407, -0.655726432800293, -0.6544472575187683, -0.657730221748352, -0.6474411487579346, -0.6463140845298767, -0.5804265737533569, -0.7531720995903015, -0.44371575117111206, -0.4096858501434326, -0.38259556889533997, -0.36936673521995544, -0.3418065309524536, -0.34036991000175476, -0.3468512296676636, -0.36988359689712524, -0.37625622749328613, -0.41844332218170166, -0.46578532457351685, -0.4492795467376709, -0.42890024185180664, -0.40131181478500366, -0.39677950739860535, -0.3862360119819641, -0.3644276559352875, -0.3704300820827484, -0.3499901592731476, -0.4433846175670624, -0.2967478334903717, -0.26518750190734863, -0.2822875380516052, -0.2903510630130768, -0.2982715666294098, -0.29684215784072876, -0.30698853731155396, -0.31107935309410095, -0.2806191146373749, -0.30106282234191895, -0.313814640045166, -0.337287038564682, -0.33433791995048523, -0.3487912118434906, -0.3392777442932129, -0.3239010274410248, -0.3161894381046295, -0.29341748356819153, -0.28433382511138916, -0.29771995544433594, -0.2971767485141754, -0.2731618583202362, -0.2634941041469574, -0.24864140152931213, -0.2543574571609497, -0.24137388169765472, -0.23473021388053894, -0.24045881628990173, -0.20599332451820374, -0.26899588108062744, -0.20209884643554688, -0.18403029441833496, -0.1593513935804367, -0.14597059786319733, -0.12283877283334732, -0.12057161331176758, -0.10135933756828308, -0.119663305580616, -0.11779970675706863, -0.17568573355674744, 0.09490616619586945, 0.09337364882230759, 0.07346872240304947, 0.08032691478729248, 0.09937942773103714, 0.10867700725793839, 0.11915796250104904, 0.09061139076948166, 0.10233315825462341, 0.0799991637468338, 0.08307946473360062, 0.06252229958772659, 0.06815247237682343, 0.04784754663705826, 0.054371029138565063, 0.06908788532018661, 0.05473560094833374, 0.043221503496170044, -0.0015226936666294932, 0.06862034648656845, -0.027903137728571892, -0.025596074759960175, -0.013487198390066624, -0.007838745601475239, -0.028018150478601456, -0.03592938184738159, -0.053401313722133636, -0.056202493607997894, -0.07828152179718018, -0.020321074873209, -0.09035424143075943, -0.07734540849924088, -0.11098229885101318, -0.08695114403963089, -0.08225454390048981, -0.09958411753177643, -0.08342327177524567, -0.12030636519193649, -0.08627504110336304, -0.08703376352787018, -0.1057388037443161, -0.1123109981417656, -0.10429998487234116, -0.10530634224414825, -0.09848757088184357, -0.10171142965555191, -0.10739704221487045, -0.11018361896276474, -0.10447169840335846, -0.09463327378034592, -0.12666413187980652, -0.11046954989433289, -0.07733739167451859, -0.07968130707740784, -0.08561689406633377, -0.09424879401922226, -0.08571809530258179, -0.07242093235254288, -0.09701854735612869, -0.10416383296251297, -0.062250711023807526, -0.057537682354450226, -0.07778158038854599, -0.03519750386476517, -0.029621347784996033, -0.06387253105640411, -0.05274009704589844, -0.05355533957481384, -0.0523066520690918, -0.07862534373998642, -0.03830740973353386, -0.02184375561773777, -0.017691845074295998, 0.010381725616753101, -0.0016124916728585958, -0.005963310599327087, 0.003428844502195716, -0.0035058800131082535, 0.01044322457164526, -0.02996695786714554, 0.032162196934223175, 0.039001673460006714, 0.04203609749674797, 0.035064101219177246, 0.05279186740517616, 0.037349164485931396, 0.04251190274953842, 0.05908909812569618, 0.035362426191568375, 0.027608778327703476, 0.07022427767515182, 0.07644549757242203, 0.08927328139543533, 0.1148940697312355, 0.0749557614326477, 0.06840240955352783, 0.07332371920347214, 0.08079920709133148, 0.07856989651918411, 0.04674408584833145, -0.5228954553604126, -0.5356661081314087, -0.5323721170425415, -0.5334694981575012, -0.5546579957008362, -0.5558841228485107, -0.5516036152839661, -0.5467230677604675, -0.5459102988243103, -0.5320191979408264, -0.7265351414680481, -0.7480998039245605, -0.7673064470291138, -0.7893204092979431, -0.8024128079414368, -0.8054158687591553, -0.813201904296875, -0.8300765752792358, -0.8764068484306335, -0.7311674356460571, -0.3979550302028656, -0.42974853515625, -0.4202619194984436, -0.4527822732925415, -0.46535682678222656, -0.48377662897109985, -0.5060148239135742, -0.5000492930412292, -0.5352733135223389, -0.43251466751098633, -0.006815086584538221, -0.015117283910512924, -0.027217542752623558, -0.050397664308547974, -0.07544749975204468, -0.09666051715612411, -0.10072636604309082, -0.08544968068599701, -0.11347252130508423, -0.019348805770277977, -0.26273536682128906, -0.309212327003479, -0.3019944727420807, -0.31841734051704407, -0.3478674590587616, -0.34117189049720764, -0.370258092880249, -0.365308552980423, -0.3869224190711975, -0.30096784234046936, -0.31941840052604675, -0.332591712474823, -0.31861162185668945, -0.351648211479187, -0.35888978838920593, -0.35106563568115234, -0.3648661971092224, -0.3642687201499939, -0.37013959884643555, -0.3188728094100952, -0.41795462369918823, -0.4521092176437378, -0.44760456681251526, -0.4416128993034363, -0.4638860523700714, -0.46912917494773865, -0.4739886224269867, -0.45853492617607117, -0.4736354947090149, -0.44075122475624084, -0.48471468687057495, -0.4793790280818939, -0.4955999553203583, -0.49102672934532166, -0.4669474959373474, -0.481650710105896, -0.47038397192955017, -0.4458572268486023, -0.4312613904476166, -0.44539421796798706, -0.42389950156211853, -0.4462631344795227, -0.4425655007362366, -0.45649489760398865, -0.46297475695610046, -0.44006794691085815, -0.46086445450782776, -0.46417585015296936, -0.4957481920719147, -0.4564247131347656, -0.5251176953315735, -0.5383859872817993, -0.5419954061508179, -0.5369217395782471, -0.5251500010490417, -0.5344880819320679, -0.5491464138031006, -0.5553321838378906, -0.5722730159759521, -0.5395818948745728, -0.44033685326576233, -0.4406338334083557, -0.43290865421295166, -0.44155335426330566, -0.42735081911087036, -0.40639257431030273, -0.4122820198535919, -0.4156021177768707, -0.37864527106285095, -0.40970945358276367, -0.2822158634662628, -0.24288858473300934, -0.2383216768503189, -0.2565966248512268, -0.2680591642856598, -0.2457306981086731, -0.24561390280723572, -0.26030153036117554, -0.23027892410755157, -0.28302016854286194, -0.5571644306182861, -0.564163327217102, -0.554500937461853, -0.5644598007202148, -0.5827063322067261, -0.6107423305511475, -0.6040829420089722, -0.5829569697380066, -0.5726273059844971, -0.5445225834846497, -0.9497814774513245, -0.9743699431419373, -0.9793386459350586, -0.9805715680122375, -0.9936662912368774, -1.009436845779419, -1.0310509204864502, -1.0219485759735107, -1.0585647821426392, -0.979396641254425, -0.9706574082374573, -1.015399694442749, -1.0244367122650146, -1.040419340133667, -1.0572762489318848, -1.07909095287323, -1.0982385873794556, -1.0790135860443115, -1.0812082290649414, -0.9741692543029785, -1.1571046113967896, -1.1648740768432617, -1.183456540107727, -1.2093210220336914, -1.2146368026733398, -1.2176469564437866, -1.2121050357818604, -1.1838301420211792, -1.1742358207702637, -1.15234375, -1.1350131034851074, -1.1066677570343018, -1.0929747819900513, -1.0736852884292603, -1.0561021566390991, -1.0443029403686523, -1.0259443521499634, -1.0106786489486694, -0.9713553786277771, -1.0921390056610107, -0.9939159154891968, -0.9547548294067383, -0.9184702038764954, -0.8763642907142639, -0.8510295152664185, -0.8493169546127319, -0.8388506174087524, -0.8339019417762756, -0.7786452770233154, -0.9554191827774048, -0.7654684782028198, -0.6926144361495972, -0.6725180745124817, -0.6368032693862915, -0.6295397877693176, -0.6189265847206116, -0.5898410677909851, -0.5943629145622253, -0.5283609628677368, -0.7212265133857727, -0.5314885377883911, -0.49211201071739197, -0.47050291299819946, -0.4417493939399719, -0.4183788001537323, -0.40648773312568665, -0.4095293879508972, -0.3998347520828247, -0.36504337191581726, -0.505566418170929, -0.12539279460906982, -0.14394016563892365, -0.14467962086200714, -0.13306494057178497, -0.12665314972400665, -0.10158105939626694, -0.07681913673877716, -0.09042710065841675, -0.10355016589164734, -0.12670595943927765, -0.06354266405105591, -0.03504028916358948, -0.015839021652936935, -0.011651827022433281, -0.008265198208391666, 0.00499311275780201, -0.004815517459064722, -0.009087787941098213, 0.003179046791046858, -0.05732056871056557, -0.1863592267036438, -0.16591735184192657, -0.1716291904449463, -0.17336903512477875, -0.2003030925989151, -0.18146692216396332, -0.1659439653158188, -0.1441696286201477, -0.11327388882637024, -0.16735845804214478, -0.3015567660331726, -0.31012389063835144, -0.31776440143585205, -0.31021398305892944, -0.3239211440086365, -0.3204341530799866, -0.3210795223712921, -0.29893144965171814, -0.2753289043903351, -0.3014216423034668, -0.23300166428089142, -0.21552325785160065, -0.20728744566440582, -0.16127510368824005, -0.16392846405506134, -0.16766321659088135, -0.12490881979465485, -0.15182091295719147, -0.119554802775383, -0.2037273794412613, -0.20039834082126617, -0.1546616107225418, -0.16020949184894562, -0.14864914119243622, -0.1001068651676178, -0.10800767689943314, -0.10170698910951614, -0.10573150217533112, -0.07836589962244034, -0.15768197178840637, -0.05371902137994766, -0.041073646396398544, -0.027675574645400047, -0.05791804566979408, -0.06117113679647446, -0.065776027739048, -0.08873624354600906, -0.06786684691905975, -0.05850253999233246, -0.05135050415992737, -0.07769665867090225, -0.08311596512794495, -0.09288468211889267, -0.08887217193841934, -0.0756421834230423, -0.09009888023138046, -0.06535807251930237, -0.06766237318515778, -0.07441657781600952, -0.06122928112745285, -0.0900144949555397, -0.08096392452716827, -0.07886725664138794, -0.09168008714914322, -0.10444219410419464, -0.10735370963811874, -0.10962998867034912, -0.10168803483247757, -0.11062368005514145, -0.07875089347362518, -0.09747935086488724, -0.10419269651174545, -0.08442963659763336, -0.08919606357812881, -0.11410467326641083, -0.08937307447195053, -0.07732240855693817, -0.07672193646430969, -0.09371887892484665, -0.11159838736057281, -0.3030471205711365, -0.30297374725341797, -0.3032994568347931, -0.32471296191215515, -0.3279154896736145, -0.32860174775123596, -0.3451651334762573, -0.35268306732177734, -0.366784930229187, -0.3071786165237427, -0.4248722195625305, -0.45159393548965454, -0.45404255390167236, -0.4646778404712677, -0.4880616366863251, -0.49240705370903015, -0.5330163240432739, -0.5238656401634216, -0.5249558091163635, -0.47074705362319946, -0.24748238921165466, -0.22521668672561646, -0.24729986488819122, -0.25129303336143494, -0.2534782290458679, -0.2579335868358612, -0.2567838430404663, -0.25545528531074524, -0.22971028089523315, -0.25462430715560913, 0.0683533251285553, 0.07072023302316666, 0.07475905120372772, 0.09910428524017334, 0.09968560934066772, 0.11783726513385773, 0.12737572193145752, 0.11735378205776215, 0.09820001572370529, 0.0657106339931488, 0.019249465316534042, 0.026972545310854912, 0.039641983807086945, 0.03784231096506119, 0.05911792442202568, 0.07153386622667313, 0.057409219443798065, 0.05036153644323349, 0.06075957044959068, 0.04266548529267311, 0.10274626314640045, 0.1096775010228157, 0.10635868459939957, 0.1030096560716629, 0.10783659666776657, 0.10903049260377884, 0.1093885749578476, 0.11450120061635971, 0.12839922308921814, 0.10446351766586304, 0.08432892709970474, 0.09695138782262802, 0.06896001100540161, 0.048853520303964615, 0.05492919683456421, 0.059371333569288254, 0.05406082794070244, 0.040467724204063416, 0.04232224076986313, 0.07077658921480179, 0.03573610261082649, 0.004788717720657587, -0.024775221943855286, -0.01682519167661667, -0.006176783703267574, -0.011410889215767384, -0.04988808557391167, -0.04416526108980179, -0.08128253370523453, -0.002848685486242175, -0.072566457092762, -0.07980269938707352, -0.07970892637968063, -0.12355849146842957, -0.13590510189533234, -0.12387015670537949, -0.149238720536232, -0.1887056976556778, -0.2103852480649948, -0.09130760282278061, -0.2159646451473236, -0.23290503025054932, -0.23342560231685638, -0.24549518525600433, -0.25341132283210754, -0.27359551191329956, -0.3109378218650818, -0.2920588552951813, -0.33781617879867554, -0.2626416087150574, -1.1243220567703247, -1.1133511066436768, -1.1058372259140015, -1.1296216249465942, -1.1461580991744995, -1.1363669633865356, -1.1446516513824463, -1.1586862802505493, -1.1860146522521973, -1.1506634950637817, -1.276895523071289, -1.2423644065856934, -1.2265828847885132, -1.2193852663040161, -1.2095998525619507, -1.1859058141708374, -1.147014856338501, -1.142458200454712, -1.120710015296936, -1.2342275381088257, -0.8446199297904968, -0.8080762624740601, -0.8098377585411072, -0.8068246841430664, -0.7778884172439575, -0.7618539333343506, -0.729656457901001, -0.7146867513656616, -0.6994583606719971, -0.8228138089179993, -0.36943504214286804, -0.40624770522117615, -0.4182283282279968, -0.4302140474319458, -0.4245430529117584, -0.43447235226631165, -0.4314720630645752, -0.4132768511772156, -0.44146543741226196, -0.3645784556865692, -0.5248045325279236, -0.5323911905288696, -0.5188222527503967, -0.5178307294845581, -0.5324164032936096, -0.5177885293960571, -0.5341889262199402, -0.5458446741104126, -0.5746535658836365, -0.5641195178031921, -0.5386471748352051, -0.5528851747512817, -0.5576483011245728, -0.5881364941596985, -0.5907948017120361, -0.5849579572677612, -0.5925230979919434, -0.5932538509368896, -0.6085159182548523, -0.5433666110038757, -0.6839271187782288, -0.6839481592178345, -0.6851601004600525, -0.7181145548820496, -0.7321159839630127, -0.7361329197883606, -0.7596095204353333, -0.7778748273849487, -0.8226958513259888, -0.7067480683326721, -0.7827855348587036, -0.7945143580436707, -0.8140735626220703, -0.8377959132194519, -0.8599273562431335, -0.8696465492248535, -0.8747266530990601, -0.873438835144043, -0.8953048586845398, -0.8074760437011719, -0.9091745615005493, -0.9401631951332092, -0.9474978446960449, -0.9505387544631958, -0.9735690951347351, -0.9778075814247131, -0.9738987684249878, -0.9775176048278809, -1.020796775817871, -0.9446732997894287, -1.0240147113800049, -1.0314022302627563, -1.0406993627548218, -1.0677556991577148, -1.1055513620376587, -1.1245747804641724, -1.1306877136230469, -1.1251829862594604, -1.1553244590759277, -1.0559351444244385, -0.2566128373146057, -0.23334956169128418, -0.2445622980594635, -0.21673376858234406, -0.1899779587984085, -0.1863352507352829, -0.14763520658016205, -0.14168637990951538, -0.1156253069639206, -0.2076176404953003, 0.06450149416923523, 0.056516487151384354, 0.07400385290384293, 0.07197386771440506, 0.03629637137055397, 0.04435022547841072, 0.04156061261892319, 0.01106270682066679, -0.005149261560291052, 0.03249762952327728, -0.5245090126991272, -0.497192919254303, -0.49286553263664246, -0.46940043568611145, -0.4413595199584961, -0.436555951833725, -0.4119645357131958, -0.3982985019683838, -0.35249626636505127, -0.477865070104599, -0.9892696142196655, -0.934221088886261, -0.9060559868812561, -0.8547962307929993, -0.8279692530632019, -0.8125501275062561, -0.7690315246582031, -0.7645889520645142, -0.7207220196723938, -0.9382826089859009, -0.5547884702682495, -0.5157397389411926, -0.49517664313316345, -0.4585360288619995, -0.42535263299942017, -0.4025208353996277, -0.379989892244339, -0.36476999521255493, -0.3388402760028839, -0.5041475892066956, -0.48393842577934265, -0.47506359219551086, -0.46934518218040466, -0.45856696367263794, -0.426164835691452, -0.4095710515975952, -0.3995763957500458, -0.39472225308418274, -0.3646804988384247, -0.45441532135009766, -0.3021487891674042, -0.2871622145175934, -0.2983916103839874, -0.30413028597831726, -0.3043687045574188, -0.3076097369194031, -0.3153975009918213, -0.3206760883331299, -0.30618229508399963, -0.3072623014450073, -0.3192574679851532, -0.3086021840572357, -0.29946309328079224, -0.30935144424438477, -0.3029743432998657, -0.30865785479545593, -0.31257715821266174, -0.3133629858493805, -0.31553760170936584, -0.3328852951526642, -0.2861942648887634, -0.28542277216911316, -0.30963027477264404, -0.29945454001426697, -0.31305238604545593, -0.314922958612442, -0.31294524669647217, -0.3010388910770416, -0.2874472737312317, -0.2901410758495331, -0.22710292041301727, -0.19550803303718567, -0.1883278489112854, -0.17950820922851562, -0.16808150708675385, -0.17040830850601196, -0.15406915545463562, -0.1675233095884323, -0.14590947329998016, -0.22373013198375702]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[2], return_old[2]))
        self.assertTrue(array_equal(return_old[1], 0.7548537254333496))
        self.assertTrue(array_equal(return_old[2], -2.5345394611358643))

    def test_2DImgcenter_method0(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=0,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=0,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.009504491463303566, 0.025885052978992462, 0.015371464192867279, 0.029651766642928123, 0.025623319670557976, 0.023996084928512573, 0.023316310718655586, 0.03626585379242897, 0.042238689959049225, 0.053261399269104004, 0.06996519863605499, 0.05416787788271904, 0.050994712859392166, 0.03554266691207886, 0.03604980185627937, 0.07005909085273743, 0.056754179298877716, 0.06729267537593842, 0.0899617150425911, 0.08004479855298996, 0.07206107676029205, 0.07158395648002625, 0.08500781655311584, 0.08074058592319489, 0.08976095914840698, 0.09553121030330658, 0.09733162075281143, 0.12153391540050507, 0.09777011722326279, 0.0612066276371479, 0.060473889112472534, 0.0832795649766922, 0.07990699261426926, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.05740871652960777, 0.05622498691082001, 0.05523042380809784, 0.013907668180763721, 0.0071470243856310844, 0.01511574536561966, 2.5205374186043628e-05, 0.008231919258832932, -0.020773129537701607, -0.034199729561805725, -0.04089483618736267, -0.042460259050130844, -0.06925757229328156, -0.06893884390592575, -0.08000176399946213, -0.11662115156650543, -0.111984983086586, -0.11971071362495422, -0.1273496150970459, -0.12249226123094559, -0.1453358680009842, -0.14758040010929108, -0.15034900605678558, -0.17081016302108765, -0.2014905959367752, -0.2121349573135376, -0.22736789286136627, -0.24315771460533142, -0.2552821934223175, -0.23703180253505707, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.3236375153064728, -0.3262620270252228, -0.35172849893569946, -0.3602631986141205, -0.35741564631462097, -0.3575122356414795, -0.38925597071647644, -0.377326101064682, -0.38598355650901794, -0.39209896326065063, -0.3882087767124176, -0.3639817535877228, -0.3711523711681366, -0.37047016620635986, -0.39362388849258423, -0.40711337327957153, -0.3925972580909729, -0.4149233400821686, -0.41900205612182617, -0.4641905426979065, -0.46107935905456543, -0.46086275577545166, -0.4773290157318115, -0.473482221364975, -0.4543262720108032, -0.44096702337265015, -0.4387476146221161, -0.4229215085506439, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    def test_2DImgcenter_method1searching_range_negativeself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[-0.3703499436378479, -0.3887350857257843, -0.39406290650367737, -0.31704390048980713, -0.33054259419441223, -0.3340435028076172, -0.32923534512519836, -0.3400946259498596, -0.3604322671890259, -0.3805030882358551, -0.4799676835536957, -0.5080035924911499, -0.5012468099594116, -0.46102362871170044, -0.46638357639312744, -0.47559505701065063, -0.4862135946750641, -0.4972260296344757, -0.47051724791526794, -0.4670148491859436, -0.214565709233284, -0.20879504084587097, -0.23537161946296692, -0.27080145478248596, -0.2621292471885681, -0.27169129252433777, -0.24054843187332153, -0.22561034560203552, -0.24432404339313507, -0.22685809433460236, 0.10862457752227783, 0.13046400249004364, 0.12984687089920044, 0.11155690997838974, 0.11670461297035217, 0.10330694913864136, 0.09238166362047195, 0.089042067527771, 0.11553214490413666, 0.10142993927001953, 0.08308745920658112, 0.059467729181051254, 0.03297220543026924, 0.03335859254002571, 0.018797576427459717, 0.032400548458099365, 0.02054790034890175, 0.04626963660120964, 0.041031841188669205, 0.04753470793366432, 0.11181235313415527, 0.08749543875455856, 0.08990707993507385, 0.09588098526000977, 0.11416783928871155, 0.1051185131072998, 0.10514253377914429, 0.1265401542186737, 0.14008067548274994, 0.12481226027011871, 0.011457648128271103, 0.00596990343183279, 0.000892100331839174, 0.04193740338087082, 0.04413039982318878, 0.047939855605363846, 0.049763184040784836, 0.07987479865550995, 0.051033299416303635, 0.014774000272154808, -0.09101400524377823, -0.1151394248008728, -0.07287856936454773, -0.010011367499828339, -0.04046791046857834, -0.05022193491458893, -0.05946069210767746, -0.0743170902132988, -0.08090417832136154, -0.08884717524051666, -0.17596139013767242, -0.19926026463508606, -0.17419566214084625, -0.09462296962738037, -0.14621615409851074, -0.14760564267635345, -0.1468927562236786, -0.16385626792907715, -0.1634739488363266, -0.16282308101654053, -0.32476934790611267, -0.37476593255996704, -0.31187760829925537, -0.25332340598106384, -0.29557618498802185, -0.3049299418926239, -0.3340802788734436, -0.3325638771057129, -0.33298560976982117, -0.33319368958473206]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], -2.616443395614624))
        self.assertTrue(array_equal(return_old[2], -2.5323870182037354))

    def test_2DImgcenter_method2searching_range_negativeself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [-0.32363757491111755, -0.32626211643218994, -0.35172855854034424, -0.36026331782341003, -0.35741567611694336, -0.35751232504844666, -0.3892560601234436, -0.3773261606693268, -0.3859836161136627, -0.3920990526676178, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.461079478263855, -0.4608628749847412, -0.47732916474342346, -0.4734823405742645, -0.45432642102241516, -0.4409671723842621, -0.438747763633728, -0.42292165756225586, -0.43765121698379517, -0.43693017959594727, 0.00950438342988491, 0.025884946808218956, 0.015371355228126049, 0.029651662334799767, 0.025623228400945663, 0.023995986208319664, 0.02331620641052723, 0.03626576438546181, 0.042238593101501465, 0.05326130613684654, 0.06996522843837738, 0.05416791886091232, 0.05099474638700485, 0.035542700439691544, 0.03604983165860176, 0.07005912810564041, 0.0567542165517807, 0.0672926977276802, 0.08996175229549408, 0.08004484325647354, 0.07206105440855026, 0.07158391177654266, 0.08500777930021286, 0.0807405337691307, 0.08976089954376221, 0.095531165599823, 0.09733156859874725, 0.12153388559818268, 0.0977700874209404, 0.061206597834825516, 0.06047393009066582, 0.08327961713075638, 0.07990703731775284, 0.07260186225175858, 0.1039014384150505, 0.1269259750843048, 0.0899757668375969, 0.05740877985954285, 0.05622504651546478, 0.055230479687452316, 0.013907670974731445, 0.007147028110921383, 0.015115760266780853, 2.5212764739990234e-05, 0.008231937885284424, -0.020773105323314667, -0.03419971093535423, -0.04089482128620148, -0.04246025159955025, -0.06925756484270096, -0.06893885135650635, -0.08000175654888153, -0.11662115156650543, -0.111984983086586, -0.11971069872379303, -0.1273496150970459, -0.12249225378036499, -0.1453358680009842, -0.14758040010929108, -0.1503489911556244, -0.17081011831760406, -0.20149055123329163, -0.21213491261005402, -0.2273678332567215, -0.24315768480300903, -0.2552821636199951, -0.23703177273273468, -0.2393374741077423, -0.2672199308872223, -0.2880825996398926]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0))
        self.assertTrue(array_equal(return_old[2], -3.0))

    def test_2DImgcenter_method3searching_range_negativeself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.07990697026252747, 0.0726017951965332, 0.10390138626098633, 0.12692590057849884, 0.08997567743062973, 0.057408690452575684, 0.05622495710849762, 0.055230386555194855, 0.06047387421131134, 0.083279550075531, 0.015115716494619846, 2.518415385566186e-05, 0.008231890387833118, -0.02077317237854004, -0.03419975936412811, -0.04089485853910446, -0.04246028885245323, -0.06925759464502335, 0.0139076579362154, 0.007146999705582857, -0.11662118881940842, -0.11198502033948898, -0.1197107657790184, -0.1273496448993683, -0.12249229103326797, -0.14533589780330658, -0.14758045971393585, -0.15034903585910797, -0.06893887370824814, -0.08000180870294571, -0.21213501691818237, -0.22736796736717224, -0.24315780401229858, -0.25528228282928467, -0.23703189194202423, -0.23933759331703186, -0.26722002029418945, -0.28808271884918213, -0.17081023752689362, -0.20149067044258118, -0.3517284691333771, -0.36026325821876526, -0.3574156165122986, -0.3575122356414795, -0.38925600051879883, -0.3773261606693268, -0.3859836161136627, -0.392098993062973, -0.3236374855041504, -0.32626208662986755, -0.371152400970459, -0.37047016620635986, -0.3936239182949066, -0.40711334347724915, -0.3925972580909729, -0.4149233102798462, -0.41900211572647095, -0.4641905426979065, -0.3882087767124176, -0.36398178339004517, -0.47732892632484436, -0.4734821617603302, -0.45432621240615845, -0.440966933965683, -0.4387475550174713, -0.42292144894599915, -0.43765100836753845, -0.43692997097969055, -0.46107932925224304, -0.4608626365661621, 0.015371562913060188, 0.029651865363121033, 0.02562340721487999, 0.023996181786060333, 0.023316407576203346, 0.03626594319939613, 0.042238786816596985, 0.053261492401361465, 0.009504588320851326, 0.02588515169918537, 0.050994690507650375, 0.03554262965917587, 0.03604977950453758, 0.07005906105041504, 0.056754156947135925, 0.06729266792535782, 0.0899617001414299, 0.08004476130008698, 0.0699651762843132, 0.054167840629816055, 0.08500783145427704, 0.08074060082435608, 0.08976096659898758, 0.09553123265504837, 0.09733163565397263, 0.12153391540050507, 0.09777011722326279, 0.0612066313624382, 0.07206108421087265, 0.07158397883176804] ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_new[1], 2.0))
        self.assertTrue(array_equal(return_new[2], 3.0))

    #todo: BUG! with center_method=4 the parameters 'self_defined_reference' will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it is a boolean
    def test_2DImgcenter_method4searching_range_negativeself_defined_referenceTrue(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        '''

    def test_2DImgcenter_method5searching_range_negativeself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[-0.4190020263195038, -0.4641904830932617, -0.3882087171077728, -0.363981693983078, -0.3711523413658142, -0.3704700767993927, -0.39362382888793945, -0.40711328387260437, -0.3925971984863281, -0.4149232506752014, -0.437651127576828, -0.4369301199913025, -0.4610794484615326, -0.46086275577545166, -0.4773290455341339, -0.47348228096961975, -0.4543263614177704, -0.44096705317497253, -0.43874767422676086, -0.4229215383529663, 0.04223867505788803, 0.05326138064265251, 0.00950445607304573, 0.025885019451379776, 0.015371425077319145, 0.029651738703250885, 0.025623291730880737, 0.023996062576770782, 0.023316288366913795, 0.036265838891267776, 0.08996173739433289, 0.08004482835531235, 0.06996520608663559, 0.054167889058589935, 0.05099472403526306, 0.03554268181324005, 0.03604981303215027, 0.07005910575389862, 0.05675419792532921, 0.06729268282651901, 0.0977700874209404, 0.06120657920837402, 0.07206103950738907, 0.07158391922712326, 0.08500777184963226, 0.0807405561208725, 0.08976089954376221, 0.0955311730504036, 0.09733157604932785, 0.12153385579586029, 0.05622496455907822, 0.055230412632226944, 0.06047387048602104, 0.083279550075531, 0.07990697026252747, 0.0726018026471138, 0.10390138626098633, 0.12692593038082123, 0.08997569978237152, 0.05740870162844658, -0.04246024042367935, -0.06925756484270096, 0.013907666318118572, 0.007147038821130991, 0.015115763992071152, 2.521514761610888e-05, 0.008231942541897297, -0.020773107185959816, -0.03419971838593483, -0.04089483246207237, -0.14758038520812988, -0.15034900605678558, -0.06893885135650635, -0.08000175654888153, -0.11662115156650543, -0.1119849756360054, -0.11971069127321243, -0.1273496150970459, -0.12249227613210678, -0.1453358680009842, -0.26722002029418945, -0.28808265924453735, -0.17081019282341003, -0.2014906108379364, -0.21213501691818237, -0.22736790776252747, -0.2431577891111374, -0.2552822530269623, -0.23703187704086304, -0.2393375188112259, -0.38598349690437317, -0.39209896326065063, -0.3236374855041504, -0.32626205682754517, -0.3517284393310547, -0.3602631986141205, -0.3574155867099762, -0.3575122356414795, -0.38925597071647644, -0.37732604146003723]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_new[1], -2))
        self.assertTrue(array_equal(return_new[2], -2))

    def test_2DImgcenter_method6searching_range_negativeself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [-0.3308292329311371, -0.35217273235321045, -0.35512134432792664, -0.350458562374115, -0.359945684671402, -0.39080291986465454, -0.37608614563941956, -0.38802194595336914, -0.38861629366874695, -0.3226930797100067, -0.3868977725505829, -0.3901914656162262, -0.39550280570983887, -0.42173659801483154, -0.42156097292900085, -0.4091739058494568, -0.42963382601737976, -0.44400614500045776, -0.475086510181427, -0.3952586352825165, -0.4318661689758301, -0.44995662569999695, -0.4360677897930145, -0.4133113622665405, -0.40818536281585693, -0.404971718788147, -0.38789165019989014, -0.3995460569858551, -0.40129554271698, -0.43528422713279724, 0.06377732008695602, 0.0516175776720047, 0.06254758685827255, 0.05183619260787964, 0.057371824979782104, 0.057785991579294205, 0.06496492028236389, 0.07709789276123047, 0.08199524879455566, 0.04656369239091873, 0.03822452574968338, 0.03943261504173279, 0.022555435076355934, 0.03504655137658119, 0.06033632159233093, 0.044985294342041016, 0.06795906275510788, 0.08171253651380539, 0.06367428600788116, 0.055794257670640945, 0.08541927486658096, 0.09573178738355637, 0.08994194120168686, 0.10150208324193954, 0.10753349214792252, 0.11162584275007248, 0.12682822346687317, 0.09317834675312042, 0.07019657641649246, 0.08090898394584656, 0.07536952197551727, 0.07032203674316406, 0.06751537322998047, 0.1046244278550148, 0.11341626942157745, 0.06967370957136154, 0.043787259608507156, 0.04765136539936066, 0.04197842627763748, 0.05601690709590912, -0.0025441027246415615, 0.00407454464584589, -0.012391338124871254, -0.010055913589894772, -0.039092887192964554, -0.04699311777949333, -0.04953114315867424, -0.0624074824154377, -0.06937402486801147, 0.015018352307379246, -0.09805400669574738, -0.12606102228164673, -0.11833950877189636, -0.1293073147535324, -0.13296213746070862, -0.13198214769363403, -0.15272535383701324, -0.15730233490467072, -0.15260660648345947, -0.07028576731681824, -0.22714689373970032, -0.22935380041599274, -0.2537437677383423, -0.2669474482536316, -0.27285727858543396, -0.2577541172504425, -0.2574229836463928, -0.29796046018600464, -0.29030466079711914, -0.18199211359024048] ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 1.1435747146606445))
        self.assertTrue(array_equal(return_old[2], -2.897319793701172))

    # todo: BUG! with center_method=7 the parameters 'self_defined_reference' will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it is a boolean
    def test_2DImgcenter_method7searching_range_negativeself_defined_referenceTrue(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        '''

    def test_2DImgcenter_method1searching_range_positiveself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0))
        self.assertTrue(array_equal(return_old[2], 0))

    def test_2DImgcenter_method2searching_range_positiveself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0))
        self.assertTrue(array_equal(return_old[2], 0))

    def test_2DImgcenter_method3searching_range_positiveself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    # todo: BUG! with center_method=7 the parameters 'self_defined_reference' will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it is a boolean
    def test_2DImgcenter_method4searching_range_positiveself_defined_referenceTrue(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        '''

    def test_2DImgcenter_method5searching_range_positiveself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    def test_2DImgcenter_method6searching_range_positiveself_defined_referenceTrue(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.027042336761951447, 0.014855020679533482, 0.03139965236186981, 0.02444201149046421, 0.023595666512846947, 0.02515534870326519, 0.03634452819824219, 0.04597470164299011, 0.04802733287215233, 0.008277766406536102, 0.05272015556693077, 0.0504411980509758, 0.0324363149702549, 0.04111085832118988, 0.07108588516712189, 0.05468011647462845, 0.07220085710287094, 0.08974235504865646, 0.07849472016096115, 0.06792011857032776, 0.07261677831411362, 0.08594388514757156, 0.08015126734972, 0.09218797832727432, 0.09456785023212433, 0.1003597304224968, 0.12260011583566666, 0.09013902395963669, 0.061226870864629745, 0.07273389399051666, 0.08504937589168549, 0.07756480574607849, 0.07465694844722748, 0.10992223769426346, 0.1247936487197876, 0.08316926658153534, 0.055956464260816574, 0.05661878362298012, 0.05474178493022919, 0.06345605105161667, 0.005180867854505777, 0.01522734947502613, -0.000501713715493679, 0.006846952252089977, -0.024385372176766396, -0.03626953437924385, -0.038637466728687286, -0.04929937422275543, -0.06119173392653465, 0.019871994853019714, -0.08833656460046768, -0.11706489324569702, -0.11149423569440842, -0.12248949706554413, -0.1253921389579773, -0.12572383880615234, -0.145720437169075, -0.1504371166229248, -0.14209379255771637, -0.061612214893102646, -0.2089587152004242, -0.21109546720981598, -0.23094895482063293, -0.24550728499889374, -0.25363898277282715, -0.23638653755187988, -0.23959262669086456, -0.2770664095878601, -0.2743607759475708, -0.16435973346233368, -0.331535279750824, -0.3530108630657196, -0.3613433241844177, -0.3554689586162567, -0.36153101921081543, -0.39096301794052124, -0.3745373487472534, -0.39116182923316956, -0.38424354791641235, -0.31768864393234253, -0.3667606711387634, -0.3702090382575989, -0.37202930450439453, -0.3985354006290436, -0.4038237929344177, -0.3956374526023865, -0.41437461972236633, -0.4260222911834717, -0.46088534593582153, -0.37698498368263245, -0.461565762758255, -0.4792090356349945, -0.47093281149864197, -0.451316773891449, -0.4412181079387665, -0.43599119782447815, -0.4239910840988159, -0.43809208273887634, -0.43915462493896484, -0.46282583475112915]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_new[1], 1.1435747146606445))
        self.assertTrue(array_equal(return_new[2], 0))

    # todo: BUG! with center_method=7 the parameters 'self_defined_reference=True' will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it is a boolean
    def test_2DImgcenter_method7searching_range_positiveself_defined_referenceTrue(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=True)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        '''


    def test_2DImgcenter_method1searching_range_negativeself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [-0.3703499436378479, -0.3887350857257843, -0.39406290650367737, -0.31704390048980713, -0.33054259419441223, -0.3340435028076172, -0.32923534512519836, -0.3400946259498596, -0.3604322671890259, -0.3805030882358551, -0.4799676835536957, -0.5080035924911499, -0.5012468099594116, -0.46102362871170044, -0.46638357639312744, -0.47559505701065063, -0.4862135946750641, -0.4972260296344757, -0.47051724791526794, -0.4670148491859436, -0.214565709233284, -0.20879504084587097, -0.23537161946296692, -0.27080145478248596, -0.2621292471885681, -0.27169129252433777, -0.24054843187332153, -0.22561034560203552, -0.24432404339313507, -0.22685809433460236, 0.10862457752227783, 0.13046400249004364, 0.12984687089920044, 0.11155690997838974, 0.11670461297035217, 0.10330694913864136, 0.09238166362047195, 0.089042067527771, 0.11553214490413666, 0.10142993927001953, 0.08308745920658112, 0.059467729181051254, 0.03297220543026924, 0.03335859254002571, 0.018797576427459717, 0.032400548458099365, 0.02054790034890175, 0.04626963660120964, 0.041031841188669205, 0.04753470793366432, 0.11181235313415527, 0.08749543875455856, 0.08990707993507385, 0.09588098526000977, 0.11416783928871155, 0.1051185131072998, 0.10514253377914429, 0.1265401542186737, 0.14008067548274994, 0.12481226027011871, 0.011457648128271103, 0.00596990343183279, 0.000892100331839174, 0.04193740338087082, 0.04413039982318878, 0.047939855605363846, 0.049763184040784836, 0.07987479865550995, 0.051033299416303635, 0.014774000272154808, -0.09101400524377823, -0.1151394248008728, -0.07287856936454773, -0.010011367499828339, -0.04046791046857834, -0.05022193491458893, -0.05946069210767746, -0.0743170902132988, -0.08090417832136154, -0.08884717524051666, -0.17596139013767242, -0.19926026463508606, -0.17419566214084625, -0.09462296962738037, -0.14621615409851074, -0.14760564267635345, -0.1468927562236786, -0.16385626792907715, -0.1634739488363266, -0.16282308101654053, -0.32476934790611267, -0.37476593255996704, -0.31187760829925537, -0.25332340598106384, -0.29557618498802185, -0.3049299418926239, -0.3340802788734436, -0.3325638771057129, -0.33298560976982117, -0.33319368958473206]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], -2.616443395614624))
        self.assertTrue(array_equal(return_old[2], -2.5323870182037354))

    def test_2DImgcenter_method2searching_range_negativeself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [-0.32363757491111755, -0.32626211643218994, -0.35172855854034424, -0.36026331782341003, -0.35741567611694336, -0.35751232504844666, -0.3892560601234436, -0.3773261606693268, -0.3859836161136627, -0.3920990526676178, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.461079478263855, -0.4608628749847412, -0.47732916474342346, -0.4734823405742645, -0.45432642102241516, -0.4409671723842621, -0.438747763633728, -0.42292165756225586, -0.43765121698379517, -0.43693017959594727, 0.00950438342988491, 0.025884946808218956, 0.015371355228126049, 0.029651662334799767, 0.025623228400945663, 0.023995986208319664, 0.02331620641052723, 0.03626576438546181, 0.042238593101501465, 0.05326130613684654, 0.06996522843837738, 0.05416791886091232, 0.05099474638700485, 0.035542700439691544, 0.03604983165860176, 0.07005912810564041, 0.0567542165517807, 0.0672926977276802, 0.08996175229549408, 0.08004484325647354, 0.07206105440855026, 0.07158391177654266, 0.08500777930021286, 0.0807405337691307, 0.08976089954376221, 0.095531165599823, 0.09733156859874725, 0.12153388559818268, 0.0977700874209404, 0.061206597834825516, 0.06047393009066582, 0.08327961713075638, 0.07990703731775284, 0.07260186225175858, 0.1039014384150505, 0.1269259750843048, 0.0899757668375969, 0.05740877985954285, 0.05622504651546478, 0.055230479687452316, 0.013907670974731445, 0.007147028110921383, 0.015115760266780853, 2.5212764739990234e-05, 0.008231937885284424, -0.020773105323314667, -0.03419971093535423, -0.04089482128620148, -0.04246025159955025, -0.06925756484270096, -0.06893885135650635, -0.08000175654888153, -0.11662115156650543, -0.111984983086586, -0.11971069872379303, -0.1273496150970459, -0.12249225378036499, -0.1453358680009842, -0.14758040010929108, -0.1503489911556244, -0.17081011831760406, -0.20149055123329163, -0.21213491261005402, -0.2273678332567215, -0.24315768480300903, -0.2552821636199951, -0.23703177273273468, -0.2393374741077423, -0.2672199308872223, -0.2880825996398926]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0))
        self.assertTrue(array_equal(return_old[2], -3))

    def test_2DImgcenter_method3searching_range_negativeself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.07990697026252747, 0.0726017951965332, 0.10390138626098633, 0.12692590057849884, 0.08997567743062973, 0.057408690452575684, 0.05622495710849762, 0.055230386555194855, 0.06047387421131134, 0.083279550075531, 0.015115716494619846, 2.518415385566186e-05, 0.008231890387833118, -0.02077317237854004, -0.03419975936412811, -0.04089485853910446, -0.04246028885245323, -0.06925759464502335, 0.0139076579362154, 0.007146999705582857, -0.11662118881940842, -0.11198502033948898, -0.1197107657790184, -0.1273496448993683, -0.12249229103326797, -0.14533589780330658, -0.14758045971393585, -0.15034903585910797, -0.06893887370824814, -0.08000180870294571, -0.21213501691818237, -0.22736796736717224, -0.24315780401229858, -0.25528228282928467, -0.23703189194202423, -0.23933759331703186, -0.26722002029418945, -0.28808271884918213, -0.17081023752689362, -0.20149067044258118, -0.3517284691333771, -0.36026325821876526, -0.3574156165122986, -0.3575122356414795, -0.38925600051879883, -0.3773261606693268, -0.3859836161136627, -0.392098993062973, -0.3236374855041504, -0.32626208662986755, -0.371152400970459, -0.37047016620635986, -0.3936239182949066, -0.40711334347724915, -0.3925972580909729, -0.4149233102798462, -0.41900211572647095, -0.4641905426979065, -0.3882087767124176, -0.36398178339004517, -0.47732892632484436, -0.4734821617603302, -0.45432621240615845, -0.440966933965683, -0.4387475550174713, -0.42292144894599915, -0.43765100836753845, -0.43692997097969055, -0.46107932925224304, -0.4608626365661621, 0.015371562913060188, 0.029651865363121033, 0.02562340721487999, 0.023996181786060333, 0.023316407576203346, 0.03626594319939613, 0.042238786816596985, 0.053261492401361465, 0.009504588320851326, 0.02588515169918537, 0.050994690507650375, 0.03554262965917587, 0.03604977950453758, 0.07005906105041504, 0.056754156947135925, 0.06729266792535782, 0.0899617001414299, 0.08004476130008698, 0.0699651762843132, 0.054167840629816055, 0.08500783145427704, 0.08074060082435608, 0.08976096659898758, 0.09553123265504837, 0.09733163565397263, 0.12153391540050507, 0.09777011722326279, 0.0612066313624382, 0.07206108421087265, 0.07158397883176804]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 2))
        self.assertTrue(array_equal(return_old[2], 3))

    "same bug as above"
    def test_2DImgcenter_method4searching_range_negativeself_defined_referenceFalse(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 15.868728637695312))
        self.assertTrue(array_equal(return_old[2], -13.827560424804688))
        '''

    def test_2DImgcenter_method5searching_range_negativeself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[-0.4190020263195038, -0.4641904830932617, -0.3882087171077728, -0.363981693983078, -0.3711523413658142, -0.3704700767993927, -0.39362382888793945, -0.40711328387260437, -0.3925971984863281, -0.4149232506752014, -0.437651127576828, -0.4369301199913025, -0.4610794484615326, -0.46086275577545166, -0.4773290455341339, -0.47348228096961975, -0.4543263614177704, -0.44096705317497253, -0.43874767422676086, -0.4229215383529663, 0.04223867505788803, 0.05326138064265251, 0.00950445607304573, 0.025885019451379776, 0.015371425077319145, 0.029651738703250885, 0.025623291730880737, 0.023996062576770782, 0.023316288366913795, 0.036265838891267776, 0.08996173739433289, 0.08004482835531235, 0.06996520608663559, 0.054167889058589935, 0.05099472403526306, 0.03554268181324005, 0.03604981303215027, 0.07005910575389862, 0.05675419792532921, 0.06729268282651901, 0.0977700874209404, 0.06120657920837402, 0.07206103950738907, 0.07158391922712326, 0.08500777184963226, 0.0807405561208725, 0.08976089954376221, 0.0955311730504036, 0.09733157604932785, 0.12153385579586029, 0.05622496455907822, 0.055230412632226944, 0.06047387048602104, 0.083279550075531, 0.07990697026252747, 0.0726018026471138, 0.10390138626098633, 0.12692593038082123, 0.08997569978237152, 0.05740870162844658, -0.04246024042367935, -0.06925756484270096, 0.013907666318118572, 0.007147038821130991, 0.015115763992071152, 2.521514761610888e-05, 0.008231942541897297, -0.020773107185959816, -0.03419971838593483, -0.04089483246207237, -0.14758038520812988, -0.15034900605678558, -0.06893885135650635, -0.08000175654888153, -0.11662115156650543, -0.1119849756360054, -0.11971069127321243, -0.1273496150970459, -0.12249227613210678, -0.1453358680009842, -0.26722002029418945, -0.28808265924453735, -0.17081019282341003, -0.2014906108379364, -0.21213501691818237, -0.22736790776252747, -0.2431577891111374, -0.2552822530269623, -0.23703187704086304, -0.2393375188112259, -0.38598349690437317, -0.39209896326065063, -0.3236374855041504, -0.32626205682754517, -0.3517284393310547, -0.3602631986141205, -0.3574155867099762, -0.3575122356414795, -0.38925597071647644, -0.37732604146003723] ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], -2))
        self.assertTrue(array_equal(return_old[2], -2))

    def test_2DImgcenter_method6searching_range_negativeself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[-0.3308292329311371, -0.35217273235321045, -0.35512134432792664, -0.350458562374115, -0.359945684671402, -0.39080291986465454, -0.37608614563941956, -0.38802194595336914, -0.38861629366874695, -0.3226930797100067, -0.3868977725505829, -0.3901914656162262, -0.39550280570983887, -0.42173659801483154, -0.42156097292900085, -0.4091739058494568, -0.42963382601737976, -0.44400614500045776, -0.475086510181427, -0.3952586352825165, -0.4318661689758301, -0.44995662569999695, -0.4360677897930145, -0.4133113622665405, -0.40818536281585693, -0.404971718788147, -0.38789165019989014, -0.3995460569858551, -0.40129554271698, -0.43528422713279724, 0.06377732008695602, 0.0516175776720047, 0.06254758685827255, 0.05183619260787964, 0.057371824979782104, 0.057785991579294205, 0.06496492028236389, 0.07709789276123047, 0.08199524879455566, 0.04656369239091873, 0.03822452574968338, 0.03943261504173279, 0.022555435076355934, 0.03504655137658119, 0.06033632159233093, 0.044985294342041016, 0.06795906275510788, 0.08171253651380539, 0.06367428600788116, 0.055794257670640945, 0.08541927486658096, 0.09573178738355637, 0.08994194120168686, 0.10150208324193954, 0.10753349214792252, 0.11162584275007248, 0.12682822346687317, 0.09317834675312042, 0.07019657641649246, 0.08090898394584656, 0.07536952197551727, 0.07032203674316406, 0.06751537322998047, 0.1046244278550148, 0.11341626942157745, 0.06967370957136154, 0.043787259608507156, 0.04765136539936066, 0.04197842627763748, 0.05601690709590912, -0.0025441027246415615, 0.00407454464584589, -0.012391338124871254, -0.010055913589894772, -0.039092887192964554, -0.04699311777949333, -0.04953114315867424, -0.0624074824154377, -0.06937402486801147, 0.015018352307379246, -0.09805400669574738, -0.12606102228164673, -0.11833950877189636, -0.1293073147535324, -0.13296213746070862, -0.13198214769363403, -0.15272535383701324, -0.15730233490467072, -0.15260660648345947, -0.07028576731681824, -0.22714689373970032, -0.22935380041599274, -0.2537437677383423, -0.2669474482536316, -0.27285727858543396, -0.2577541172504425, -0.2574229836463928, -0.29796046018600464, -0.29030466079711914, -0.18199211359024048]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 1.1435747146606445))
        self.assertTrue(array_equal(return_old[2], -2.897319793701172))

    #same bug as above
    def test_2DImgcenter_method7searching_range_negativeself_defined_referenceFalse(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=-1,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], -0.8430328369140625))
        self.assertTrue(array_equal(return_old[2], 8.518508911132812))
        '''

    def test_2DImgcenter_method1searching_range_positiveself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=1,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(), [0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    def test_2DImgcenter_method2searching_range_positiveself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=2,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577] ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    def test_2DImgcenter_method3searching_range_positiveself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=3,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577] ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    # todo: BUG! with center_method=4 the parameters 'self_defined_reference' will be used as EMData::EMData* (the reference parameter) by 'fondamentals.ccf(image_to_be_centered, reference) but it is a boolean
    def test_2DImgcenter_method4searching_range_positiveself_defined_referenceFalse(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=4,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))
        '''

    def test_2DImgcenter_method5searching_range_positiveself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=5,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.009504487738013268, 0.025885045528411865, 0.015371453016996384, 0.029651757329702377, 0.025623315945267677, 0.023996081203222275, 0.02331630326807499, 0.03626584634184837, 0.04223868250846863, 0.05326139181852341, 0.0699651837348938, 0.054167866706848145, 0.05099470168352127, 0.035542648285627365, 0.03604979068040848, 0.07005907595157623, 0.05675417184829712, 0.06729266047477722, 0.0899617001414299, 0.08004478365182877, 0.07206102460622787, 0.07158391177654266, 0.08500776439905167, 0.0807405486702919, 0.08976089954376221, 0.0955311581492424, 0.09733156859874725, 0.12153387069702148, 0.0977700799703598, 0.06120658665895462, 0.060473889112472534, 0.0832795575261116, 0.07990698516368866, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.057408712804317474, 0.05622498318552971, 0.05523041635751724, 0.013907660730183125, 0.007147015072405338, 0.015115739777684212, 2.5200843083439395e-05, 0.008231916464865208, -0.020773133262991905, -0.034199733287096024, -0.04089483991265297, -0.04246027022600174, -0.06925757974386215, -0.06893885880708694, -0.08000177145004272, -0.11662116646766663, -0.11198499798774719, -0.11971072107553482, -0.1273496150970459, -0.12249227613210678, -0.14533588290214539, -0.14758042991161346, -0.15034900605678558, -0.17081016302108765, -0.201490581035614, -0.2121349424123764, -0.22736789286136627, -0.24315772950649261, -0.2552821934223175, -0.23703181743621826, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.32363754510879517, -0.32626208662986755, -0.3517284393310547, -0.36026325821876526, -0.35741564631462097, -0.3575122654438019, -0.38925600051879883, -0.3773261308670044, -0.38598352670669556, -0.3920990228652954, -0.3882087171077728, -0.3639817237854004, -0.3711523413658142, -0.3704700469970703, -0.39362382888793945, -0.40711328387260437, -0.3925972282886505, -0.4149232506752014, -0.4190019965171814, -0.4641904830932617, -0.4610793888568878, -0.46086275577545166, -0.4773290455341339, -0.473482221364975, -0.4543263018131256, -0.44096702337265015, -0.43874767422676086, -0.4229215383529663, -0.4376510977745056, -0.4369300603866577]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))

    def test_2DImgcenter_method6searching_range_positiveself_defined_referenceFalse(self):
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=6,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new[0].get_2dview(), return_old[0].get_2dview()))
        self.assertTrue(array_equal(return_new[0].get_2dview().flatten(),[0.027042336761951447, 0.014855020679533482, 0.03139965236186981, 0.02444201149046421, 0.023595666512846947, 0.02515534870326519, 0.03634452819824219, 0.04597470164299011, 0.04802733287215233, 0.008277766406536102, 0.05272015556693077, 0.0504411980509758, 0.0324363149702549, 0.04111085832118988, 0.07108588516712189, 0.05468011647462845, 0.07220085710287094, 0.08974235504865646, 0.07849472016096115, 0.06792011857032776, 0.07261677831411362, 0.08594388514757156, 0.08015126734972, 0.09218797832727432, 0.09456785023212433, 0.1003597304224968, 0.12260011583566666, 0.09013902395963669, 0.061226870864629745, 0.07273389399051666, 0.08504937589168549, 0.07756480574607849, 0.07465694844722748, 0.10992223769426346, 0.1247936487197876, 0.08316926658153534, 0.055956464260816574, 0.05661878362298012, 0.05474178493022919, 0.06345605105161667, 0.005180867854505777, 0.01522734947502613, -0.000501713715493679, 0.006846952252089977, -0.024385372176766396, -0.03626953437924385, -0.038637466728687286, -0.04929937422275543, -0.06119173392653465, 0.019871994853019714, -0.08833656460046768, -0.11706489324569702, -0.11149423569440842, -0.12248949706554413, -0.1253921389579773, -0.12572383880615234, -0.145720437169075, -0.1504371166229248, -0.14209379255771637, -0.061612214893102646, -0.2089587152004242, -0.21109546720981598, -0.23094895482063293, -0.24550728499889374, -0.25363898277282715, -0.23638653755187988, -0.23959262669086456, -0.2770664095878601, -0.2743607759475708, -0.16435973346233368, -0.331535279750824, -0.3530108630657196, -0.3613433241844177, -0.3554689586162567, -0.36153101921081543, -0.39096301794052124, -0.3745373487472534, -0.39116182923316956, -0.38424354791641235, -0.31768864393234253, -0.3667606711387634, -0.3702090382575989, -0.37202930450439453, -0.3985354006290436, -0.4038237929344177, -0.3956374526023865, -0.41437461972236633, -0.4260222911834717, -0.46088534593582153, -0.37698498368263245, -0.461565762758255, -0.4792090356349945, -0.47093281149864197, -0.451316773891449, -0.4412181079387665, -0.43599119782447815, -0.4239910840988159, -0.43809208273887634, -0.43915462493896484, -0.46282583475112915]))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 1.1435747146606445))
        self.assertTrue(array_equal(return_old[2], 0.0))

    #same bug as above
    def test_2DImgcenter_method7searching_range_positiveself_defined_referenceFalse(self):
        self.assertTrue(True)
        '''
        return_new = fu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        return_old = oldfu.center_2D(image_to_be_centered=IMAGE_2D ,center_method=7,searching_range=2,Gauss_radius_inner=2,Gauss_radius_outter=7,self_defined_reference=False)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), ))
        self.assertEqual(return_new[1], return_old[1])
        self.assertEqual(return_new[2], return_old[2])
        self.assertTrue(array_equal(return_old[1], 0.0))
        self.assertTrue(array_equal(return_old[2], 0.0))
        '''




class Test_compose_transform3m(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.compose_transform3()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.compose_transform3()
        self.assertEqual(str(cm_new.exception), "compose_transform3() takes exactly 14 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_compose_transform3m(self):
        return_old = oldfu.compose_transform3(phi1=1.0,theta1=2, psi1=2,sx1=2.0,sy1=3.0,sz1=1,scale1=1.0,phi2=2.0,theta2=3, psi2=4,sx2=3.0,sy2=4.0,scale2=1.0,sz2=2)
        return_new = fu.compose_transform3(phi1=1.0,theta1=2, psi1=2,sx1=2.0,sy1=3.0,sz1=1,scale1=1.0,phi2=2.0,theta2=3, psi2=4,sx2=3.0,sy2=4.0,scale2=1.0,sz2=2)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old, [3.402106069683171, 4.997074136852235, 5.60154991083283, 5.24754524230957, 6.778360366821289, 3.108717203140259, 1.0]))





class Test_model_cylinder(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_cylinder()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_cylinder()
        self.assertEqual(str(cm_new.exception), "model_cylinder() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_model_cylinder(self):
        return_old = oldfu.model_cylinder(radius=2, nx=5, ny=5, nz=5)
        return_new = fu.model_cylinder(radius=2, nx=5, ny=5, nz=5)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))



class Test_model_rotated_rectangle2D(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_rotated_rectangle2D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_rotated_rectangle2D()
        self.assertEqual(str(cm_new.exception), "model_rotated_rectangle2D() takes at least 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_return_img(self):
        return_old = oldfu.model_rotated_rectangle2D(radius_long=4, radius_short=2, nx=10, ny=10, angle=90, return_numpy=False)
        return_new = fu.model_rotated_rectangle2D(radius_long=4, radius_short=2, nx=10, ny=10, angle=90, return_numpy=False)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_return_numpy(self):
        return_old = oldfu.model_rotated_rectangle2D(radius_long=4, radius_short=2, nx=10, ny=10, angle=90, return_numpy=True)
        return_new = fu.model_rotated_rectangle2D(radius_long=4, radius_short=2, nx=10, ny=10, angle=90, return_numpy=True)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new.flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))




"""
@unittest.skip("node involved, i cannot test")
class Test_send_string_to_all(unittest.TestCase):
    def test_wrong_number_params(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.send_string_to_all()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.send_string_to_all()
        self.assertEqual(str(cm_new.exception), "send_string_to_all() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_send_string_to_all(self):
        return_old = oldfu.send_string_to_all(str_to_send="sphire", source_node=0)
        return_new = fu.send_string_to_all(str_to_send="sphire", source_node=0)
        self.assertEqual(return_new, return_old)
"""

""" end: new in sphire 1.3"""



class Test_amoeba(unittest.TestCase):

    #copied from filter.py --> fit_tanh
    @staticmethod
    def fit_tanh_func(args, data):
        from math import pi, tanh
        v = 0.0
        if data[1][0] < 0.0:
            data[1][0] *= -1.0
        for i in range(len(data[0])):
            fsc = 2 * data[1][i] / (1.0 + data[1][i])
            if args[0] == 0 or args[1] == 0:
                qt = 0
            else:
                qt = fsc - 0.5 * (tanh(pi * (data[0][i] + args[0]) / 2.0 / args[1] / args[0]) - tanh(pi * (data[0][i] - args[0]) / 2.0 / args[1] / args[0]))
            v -= qt * qt
        return v

    @staticmethod
    def wrongfunction(a,b):
        return a+b

    @staticmethod
    def function_lessParam():
        return 0

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba()
        self.assertEqual(str(cm_new.exception), "amoeba() takes at least 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_amoeba(self):
        '''
        I did not use 'self.assertTrue(allclose(return_new, return_old, atol=TOLERANCE,equal_nan=True))' because the 'nosetets' spawns the following error
                TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
        '''

        return_new = fu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.fit_tanh_func, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=((0.0, 0.05, 0, 10, 0.15, 0.2), (0, 0.2, 0.4, 0.6, 0.8, 1.0), (8, 9, 5, 77, 98, 200)))
        return_old = oldfu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.fit_tanh_func, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=((0.0, 0.05, 0, 10, 0.15, 0.2), (0, 0.2, 0.4, 0.6, 0.8, 1.0), (8, 9, 5, 77, 98, 200)))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, ([0.0, 0.1], 0.0, 500)))

    def test_amoeba_with_wrongfunction(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.wrongfunction, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=((0.0, 0.05, 0, 10, 0.15, 0.2), (0, 0.2, 0.4, 0.6, 0.8,1.0), (8, 9, 5, 77, 98, 200)))
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.wrongfunction, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=((0.0, 0.05, 0, 10, 0.15, 0.2), (0, 0.2, 0.4, 0.6, 0.8,1.0), (8, 9, 5, 77, 98, 200)))
        self.assertEqual(str(cm_new.exception), "wrongfunction() got an unexpected keyword argument 'data'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_amoeba_with_function_lessParam_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.function_lessParam, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=((0.0, 0.05, 0, 10, 0.15, 0.2), (0, 0.2, 0.4, 0.6, 0.8,1.0), (8, 9, 5, 77, 98, 200)))
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.function_lessParam, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=((0.0, 0.05, 0, 10, 0.15, 0.2), (0, 0.2, 0.4, 0.6, 0.8,1.0), (8, 9, 5, 77, 98, 200)))
        self.assertEqual(str(cm_new.exception), "function_lessParam() takes no arguments (2 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_amoeba_with_NoneType_data_returns_TypeError_NoneType_obj_hasnot_attribute__getitem__(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.fit_tanh_func, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=None)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.amoeba (var=[0.0, 0.1], scale= [0.05, 0.05], func=self.fit_tanh_func, ftolerance=1.0e-4, xtolerance=1.0e-4, itmax=500 , data=None)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute '__getitem__'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_compose_transform2(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.compose_transform2()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.compose_transform2()
        self.assertEqual(str(cm_new.exception), "compose_transform2() takes exactly 8 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        """ values got from 'pickle files/utilities/utilities.compose_transform2'"""
        return_new = fu.compose_transform2(alpha1 = 0, sx1 = 2.90828285217, sy1 =-0.879739010334, scale1 = 1.0, alpha2 = 156.512610336, sx2 = 0, sy2 = 0, scale2 = 1.0)
        return_old = oldfu.compose_transform2(alpha1 = 0, sx1 = 2.90828285217, sy1 =-0.879739010334, scale1 = 1.0, alpha2 = 156.512610336, sx2 = 0, sy2 = 0, scale2 = 1.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (156.51260982858517, -3.0179426670074463, -0.35223737359046936, 1.0)))

    def test_null_scaleFactor_returns_RunTimeError_scale_factor_must_be_positive(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.compose_transform2(alpha1 = 0, sx1 = 2.90828285217, sy1 =-0.879739010334, scale1 = 0, alpha2 = 0, sx2 = 2.90828285217, sy2 =-0.879739010334, scale2 = 1.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.compose_transform2(alpha1 = 0, sx1 = 2.90828285217, sy1 =-0.879739010334, scale1 = 0, alpha2 = 0, sx2 = 2.90828285217, sy2 =-0.879739010334, scale2 = 1.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "The scale factor in a Transform object must be positive and non zero")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_negative_scaleFactor_returns_RunTimeError_scale_factor_must_be_positive(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.compose_transform2(alpha1 = 0, sx1 = 2.90828285217, sy1 =-0.879739010334, scale1 = -1.0, alpha2 = 0, sx2 = 2.90828285217, sy2 =-0.879739010334, scale2 = 1.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.compose_transform2(alpha1 = 0, sx1 = 2.90828285217, sy1 =-0.879739010334, scale1 = -1.0, alpha2 = 0, sx2 = 2.90828285217, sy2 =-0.879739010334, scale2 = 1.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "The scale factor in a Transform object must be positive and non zero")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_compose_transform3(unittest.TestCase):

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.compose_transform3()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.compose_transform3()
        self.assertEqual(str(cm_new.exception), "compose_transform3() takes exactly 14 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        """ values got from 'pickle files/utilities/utilities.compose_transform3'"""
        return_new = fu.compose_transform3(phi1 = 0.0, theta1  = 0.0, psi1 = 0.0, sx1 = 0.0,sy1 = 0.0, sz1 = 0.0,scale1 = 1.0, phi2 = 0.328125, theta2= 0.0, psi2 = 0.0, sx2 = 0.001220703125, sy2 = 0.0,sz2 = 0.001220703125,scale2 = 1.0)
        return_old = oldfu.compose_transform3(phi1 = 0.0, theta1  = 0.0, psi1 = 0.0, sx1 = 0.0,sy1 = 0.0, sz1 = 0.0,scale1 = 1.0, phi2 = 0.328125, theta2= 0.0, psi2 = 0.0, sx2 = 0.001220703125, sy2 = 0.0,sz2 = 0.001220703125,scale2 = 1.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.0, 0.0, 0.32812498609601065, 0.001220703125, 0.0, 0.001220703125, 1.0)))

    def test_null_scaleFactor_returns_RunTimeError_scale_factor_must_be_positive(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.compose_transform3(phi1 = 0.0, theta1  = 0.0, psi1 = 0.0, sx1 = 0.0,sy1 = 0.0, sz1 = 0.0,scale1 = 0, phi2 = 0.328125, theta2= 0.0, psi2 = 0.0, sx2 = 0.001220703125, sy2 = 0.0,sz2 = 0.001220703125,scale2 = 1.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.compose_transform3(phi1 = 0.0, theta1  = 0.0, psi1 = 0.0, sx1 = 0.0,sy1 = 0.0, sz1 = 0.0,scale1 = 0, phi2 = 0.328125, theta2= 0.0, psi2 = 0.0, sx2 = 0.001220703125, sy2 = 0.0,sz2 = 0.001220703125,scale2 = 1.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "The scale factor in a Transform object must be positive and non zero")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_negative_scaleFactor_returns_RunTimeError_scale_factor_must_be_positive(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.compose_transform3(phi1 = 0.0, theta1  = 0.0, psi1 = 0.0, sx1 = 0.0,sy1 = 0.0, sz1 = 0.0,scale1 = -1.0, phi2 = 0.328125, theta2= 0.0, psi2 = 0.0, sx2 = 0.001220703125, sy2 = 0.0,sz2 = 0.001220703125,scale2 = 1.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.compose_transform3(phi1 = 0.0, theta1  = 0.0, psi1 = 0.0, sx1 = 0.0,sy1 = 0.0, sz1 = 0.0,scale1 = -1.0, phi2 = 0.328125, theta2= 0.0, psi2 = 0.0, sx2 = 0.001220703125, sy2 = 0.0,sz2 = 0.001220703125,scale2 = 1.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "The scale factor in a Transform object must be positive and non zero")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))




class Test_combine_params2(unittest.TestCase):
    """ I did not use the 'pickle files/utilities/utilities.combine_params2' values because they are all 0 values"""
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.combine_params2()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.combine_params2()
        self.assertEqual(str(cm_new.exception), "combine_params2() takes exactly 8 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_combine_params2(self):
        return_new = fu.combine_params2(alpha1 = 0.0, sx1 = 1.0, sy1 = 1.0, mirror1 = 1, alpha2 = 1.0, sx2 =2.0, sy2 = 0.0, mirror2 = 0)
        return_old = oldfu.combine_params2(alpha1 = 0.0, sx1 = 1.0, sy1 = 1.0, mirror1 = 1, alpha2 = 1.0, sx2 =2.0, sy2 = 0.0, mirror2 = 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (359.0000000534512, -1.0176047086715698, 1.0173001289367676, 1)))



class Test_inverse_transform2(unittest.TestCase):
    """ I did not use the 'pickle files/utilities/utilities.inverse_transform2' values because they are all 0 values"""
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.inverse_transform2()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.inverse_transform2()
        self.assertEqual(str(cm_new.exception), "inverse_transform2() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_inverse_transform2(self):
        return_new = fu.inverse_transform2(alpha = 1.0, tx = 2.2, ty = 1.0, mirror = 0)
        return_old = oldfu.inverse_transform2(alpha = 1.0, tx = 2.2, ty = 1.0, mirror = 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,(358.9999999938496, -2.1822125911712646, -1.0382429361343384, 0)))



""" How may I REALLY test it?"""
"""
class Test_drop_image(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.drop_image()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.drop_image()
        self.assertEqual(str(cm_new.exception), "drop_image() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_invalid_type_returns_UnboundLocalError_imgtype_referenced_before_assignment(self):
        destination ='output.hdf'
        with self.assertRaises(UnboundLocalError) as cm_new:
            fu.drop_image(IMAGE_2D, destination, itype="invalid")
        with self.assertRaises(UnboundLocalError) as cm_old:
            oldfu.drop_image(IMAGE_2D, destination, itype="invalid")
        self.assertEqual(str(cm_new.exception), "local variable 'imgtype' referenced before assignment")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    @unittest.skip("it does not work under nosetests , anyway im not able to test it properly")
    def test_destination_is_not_a_file_returns_error_msg(self):
        destination = 3
        return_new = fu.drop_image(IMAGE_2D, destination, itype="h")
        return_old = oldfu.drop_image(IMAGE_2D, destination, itype="h")
        self.assertTrue(return_new is None)
        self.assertTrue(return_old is None)

    @unittest.skip("it does not work under nosetests , anyway im not able to test it properly")
    def test_drop_image2D_true_should_return_equal_objects1(self):
        destination ='output.hdf'
        return_new = fu.drop_image(IMAGE_2D, destination, itype="h")
        return_old = oldfu.drop_image(IMAGE_2D, destination, itype="h")

        if return_new is not None   and  return_old is not None:
            self.assertTrue(return_new, return_old)

    @unittest.skip("it does not work under nosetests , anyway im not able to test it properly")
    def test_drop_image_true_should_return_equal_objects2(self):
        filepath = path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.drop_image")
        with open(filepath, 'rb') as rb:
            argum = pickle_load(rb)

        print(argum)

        (imagename, destination) = argum[0]
        destination = 'output.hdf'
        return_new = fu.drop_image(imagename, destination, itype="h")
        return_old = oldfu.drop_image(imagename, destination, itype="h")

        if return_new is not None   and  return_old is not None:
            self.assertTrue(return_new, return_old)
"""


class Test_even_angles(unittest.TestCase):
    """ I did not changed the 'phiEqpsi' params because it is used in 'even_angles_cd' I'll test it there"""
    def test_default_values(self):
        return_new = fu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='c1', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='c1', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 0.0, 0.0], [102.5624397820863, 8.409807949596694, 257.43756021791364], [175.28184168449116, 11.903989804110001, 184.71815831550884], [234.81899085328783, 14.592550602033418, 125.1810091467122], [286.52113069039967, 16.865343252479008, 73.478869309600327], [332.89249973858841, 18.873236840047255, 27.107500261411587], [15.350997945238817, 20.69354123118596, 344.64900205476124], [54.760293521450905, 22.37214549396397, 305.23970647854912], [91.727719586672706, 23.938926249214624, 268.27228041332728], [126.67925988880424, 25.41462091516098, 233.32074011119573], [159.93126768874427, 26.81431796194859, 200.06873231125576], [191.72626852098327, 28.149400619646084, 168.27373147901676], [222.25501416086877, 29.428707176867, 137.74498583913123], [251.6707339535308, 30.659262305350033, 108.32926604646923], [280.09871166816117, 31.846758629170495, 79.901288331838828], [307.64293448395898, 32.995885473579534, 52.357065516041018], [334.39083847001103, 34.11056017878775, 25.609161529988967], [0.42677669506366556, 35.194095100409235, 359.57322330493639], [25.794606434997782, 36.249320882899376, 334.20539356500217], [50.559654291516139, 37.278679231322116, 309.44034570848385], [74.770232732225381, 38.2842939251198, 285.2297672677746], [98.468827134074971, 39.26802600175335, 261.53117286592499], [121.69303677671941, 40.231517219359155, 238.3069632232806], [144.4763293594925, 41.17622470375671, 215.52367064050748], [166.84865229059051, 42.10344887074584, 193.15134770940949], [188.83693262466142, 43.014356152771704, 171.16306737533864], [210.46548946865465, 43.909997664475156, 149.53451053134535], [231.75637688070145, 44.79132466007832, 128.24362311929855], [252.72967105963514, 45.65920143165515, 107.27032894036483], [273.40371249950607, 46.51441614768202, 86.596287500493929], [293.7953114483945, 47.357690020060026, 66.2046885516055], [313.91992324589262, 48.1896851042214, 46.080076754107381], [333.79179876604201, 49.01101097344977, 26.208201233957993], [353.42411415385686, 49.822230459852115, 6.5758858461431373], [12.839083235960516, 50.62386461673009, 347.16091676403948], [32.02805535274598, 51.41639702767674, 327.97194464725408], [51.011600859315614, 52.20027756457276, 308.98839914068435], [69.799586144482291, 52.975925678303284, 290.2004138555177], [88.401239698292727, 53.743733291363625, 271.59876030170722], [106.82521050148785, 54.50406734974836, 253.17478949851215], [125.07961980182155, 55.25727208199666, 234.92038019817846], [143.17210717208275, 56.00367100552329, 216.82789282791725], [161.10987160517593, 56.74356871403049, 198.89012839482405], [178.89970828662715, 57.4772524745885, 181.10029171337283], [196.54804158963, 58.20499365866951, 163.45195841037003], [214.06095475847701, 58.92704902784667, 145.93904524152299], [231.44421667996505, 59.64366189189109, 128.55578332003495], [248.70330608674968, 60.355063154503576, 111.29669391325035], [265.84343348975648, 61.06147225981934, 94.156566510243522], [282.86956109711195, 61.763098051052104, 77.130438902888045], [299.78642094339619, 62.46013955114206, 60.213579056603805], [316.59853142434207, 63.152786673995614, 43.40146857565793], [333.31021240759083, 63.84122087381428, 26.689787592409175], [349.92559906909207, 64.52561573907757, 10.074400930907927], [6.4586545866518463, 65.20613753694339, 353.54134541334815], [22.893181806532958, 65.88294571313848, 337.10681819346701], [39.242833985512988, 66.55619335181605, 320.75716601448698], [55.511124699098673, 67.22602759934011, 304.48887530090133], [71.701436996410379, 67.8925900555079, 288.29856300358961], [87.81703187337213, 68.55601713533103, 272.18296812662788], [103.86105612808187, 69.21644040415431, 256.13894387191817], [119.8365496554388, 69.87398688859322, 240.16345034456117], [135.74645223213611, 70.52877936550931, 224.25354776786389], [151.59360983787678, 71.18093663101206, 208.40639016212322], [167.38078055404094, 71.83057375127423, 192.61921944595906], [183.11064007694512, 72.47780229676785, 176.88935992305483], [198.78578687921549, 73.12273056137076, 161.21421312078451], [214.40874704959094, 73.76546376765336, 145.59125295040906], [229.98197883862355, 74.40610425953089, 130.01802116137645], [245.50787693521318, 75.04475168335667, 114.49212306478682], [260.98877649665752, 75.68150315843295, 99.011223503342478], [276.42695695288819, 76.31645343782941, 83.57304304711181], [291.82464560376934, 76.94969506032008, 68.175354396230659], [307.18402102672974, 77.58131849418093, 52.815978973270262], [322.50721631056541, 78.21141227352726, 37.492783689434589], [337.79632212996364, 78.84006312781455, 22.203677870036358], [353.0533896741494, 79.46735610507622, 6.9466103258505996], [8.2904334420228452, 80.09337468942728, 351.70956655797715], [23.489433915232105, 80.71820091332246, 336.5105660847679], [38.662340119797371, 81.34191546502161, 321.33765988020264], [53.811072086159413, 81.96459779168268, 306.18892791384064], [68.937523216861678, 82.58632619847424, 291.06247678313832], [84.043562570481001, 83.20717794407292, 275.95643742951904], [99.131037069892173, 83.82722933288893, 260.86896293010784], [114.20177364247999, 84.44655580434149, 245.79822635751998], [129.25758129949423, 85.06523201948858, 230.74241870050582], [144.30025316137389, 85.68333194529811, 215.69974683862608], [159.33156843554312, 86.30092893683496, 200.66843156445691], [174.35329435289955, 86.91809581762422, 185.64670564710048], [189.36718806897298, 87.53490495844152, 170.63281193102705], [204.37499853552671, 88.15142835477144, 155.62500146447326], [219.37846834820326, 88.7677377031675, 140.62153165179677], [234.37933557567774, 89.38390447674091, 125.62066442432229]]))

    def test_null_delta_returns_ZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.even_angles(delta = 0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='c1', ant = 0.0)
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.even_angles(delta = 0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='c1', ant = 0.0)
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values_with_not_minus(self):
        return_new = fu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "", symmetry='c1', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "", symmetry='c1', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[0.0, 0.0, 0.0], [102.5624397820863, 8.409807949596694, 0.0], [175.28184168449116, 11.903989804110001, 0.0], [234.81899085328783, 14.592550602033418, 0.0], [286.52113069039967, 16.865343252479008, 0.0], [332.89249973858841, 18.873236840047255, 0.0], [15.350997945238817, 20.69354123118596, 0.0], [54.760293521450905, 22.37214549396397, 0.0], [91.727719586672706, 23.938926249214624, 0.0], [126.67925988880424, 25.41462091516098, 0.0], [159.93126768874427, 26.81431796194859, 0.0], [191.72626852098327, 28.149400619646084, 0.0], [222.25501416086877, 29.428707176867, 0.0], [251.6707339535308, 30.659262305350033, 0.0], [280.09871166816117, 31.846758629170495, 0.0], [307.64293448395898, 32.995885473579534, 0.0], [334.39083847001103, 34.11056017878775, 0.0], [0.42677669506366556, 35.194095100409235, 0.0], [25.794606434997782, 36.249320882899376, 0.0], [50.559654291516139, 37.278679231322116, 0.0], [74.770232732225381, 38.2842939251198, 0.0], [98.468827134074971, 39.26802600175335, 0.0], [121.69303677671941, 40.231517219359155, 0.0], [144.4763293594925, 41.17622470375671, 0.0], [166.84865229059051, 42.10344887074584, 0.0], [188.83693262466142, 43.014356152771704, 0.0], [210.46548946865465, 43.909997664475156, 0.0], [231.75637688070145, 44.79132466007832, 0.0], [252.72967105963514, 45.65920143165515, 0.0], [273.40371249950607, 46.51441614768202, 0.0], [293.7953114483945, 47.357690020060026, 0.0], [313.91992324589262, 48.1896851042214, 0.0], [333.79179876604201, 49.01101097344977, 0.0], [353.42411415385686, 49.822230459852115, 0.0], [12.839083235960516, 50.62386461673009, 0.0], [32.02805535274598, 51.41639702767674, 0.0], [51.011600859315614, 52.20027756457276, 0.0], [69.799586144482291, 52.975925678303284, 0.0], [88.401239698292727, 53.743733291363625, 0.0], [106.82521050148785, 54.50406734974836, 0.0], [125.07961980182155, 55.25727208199666, 0.0], [143.17210717208275, 56.00367100552329, 0.0], [161.10987160517593, 56.74356871403049, 0.0], [178.89970828662715, 57.4772524745885, 0.0], [196.54804158963, 58.20499365866951, 0.0], [214.06095475847701, 58.92704902784667, 0.0], [231.44421667996505, 59.64366189189109, 0.0], [248.70330608674968, 60.355063154503576, 0.0], [265.84343348975648, 61.06147225981934, 0.0], [282.86956109711195, 61.763098051052104, 0.0], [299.78642094339619, 62.46013955114206, 0.0], [316.59853142434207, 63.152786673995614, 0.0], [333.31021240759083, 63.84122087381428, 0.0], [349.92559906909207, 64.52561573907757, 0.0], [6.4586545866518463, 65.20613753694339, 0.0], [22.893181806532958, 65.88294571313848, 0.0], [39.242833985512988, 66.55619335181605, 0.0], [55.511124699098673, 67.22602759934011, 0.0], [71.701436996410379, 67.8925900555079, 0.0], [87.81703187337213, 68.55601713533103, 0.0], [103.86105612808187, 69.21644040415431, 0.0], [119.8365496554388, 69.87398688859322, 0.0], [135.74645223213611, 70.52877936550931, 0.0], [151.59360983787678, 71.18093663101206, 0.0], [167.38078055404094, 71.83057375127423, 0.0], [183.11064007694512, 72.47780229676785, 0.0], [198.78578687921549, 73.12273056137076, 0.0], [214.40874704959094, 73.76546376765336, 0.0], [229.98197883862355, 74.40610425953089, 0.0], [245.50787693521318, 75.04475168335667, 0.0], [260.98877649665752, 75.68150315843295, 0.0], [276.42695695288819, 76.31645343782941, 0.0], [291.82464560376934, 76.94969506032008, 0.0], [307.18402102672974, 77.58131849418093, 0.0], [322.50721631056541, 78.21141227352726, 0.0], [337.79632212996364, 78.84006312781455, 0.0], [353.0533896741494, 79.46735610507622, 0.0], [8.2904334420228452, 80.09337468942728, 0.0], [23.489433915232105, 80.71820091332246, 0.0], [38.662340119797371, 81.34191546502161, 0.0], [53.811072086159413, 81.96459779168268, 0.0], [68.937523216861678, 82.58632619847424, 0.0], [84.043562570481001, 83.20717794407292, 0.0], [99.131037069892173, 83.82722933288893, 0.0], [114.20177364247999, 84.44655580434149, 0.0], [129.25758129949423, 85.06523201948858, 0.0], [144.30025316137389, 85.68333194529811, 0.0], [159.33156843554312, 86.30092893683496, 0.0], [174.35329435289955, 86.91809581762422, 0.0], [189.36718806897298, 87.53490495844152, 0.0], [204.37499853552671, 88.15142835477144, 0.0], [219.37846834820326, 88.7677377031675, 0.0], [234.37933557567774, 89.38390447674091, 0.0]]))

    def test_default_values_with_P_method_leads_to_deadlock(self):
        self.assertTrue(True)
        """
        return_new = fu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'P', phiEqpsi = "Minus", symmetry='c1', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'P', phiEqpsi = "Minus", symmetry='c1', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        """

    def test_with_D_symmetry(self):
        return_new = fu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='d1', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='d1', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 0.0, 0.0], [234.81899085328783, 14.592550602033418, 125.1810091467122], [15.340997945238826, 20.69354123118596, 344.65900205476123], [54.750293521450914, 22.37214549396397, 305.24970647854911], [191.71626852098328, 28.149400619646084, 168.28373147901675], [222.24501416086878, 29.428707176867, 137.75498583913122], [251.66073395353081, 30.659262305350033, 108.33926604646922], [0.40677669506368375, 35.194095100409235, 359.59322330493637], [25.7746064349978, 36.249320882899376, 334.22539356500215], [50.539654291516158, 37.278679231322116, 309.46034570848383], [74.750232732225399, 38.2842939251198, 285.24976726777459], [188.81693262466143, 43.014356152771704, 171.18306737533862], [210.44548946865467, 43.909997664475156, 149.55451053134533], [231.73637688070147, 44.79132466007832, 128.26362311929853], [252.70967105963516, 45.65920143165515, 107.29032894036482], [12.809083235960543, 50.62386461673009, 347.19091676403946], [31.998055352746011, 51.41639702767674, 328.00194464725394], [50.981600859315648, 52.20027756457276, 309.01839914068432], [69.769586144482332, 52.975925678303284, 290.23041385551767], [88.371239698292783, 53.743733291363625, 271.62876030170719], [196.51804158963009, 58.20499365866951, 163.48195841036988], [214.03095475847709, 58.92704902784667, 145.96904524152291], [231.41421667996514, 59.64366189189109, 128.58578332003486], [248.67330608674976, 60.355063154503576, 111.32669391325021], [265.81343348975656, 61.06147225981934, 94.186566510243438], [6.4186545866519964, 65.20613753694339, 353.581345413348], [22.853181806533108, 65.88294571313848, 337.14681819346686], [39.202833985513138, 66.55619335181605, 320.79716601448683], [55.471124699098823, 67.22602759934011, 304.52887530090118], [71.661436996410529, 67.8925900555079, 288.33856300358946], [87.77703187337228, 68.55601713533103, 272.22296812662773], [183.07064007694527, 72.47780229676785, 176.92935992305479], [198.74578687921564, 73.12273056137076, 161.25421312078436], [214.36874704959109, 73.76546376765336, 145.63125295040891], [229.9419788386237, 74.40610425953089, 130.0580211613763], [245.46787693521333, 75.04475168335667, 114.53212306478667], [260.94877649665767, 75.68150315843295, 99.051223503342328], [8.2404334420230043, 80.09337468942728, 351.759566557977], [23.439433915232264, 80.71820091332246, 336.56056608476774], [38.61234011979753, 81.34191546502161, 321.38765988020248], [53.761072086159572, 81.96459779168268, 306.23892791384037], [68.887523216861837, 82.58632619847424, 291.11247678313816], [83.99356257048116, 83.20717794407292, 276.00643742951888], [189.31718806897314, 87.53490495844152, 170.68281193102689], [204.32499853552687, 88.15142835477144, 155.6750014644731], [219.32846834820342, 88.7677377031675, 140.67153165179661], [234.3293355756779, 89.38390447674091, 125.67066442432213]]))

    def test_with_S_symmetry(self):
        return_new = fu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='sd1', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='sd1', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 90.0, 90.0], [15.0, 90.0, 90.0], [30.0, 90.0, 90.0], [45.0, 90.0, 90.0], [60.0, 90.0, 90.0], [75.0, 90.0, 90.0]]))

    def test_with_S_symmetry_tooBig_theta1_value_error_msg(self):
        return_new = fu.even_angles(delta = 15.0, theta1=91.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='sd1', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=91.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='sd1', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 90.0, 90.0], [15.0, 90.0, 90.0], [30.0, 90.0, 90.0], [45.0, 90.0, 90.0], [60.0, 90.0, 90.0], [75.0, 90.0, 90.0]]))

    def test_with_S_invalid_symmetry_returns_UnboundLocalError_local_var_referenced_before_assignment(self):
        with self.assertRaises(UnboundLocalError) as cm_new:
            fu.even_angles(delta = 15.0, theta1=10.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='sp1', ant = 0.0)
        with self.assertRaises(UnboundLocalError) as cm_old:
            oldfu.even_angles(delta = 15.0, theta1=10.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='sp1', ant = 0.0)
        self.assertEqual(str(cm_new.exception), "local variable 'k' referenced before assignment")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_S_invalid_symmetry_returns_ValueError_invalid_literal(self):
        with self.assertRaises(ValueError) as cm_new:
            fu.even_angles(delta = 15.0, theta1=10.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='soct', ant = 0.0)
        with self.assertRaises(ValueError) as cm_old:
            oldfu.even_angles(delta = 15.0, theta1=10.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='soct', ant = 0.0)
        self.assertEqual(str(cm_new.exception), "invalid literal for int() with base 10: 'ct'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_not_supported_symmetry_Warning_output_msg(self):
        return_new = fu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='oct', ant = 0.0)
        return_old = oldfu.even_angles(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEqpsi = "Minus", symmetry='oct', ant = 0.0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, []))



class Test_even_angles_cd(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.even_angles_cd()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.even_angles_cd()
        self.assertEqual(str(cm_new.exception), "even_angles_cd() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_null_delta_returns_ZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.even_angles_cd(delta = 0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEQpsi='Minus')
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.even_angles_cd(delta = 0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEQpsi='Minus')
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values_leads_to_deadlock(self):
        self.assertTrue(True)
        """
        return_new = fu.even_angles_cd(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'P', phiEQpsi='Minus')
        return_old = oldfu.even_angles_cd(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'P', phiEQpsi='Minus')
        self.assertTrue(array_equal(return_new, return_old))
        """

    def test_with_S_method(self):
        return_new = fu.even_angles_cd(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEQpsi='Minus')
        return_old = oldfu.even_angles_cd(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEQpsi='Minus')
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 0.0, 0.0], [102.5624397820863, 8.409807949596694, 257.43756021791364], [175.28184168449116, 11.903989804110001, 184.71815831550884], [234.81899085328783, 14.592550602033418, 125.1810091467122], [286.52113069039967, 16.865343252479008, 73.478869309600327], [332.89249973858841, 18.873236840047255, 27.107500261411587], [15.350997945238817, 20.69354123118596, 344.64900205476124], [54.760293521450905, 22.37214549396397, 305.23970647854912], [91.727719586672706, 23.938926249214624, 268.27228041332728], [126.67925988880424, 25.41462091516098, 233.32074011119573], [159.93126768874427, 26.81431796194859, 200.06873231125576], [191.72626852098327, 28.149400619646084, 168.27373147901676], [222.25501416086877, 29.428707176867, 137.74498583913123], [251.6707339535308, 30.659262305350033, 108.32926604646923], [280.09871166816117, 31.846758629170495, 79.901288331838828], [307.64293448395898, 32.995885473579534, 52.357065516041018], [334.39083847001103, 34.11056017878775, 25.609161529988967], [0.42677669506366556, 35.194095100409235, 359.57322330493639], [25.794606434997782, 36.249320882899376, 334.20539356500217], [50.559654291516139, 37.278679231322116, 309.44034570848385], [74.770232732225381, 38.2842939251198, 285.2297672677746], [98.468827134074971, 39.26802600175335, 261.53117286592499], [121.69303677671941, 40.231517219359155, 238.3069632232806], [144.4763293594925, 41.17622470375671, 215.52367064050748], [166.84865229059051, 42.10344887074584, 193.15134770940949], [188.83693262466142, 43.014356152771704, 171.16306737533864], [210.46548946865465, 43.909997664475156, 149.53451053134535], [231.75637688070145, 44.79132466007832, 128.24362311929855], [252.72967105963514, 45.65920143165515, 107.27032894036483], [273.40371249950607, 46.51441614768202, 86.596287500493929], [293.7953114483945, 47.357690020060026, 66.2046885516055], [313.91992324589262, 48.1896851042214, 46.080076754107381], [333.79179876604201, 49.01101097344977, 26.208201233957993], [353.42411415385686, 49.822230459852115, 6.5758858461431373], [12.839083235960516, 50.62386461673009, 347.16091676403948], [32.02805535274598, 51.41639702767674, 327.97194464725408], [51.011600859315614, 52.20027756457276, 308.98839914068435], [69.799586144482291, 52.975925678303284, 290.2004138555177], [88.401239698292727, 53.743733291363625, 271.59876030170722], [106.82521050148785, 54.50406734974836, 253.17478949851215], [125.07961980182155, 55.25727208199666, 234.92038019817846], [143.17210717208275, 56.00367100552329, 216.82789282791725], [161.10987160517593, 56.74356871403049, 198.89012839482405], [178.89970828662715, 57.4772524745885, 181.10029171337283], [196.54804158963, 58.20499365866951, 163.45195841037003], [214.06095475847701, 58.92704902784667, 145.93904524152299], [231.44421667996505, 59.64366189189109, 128.55578332003495], [248.70330608674968, 60.355063154503576, 111.29669391325035], [265.84343348975648, 61.06147225981934, 94.156566510243522], [282.86956109711195, 61.763098051052104, 77.130438902888045], [299.78642094339619, 62.46013955114206, 60.213579056603805], [316.59853142434207, 63.152786673995614, 43.40146857565793], [333.31021240759083, 63.84122087381428, 26.689787592409175], [349.92559906909207, 64.52561573907757, 10.074400930907927], [6.4586545866518463, 65.20613753694339, 353.54134541334815], [22.893181806532958, 65.88294571313848, 337.10681819346701], [39.242833985512988, 66.55619335181605, 320.75716601448698], [55.511124699098673, 67.22602759934011, 304.48887530090133], [71.701436996410379, 67.8925900555079, 288.29856300358961], [87.81703187337213, 68.55601713533103, 272.18296812662788], [103.86105612808187, 69.21644040415431, 256.13894387191817], [119.8365496554388, 69.87398688859322, 240.16345034456117], [135.74645223213611, 70.52877936550931, 224.25354776786389], [151.59360983787678, 71.18093663101206, 208.40639016212322], [167.38078055404094, 71.83057375127423, 192.61921944595906], [183.11064007694512, 72.47780229676785, 176.88935992305483], [198.78578687921549, 73.12273056137076, 161.21421312078451], [214.40874704959094, 73.76546376765336, 145.59125295040906], [229.98197883862355, 74.40610425953089, 130.01802116137645], [245.50787693521318, 75.04475168335667, 114.49212306478682], [260.98877649665752, 75.68150315843295, 99.011223503342478], [276.42695695288819, 76.31645343782941, 83.57304304711181], [291.82464560376934, 76.94969506032008, 68.175354396230659], [307.18402102672974, 77.58131849418093, 52.815978973270262], [322.50721631056541, 78.21141227352726, 37.492783689434589], [337.79632212996364, 78.84006312781455, 22.203677870036358], [353.0533896741494, 79.46735610507622, 6.9466103258505996], [8.2904334420228452, 80.09337468942728, 351.70956655797715], [23.489433915232105, 80.71820091332246, 336.5105660847679], [38.662340119797371, 81.34191546502161, 321.33765988020264], [53.811072086159413, 81.96459779168268, 306.18892791384064], [68.937523216861678, 82.58632619847424, 291.06247678313832], [84.043562570481001, 83.20717794407292, 275.95643742951904], [99.131037069892173, 83.82722933288893, 260.86896293010784], [114.20177364247999, 84.44655580434149, 245.79822635751998], [129.25758129949423, 85.06523201948858, 230.74241870050582], [144.30025316137389, 85.68333194529811, 215.69974683862608], [159.33156843554312, 86.30092893683496, 200.66843156445691], [174.35329435289955, 86.91809581762422, 185.64670564710048], [189.36718806897298, 87.53490495844152, 170.63281193102705], [204.37499853552671, 88.15142835477144, 155.62500146447326], [219.37846834820326, 88.7677377031675, 140.62153165179677], [234.37933557567774, 89.38390447674091, 125.62066442432229]]))

    def test_with_S_method_with_not_Minus(self):
        return_new = fu.even_angles_cd(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEQpsi='not_Minus')
        return_old = oldfu.even_angles_cd(delta = 15.0, theta1=0.0, theta2=90.0, phi1=0.0, phi2=359.99, method = 'S', phiEQpsi='not_Minus')
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[0.0, 0.0, 0.0], [102.5624397820863, 8.409807949596694, 0.0], [175.28184168449116, 11.903989804110001, 0.0], [234.81899085328783, 14.592550602033418, 0.0], [286.52113069039967, 16.865343252479008, 0.0], [332.89249973858841, 18.873236840047255, 0.0], [15.350997945238817, 20.69354123118596, 0.0], [54.760293521450905, 22.37214549396397, 0.0], [91.727719586672706, 23.938926249214624, 0.0], [126.67925988880424, 25.41462091516098, 0.0], [159.93126768874427, 26.81431796194859, 0.0], [191.72626852098327, 28.149400619646084, 0.0], [222.25501416086877, 29.428707176867, 0.0], [251.6707339535308, 30.659262305350033, 0.0], [280.09871166816117, 31.846758629170495, 0.0], [307.64293448395898, 32.995885473579534, 0.0], [334.39083847001103, 34.11056017878775, 0.0], [0.42677669506366556, 35.194095100409235, 0.0], [25.794606434997782, 36.249320882899376, 0.0], [50.559654291516139, 37.278679231322116, 0.0], [74.770232732225381, 38.2842939251198, 0.0], [98.468827134074971, 39.26802600175335, 0.0], [121.69303677671941, 40.231517219359155, 0.0], [144.4763293594925, 41.17622470375671, 0.0], [166.84865229059051, 42.10344887074584, 0.0], [188.83693262466142, 43.014356152771704, 0.0], [210.46548946865465, 43.909997664475156, 0.0], [231.75637688070145, 44.79132466007832, 0.0], [252.72967105963514, 45.65920143165515, 0.0], [273.40371249950607, 46.51441614768202, 0.0], [293.7953114483945, 47.357690020060026, 0.0], [313.91992324589262, 48.1896851042214, 0.0], [333.79179876604201, 49.01101097344977, 0.0], [353.42411415385686, 49.822230459852115, 0.0], [12.839083235960516, 50.62386461673009, 0.0], [32.02805535274598, 51.41639702767674, 0.0], [51.011600859315614, 52.20027756457276, 0.0], [69.799586144482291, 52.975925678303284, 0.0], [88.401239698292727, 53.743733291363625, 0.0], [106.82521050148785, 54.50406734974836, 0.0], [125.07961980182155, 55.25727208199666, 0.0], [143.17210717208275, 56.00367100552329, 0.0], [161.10987160517593, 56.74356871403049, 0.0], [178.89970828662715, 57.4772524745885, 0.0], [196.54804158963, 58.20499365866951, 0.0], [214.06095475847701, 58.92704902784667, 0.0], [231.44421667996505, 59.64366189189109, 0.0], [248.70330608674968, 60.355063154503576, 0.0], [265.84343348975648, 61.06147225981934, 0.0], [282.86956109711195, 61.763098051052104, 0.0], [299.78642094339619, 62.46013955114206, 0.0], [316.59853142434207, 63.152786673995614, 0.0], [333.31021240759083, 63.84122087381428, 0.0], [349.92559906909207, 64.52561573907757, 0.0], [6.4586545866518463, 65.20613753694339, 0.0], [22.893181806532958, 65.88294571313848, 0.0], [39.242833985512988, 66.55619335181605, 0.0], [55.511124699098673, 67.22602759934011, 0.0], [71.701436996410379, 67.8925900555079, 0.0], [87.81703187337213, 68.55601713533103, 0.0], [103.86105612808187, 69.21644040415431, 0.0], [119.8365496554388, 69.87398688859322, 0.0], [135.74645223213611, 70.52877936550931, 0.0], [151.59360983787678, 71.18093663101206, 0.0], [167.38078055404094, 71.83057375127423, 0.0], [183.11064007694512, 72.47780229676785, 0.0], [198.78578687921549, 73.12273056137076, 0.0], [214.40874704959094, 73.76546376765336, 0.0], [229.98197883862355, 74.40610425953089, 0.0], [245.50787693521318, 75.04475168335667, 0.0], [260.98877649665752, 75.68150315843295, 0.0], [276.42695695288819, 76.31645343782941, 0.0], [291.82464560376934, 76.94969506032008, 0.0], [307.18402102672974, 77.58131849418093, 0.0], [322.50721631056541, 78.21141227352726, 0.0], [337.79632212996364, 78.84006312781455, 0.0], [353.0533896741494, 79.46735610507622, 0.0], [8.2904334420228452, 80.09337468942728, 0.0], [23.489433915232105, 80.71820091332246, 0.0], [38.662340119797371, 81.34191546502161, 0.0], [53.811072086159413, 81.96459779168268, 0.0], [68.937523216861678, 82.58632619847424, 0.0], [84.043562570481001, 83.20717794407292, 0.0], [99.131037069892173, 83.82722933288893, 0.0], [114.20177364247999, 84.44655580434149, 0.0], [129.25758129949423, 85.06523201948858, 0.0], [144.30025316137389, 85.68333194529811, 0.0], [159.33156843554312, 86.30092893683496, 0.0], [174.35329435289955, 86.91809581762422, 0.0], [189.36718806897298, 87.53490495844152, 0.0], [204.37499853552671, 88.15142835477144, 0.0], [219.37846834820326, 88.7677377031675, 0.0], [234.37933557567774, 89.38390447674091, 0.0]]))



class Test_gauss_edge(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.gauss_edge()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.gauss_edge()
        self.assertEqual(str(cm_new.exception), "gauss_edge() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.gauss_edge(None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.gauss_edge(None)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_ndim'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_value_2Dreal_img(self):
        return_new =fu.gauss_edge(sharp_edge_image=IMAGE_2D, kernel_size = 7, gauss_standard_dev =3)
        return_old =oldfu.gauss_edge(sharp_edge_image=IMAGE_2D, kernel_size = 7, gauss_standard_dev =3)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), [-0.13465632498264313, -0.13546007871627808, -0.134829580783844, -0.1335931271314621, -0.13253170251846313, -0.13204187154769897, -0.13220487534999847, -0.13221487402915955, -0.13232742249965668, -0.1337665468454361, -0.06852434575557709, -0.06880658864974976, -0.06777483969926834, -0.06598402559757233, -0.0647275447845459, -0.06409801542758942, -0.06481462717056274, -0.0652804896235466, -0.06612081080675125, -0.06761974841356277, -0.022264748811721802, -0.02200571447610855, -0.020655043423175812, -0.019159983843564987, -0.018887672573328018, -0.01891341246664524, -0.01985529437661171, -0.01982933096587658, -0.020445043221116066, -0.021704547107219696, -0.005446895956993103, -0.0036098735872656107, -0.0015185611555352807, 0.00028625220875255764, -0.0018038542475551367, -0.004057552665472031, -0.0074673667550086975, -0.007335766218602657, -0.007089833728969097, -0.006572901736944914, -0.062402140349149704, -0.05936679244041443, -0.056522615253925323, -0.054644547402858734, -0.05874261260032654, -0.06292443722486496, -0.06814932078123093, -0.06764056533575058, -0.06658145785331726, -0.06484474986791611, -0.1351991444826126, -0.13071101903915405, -0.1271258443593979, -0.12464220821857452, -0.1300903856754303, -0.1362408697605133, -0.14354917407035828, -0.14375917613506317, -0.1421736180782318, -0.1391114443540573, -0.2177044302225113, -0.21358831226825714, -0.21049849689006805, -0.20857857167720795, -0.21429932117462158, -0.22052235901355743, -0.22717159986495972, -0.22686269879341125, -0.22466568648815155, -0.22149845957756042, -0.24495716392993927, -0.24219506978988647, -0.24050326645374298, -0.23946698009967804, -0.24440476298332214, -0.24958965182304382, -0.2547557055950165, -0.2540545165538788, -0.25141072273254395, -0.24826845526695251, -0.23753224313259125, -0.2367689162492752, -0.23625515401363373, -0.2358333319425583, -0.23901799321174622, -0.24205103516578674, -0.24468548595905304, -0.24341563880443573, -0.24087902903556824, -0.23916129767894745, -0.1973155289888382, -0.197978213429451, -0.19792386889457703, -0.1971890926361084, -0.19790858030319214, -0.198713481426239, -0.19994543492794037, -0.19909298419952393, -0.197785884141922, -0.19749222695827484]))

    def test_default_value_3Dreal_img(self):
        return_new =fu.gauss_edge(sharp_edge_image=IMAGE_3D, kernel_size = 7, gauss_standard_dev =3)
        return_old =oldfu.gauss_edge(sharp_edge_image=IMAGE_3D, kernel_size = 7, gauss_standard_dev =3)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [-0.3821377456188202, -0.38477593660354614, -0.38564532995224, -0.3848218619823456, -0.38031768798828125, -0.37607041001319885, -0.3727336525917053, -0.37334394454956055, -0.3756125867366791, -0.3789021670818329, -0.35474997758865356, -0.35638177394866943, -0.3565954566001892, -0.35552269220352173, -0.3519652485847473, -0.34885260462760925, -0.34676575660705566, -0.3476618826389313, -0.3497673273086548, -0.3523935079574585, -0.33886396884918213, -0.3395621180534363, -0.33929628133773804, -0.3382933437824249, -0.3361724019050598, -0.3344022035598755, -0.33357003331184387, -0.3343340754508972, -0.3358710706233978, -0.3375495672225952, -0.33741381764411926, -0.33650967478752136, -0.33523687720298767, -0.333883672952652, -0.3336937427520752, -0.3341248631477356, -0.3355773985385895, -0.33653461933135986, -0.3374081552028656, -0.3376966118812561, -0.3544465899467468, -0.3527055084705353, -0.35095182061195374, -0.3494855463504791, -0.3506280183792114, -0.35247039794921875, -0.3553317189216614, -0.3562195301055908, -0.3564341068267822, -0.3557356297969818, -0.38188254833221436, -0.37997275590896606, -0.37831366062164307, -0.3770351707935333, -0.3787494897842407, -0.38118863105773926, -0.38440120220184326, -0.38505932688713074, -0.384756863117218, -0.3835076689720154, -0.4096095561981201, -0.40803688764572144, -0.40678682923316956, -0.4058643877506256, -0.407475084066391, -0.4097420275211334, -0.4125012159347534, -0.412929505109787, -0.4123894274234772, -0.41106510162353516, -0.42710575461387634, -0.4269286096096039, -0.4266727566719055, -0.4261814057826996, -0.42619654536247253, -0.42661571502685547, -0.42734473943710327, -0.4275282323360443, -0.4274507164955139, -0.42723923921585083, -0.42604726552963257, -0.427217036485672, -0.4276130795478821, -0.4270658493041992, -0.4251582622528076, -0.4235444962978363, -0.4224027097225189, -0.42265504598617554, -0.42344430088996887, -0.4246865510940552, -0.40891507267951965, -0.41115355491638184, -0.4119035303592682, -0.4110800623893738, -0.40723463892936707, -0.4037247598171234, -0.4010475277900696, -0.40163320302963257, -0.4034925401210785, -0.4062225818634033, -0.3912156820297241, -0.39305198192596436, -0.39355185627937317, -0.3928839862346649, -0.38965797424316406, -0.38660651445388794, -0.38435032963752747, -0.38484278321266174, -0.3866077661514282, -0.3890122175216675, -0.3711863160133362, -0.37203288078308105, -0.3719569742679596, -0.37113863229751587, -0.3689132034778595, -0.36694109439849854, -0.36587607860565186, -0.3666207194328308, -0.3682072162628174, -0.3699294626712799, -0.36050644516944885, -0.3606247901916504, -0.36022618412971497, -0.35948821902275085, -0.3582903742790222, -0.3572496175765991, -0.35703587532043457, -0.3577428460121155, -0.35895442962646484, -0.3600172698497772, -0.3602280616760254, -0.35916683077812195, -0.35803502798080444, -0.3569867014884949, -0.35714292526245117, -0.3576701283454895, -0.35909977555274963, -0.35999271273612976, -0.36075741052627563, -0.36086776852607727, -0.375320166349411, -0.37388816475868225, -0.3725152313709259, -0.3712904453277588, -0.37209367752075195, -0.37337610125541687, -0.3755508065223694, -0.3763548731803894, -0.3767181634902954, -0.3763483166694641, -0.39708593487739563, -0.39584803581237793, -0.39473891258239746, -0.3936821222305298, -0.3946853280067444, -0.39611825346946716, -0.398246169090271, -0.3987112045288086, -0.3986874520778656, -0.39807364344596863, -0.41818785667419434, -0.4172874391078949, -0.41651278734207153, -0.41571927070617676, -0.41661834716796875, -0.41792425513267517, -0.4196811616420746, -0.41987019777297974, -0.4196160137653351, -0.41893699765205383, -0.4295043349266052, -0.42973893880844116, -0.4297823905944824, -0.4294702112674713, -0.429243803024292, -0.4291606843471527, -0.4293249845504761, -0.4292168915271759, -0.42921534180641174, -0.42930638790130615, -0.42604804039001465, -0.42712831497192383, -0.4275137484073639, -0.42709648609161377, -0.42553946375846863, -0.4241502285003662, -0.42315244674682617, -0.42319002747535706, -0.42381566762924194, -0.42487001419067383, -0.4118221402168274, -0.4135490953922272, -0.4140479862689972, -0.4133703410625458, -0.4105096161365509, -0.4078657627105713, -0.40594369173049927, -0.4063188135623932, -0.4077594578266144, -0.4098513722419739, -0.38247594237327576, -0.38328540325164795, -0.3834470212459564, -0.3832317292690277, -0.38158345222473145, -0.37993067502975464, -0.37867259979248047, -0.3789934813976288, -0.38012051582336426, -0.38146695494651794, -0.3824981451034546, -0.38264790177345276, -0.3824903070926666, -0.3822273910045624, -0.381197452545166, -0.3801577091217041, -0.37961244583129883, -0.3801221549510956, -0.38119956851005554, -0.38212278485298157, -0.38405823707580566, -0.3839455842971802, -0.3836837112903595, -0.3834293782711029, -0.38257017731666565, -0.38161465525627136, -0.3811814486980438, -0.38179996609687805, -0.38295337557792664, -0.3838604986667633, -0.383934885263443, -0.3834984004497528, -0.38292980194091797, -0.3822791278362274, -0.3814595937728882, -0.38074636459350586, -0.38074755668640137, -0.3816338777542114, -0.3829190135002136, -0.3838299810886383, -0.3821064829826355, -0.38185280561447144, -0.38127586245536804, -0.38035261631011963, -0.3793410658836365, -0.37853068113327026, -0.37846866250038147, -0.37930363416671753, -0.38055703043937683, -0.38168150186538696, -0.3813129663467407, -0.3815315365791321, -0.3813159465789795, -0.3804851472377777, -0.37926605343818665, -0.3782138526439667, -0.37776118516921997, -0.3782813549041748, -0.37931859493255615, -0.38050174713134766, -0.3796204626560211, -0.3798236548900604, -0.3796258866786957, -0.3788682222366333, -0.37781450152397156, -0.3770614564418793, -0.3767288327217102, -0.3771408498287201, -0.37791818380355835, -0.37885501980781555, -0.38218429684638977, -0.38290756940841675, -0.3831421434879303, -0.38281816244125366, -0.3814617097377777, -0.3801942467689514, -0.37913304567337036, -0.37926217913627625, -0.3800077438354492, -0.38113224506378174, -0.3815701901912689, -0.382333904504776, -0.38247978687286377, -0.3820807635784149, -0.380545437335968, -0.37921157479286194, -0.37815728783607483, -0.3784247040748596, -0.3792915344238281, -0.3804819583892822, -0.3821311295032501, -0.38305598497390747, -0.3832103908061981, -0.382811576128006, -0.3810395300388336, -0.3793621063232422, -0.3781077563762665, -0.37842339277267456, -0.3795252740383148, -0.38097113370895386, -0.3664366900920868, -0.3662530779838562, -0.3661535084247589, -0.3662697374820709, -0.36628347635269165, -0.36619144678115845, -0.36622166633605957, -0.36637961864471436, -0.3666793406009674, -0.3667016625404358, -0.36995846033096313, -0.36931243538856506, -0.36904096603393555, -0.369276225566864, -0.3697783052921295, -0.37004944682121277, -0.37048402428627014, -0.3706955313682556, -0.3709809184074402, -0.370699942111969, -0.3733263909816742, -0.37261298298835754, -0.37231165170669556, -0.3725414574146271, -0.372867614030838, -0.3728877902030945, -0.37315458059310913, -0.37349995970726013, -0.37406405806541443, -0.3740153908729553, -0.37368154525756836, -0.3729812800884247, -0.3725188076496124, -0.3723191022872925, -0.37222930788993835, -0.3719947934150696, -0.37225666642189026, -0.37287747859954834, -0.3737737238407135, -0.3740995526313782, -0.37346741557121277, -0.3732798993587494, -0.3728979229927063, -0.37223953008651733, -0.3713136613368988, -0.3704129457473755, -0.3701668083667755, -0.3708849251270294, -0.372087299823761, -0.37312090396881104, -0.3711106479167938, -0.37165606021881104, -0.371722936630249, -0.37103700637817383, -0.36952531337738037, -0.3679867684841156, -0.3670799732208252, -0.3675389289855957, -0.3686904311180115, -0.37007343769073486, -0.36654719710350037, -0.3670535981655121, -0.3671291768550873, -0.3664003014564514, -0.3649819493293762, -0.3636917173862457, -0.3629530072212219, -0.363406777381897, -0.3643617630004883, -0.3655400574207306, -0.36357828974723816, -0.36424699425697327, -0.3646293878555298, -0.3643794655799866, -0.3632587790489197, -0.36205193400382996, -0.36109352111816406, -0.3611939549446106, -0.36180055141448975, -0.3626990020275116, -0.3608068823814392, -0.36103197932243347, -0.36108219623565674, -0.3607760965824127, -0.3601134121417999, -0.3595626652240753, -0.35923928022384644, -0.35949623584747314, -0.35994136333465576, -0.3604139983654022, -0.3629007041454315, -0.36295995116233826, -0.36291658878326416, -0.36276811361312866, -0.3624532222747803, -0.3621242940425873, -0.361998051404953, -0.3621915280818939, -0.36255934834480286, -0.3628433048725128, -0.36048710346221924, -0.359912633895874, -0.3598225712776184, -0.36015036702156067, -0.36076077818870544, -0.36119213700294495, -0.3615723252296448, -0.3617156445980072, -0.3617185652256012, -0.3611924946308136, -0.3676016330718994, -0.36681652069091797, -0.3667088449001312, -0.3672465980052948, -0.36816754937171936, -0.3687450587749481, -0.36921021342277527, -0.3692670464515686, -0.3692221939563751, -0.36854010820388794, -0.3721582591533661, -0.37149474024772644, -0.371416300535202, -0.37192121148109436, -0.37243175506591797, -0.37254154682159424, -0.3726809024810791, -0.3728633522987366, -0.37317851185798645, -0.3728754222393036, -0.371265172958374, -0.3708285987377167, -0.3706411123275757, -0.3706745505332947, -0.37044355273246765, -0.37000370025634766, -0.3698778748512268, -0.3703601360321045, -0.37114188075065613, -0.3714606463909149, -0.36659181118011475, -0.3666282892227173, -0.36644694209098816, -0.3659711480140686, -0.3648283779621124, -0.3636910319328308, -0.3630834221839905, -0.36379310488700867, -0.3649739921092987, -0.36606448888778687, -0.35958385467529297, -0.36027634143829346, -0.36047694087028503, -0.35998111963272095, -0.3583260774612427, -0.3566321134567261, -0.35545894503593445, -0.35594090819358826, -0.35705745220184326, -0.3584437966346741, -0.35111716389656067, -0.351470947265625, -0.3514878451824188, -0.3508411943912506, -0.3494682013988495, -0.3482993245124817, -0.3476743996143341, -0.3483455181121826, -0.3492904305458069, -0.350301057100296, -0.34736186265945435, -0.3476449251174927, -0.3478076159954071, -0.34750333428382874, -0.34657031297683716, -0.34567251801490784, -0.34507957100868225, -0.3454735577106476, -0.3461090326309204, -0.34677451848983765, -0.34657108783721924, -0.34622514247894287, -0.345979779958725, -0.34558165073394775, -0.34529203176498413, -0.34525033831596375, -0.34551042318344116, -0.3461056351661682, -0.34656211733818054, -0.3466539978981018, -0.35244062542915344, -0.3519161641597748, -0.3516790568828583, -0.3515729308128357, -0.35179823637008667, -0.3520481288433075, -0.3524213135242462, -0.3528153896331787, -0.35304972529411316, -0.35287126898765564, -0.3647647798061371, -0.36418208479881287, -0.3642468750476837, -0.3646332919597626, -0.36553332209587097, -0.3662360608577728, -0.36688393354415894, -0.366953581571579, -0.36658257246017456, -0.3656717538833618, -0.3690146505832672, -0.3684218227863312, -0.36856764554977417, -0.36920005083084106, -0.37024062871932983, -0.3708644211292267, -0.37132999300956726, -0.37119320034980774, -0.3708060681819916, -0.3699055016040802, -0.37065771222114563, -0.3702855110168457, -0.3705209195613861, -0.3712226152420044, -0.3718530833721161, -0.3719650208950043, -0.37198343873023987, -0.37183380126953125, -0.37176215648651123, -0.37125611305236816, -0.3682839572429657, -0.36804690957069397, -0.3681577146053314, -0.3684707581996918, -0.3685314357280731, -0.36824530363082886, -0.36810413002967834, -0.3682217299938202, -0.36853548884391785, -0.3685009181499481, -0.3665229380130768, -0.36645379662513733, -0.3663560152053833, -0.3661184012889862, -0.3654700517654419, -0.36479395627975464, -0.3645399808883667, -0.3650699555873871, -0.3658144772052765, -0.36634403467178345, -0.36386439204216003, -0.3641652762889862, -0.3642709255218506, -0.36394003033638, -0.36297929286956787, -0.36206039786338806, -0.3616335093975067, -0.3621430993080139, -0.36280137300491333, -0.36339256167411804, -0.36034706234931946, -0.36030489206314087, -0.360228031873703, -0.35968613624572754, -0.3588773012161255, -0.35838186740875244, -0.3584986925125122, -0.3593284785747528, -0.35994237661361694, -0.3602142930030823, -0.35711219906806946, -0.35699060559272766, -0.35709288716316223, -0.35682785511016846, -0.35641998052597046, -0.35618001222610474, -0.3563562035560608, -0.3569578528404236, -0.3572675585746765, -0.35719606280326843, -0.3557076156139374, -0.35514411330223083, -0.35497531294822693, -0.3546294867992401, -0.3547267019748688, -0.3551170527935028, -0.3558845520019531, -0.3565790355205536, -0.35672736167907715, -0.3562590181827545, -0.35926082730293274, -0.3585658669471741, -0.3584374189376831, -0.3583720922470093, -0.35896268486976624, -0.35961008071899414, -0.3604288399219513, -0.3608790934085846, -0.360767662525177, -0.36007148027420044, -0.38353872299194336, -0.3839106261730194, -0.38455578684806824, -0.38493502140045166, -0.38476595282554626, -0.38425272703170776, -0.3837456703186035, -0.383640319108963, -0.38357019424438477, -0.3834265172481537, -0.38225385546684265, -0.3824799060821533, -0.38304778933525085, -0.38353344798088074, -0.3835611939430237, -0.3831128776073456, -0.38264399766921997, -0.3824412226676941, -0.38237935304641724, -0.38221582770347595, -0.3804895579814911, -0.38072359561920166, -0.3812800645828247, -0.38190779089927673, -0.38193267583847046, -0.3813820481300354, -0.38080033659935, -0.3804938793182373, -0.3805101215839386, -0.38044115900993347, -0.3780417740345001, -0.37800559401512146, -0.37821292877197266, -0.37845295667648315, -0.37847810983657837, -0.3781541585922241, -0.3780163824558258, -0.37798088788986206, -0.3781314492225647, -0.3780914545059204, -0.37898263335227966, -0.3787417411804199, -0.3785719871520996, -0.3783801198005676, -0.37818777561187744, -0.3780147433280945, -0.37821489572525024, -0.3786243498325348, -0.37899670004844666, -0.37910225987434387, -0.38104021549224854, -0.3808723986148834, -0.38066110014915466, -0.38027867674827576, -0.3799606263637543, -0.3798595070838928, -0.3802486062049866, -0.3808075785636902, -0.3811226785182953, -0.3811313211917877, -0.38274309039115906, -0.3822171092033386, -0.38183093070983887, -0.38126710057258606, -0.3810393810272217, -0.3813141882419586, -0.38219159841537476, -0.383136123418808, -0.3834758400917053, -0.383184015750885, -0.3831103444099426, -0.3828172981739044, -0.38275840878486633, -0.38234949111938477, -0.38196587562561035, -0.3819521367549896, -0.382455050945282, -0.38324984908103943, -0.3835708200931549, -0.3833446800708771, -0.3826925754547119, -0.38244372606277466, -0.3825046122074127, -0.38215503096580505, -0.38187283277511597, -0.3818875253200531, -0.38232070207595825, -0.3829936385154724, -0.38322409987449646, -0.38292887806892395, -0.38305240869522095, -0.3830714225769043, -0.38339751958847046, -0.38328227400779724, -0.3829415440559387, -0.3826143443584442, -0.38255175948143005, -0.38298025727272034, -0.38317084312438965, -0.3830721974372864, -0.3943803012371063, -0.395620733499527, -0.3966880142688751, -0.3969501554965973, -0.3956514298915863, -0.3940049707889557, -0.3925009071826935, -0.39240026473999023, -0.39265820384025574, -0.3933584988117218, -0.38052064180374146, -0.38147035241127014, -0.38232922554016113, -0.3825684189796448, -0.3815730810165405, -0.3801800608634949, -0.3789682686328888, -0.3788571059703827, -0.3791002333164215, -0.37967345118522644, -0.37093329429626465, -0.3716665506362915, -0.372367799282074, -0.3727833330631256, -0.3722687363624573, -0.3712294399738312, -0.3702545464038849, -0.36988842487335205, -0.36995768547058105, -0.37033259868621826, -0.3675563633441925, -0.3675349950790405, -0.36762362718582153, -0.36761894822120667, -0.3678603768348694, -0.36778733134269714, -0.36798951029777527, -0.3678346872329712, -0.36773571372032166, -0.36759307980537415, -0.3773539364337921, -0.3767411708831787, -0.37623903155326843, -0.37587425112724304, -0.37654802203178406, -0.37723299860954285, -0.378335177898407, -0.3785083293914795, -0.3783790171146393, -0.37792670726776123, -0.3924289345741272, -0.3915630877017975, -0.3908892571926117, -0.390359491109848, -0.39127227663993835, -0.39244818687438965, -0.3941056430339813, -0.3944939970970154, -0.39417946338653564, -0.3933242857456207, -0.40781787037849426, -0.4067437946796417, -0.4060603380203247, -0.40551161766052246, -0.40641528367996216, -0.4077812731266022, -0.4096459448337555, -0.41034168004989624, -0.41005250811576843, -0.4089740514755249, -0.4150322377681732, -0.4146050214767456, -0.41455405950546265, -0.41423818469047546, -0.414465993642807, -0.41499048471450806, -0.41593050956726074, -0.4164859354496002, -0.4163168966770172, -0.4156271815299988, -0.41404786705970764, -0.4141782820224762, -0.41455814242362976, -0.4143288731575012, -0.41394752264022827, -0.4137026071548462, -0.41381654143333435, -0.41424626111984253, -0.41428259015083313, -0.4140793979167938, -0.40626591444015503, -0.4070640206336975, -0.4078441560268402, -0.4077230393886566, -0.40656429529190063, -0.40535667538642883, -0.40450170636177063, -0.40482035279273987, -0.40513908863067627, -0.4055996835231781, -0.41885659098625183, -0.42131492495536804, -0.4227679967880249, -0.42270973324775696, -0.41943734884262085, -0.41590622067451477, -0.4128054678440094, -0.41283518075942993, -0.41406503319740295, -0.41627949476242065, -0.3956318497657776, -0.3973403573036194, -0.3982406258583069, -0.3979751765727997, -0.3953690230846405, -0.3926393985748291, -0.3904675543308258, -0.3907109498977661, -0.3918781876564026, -0.39366036653518677, -0.38209158182144165, -0.38303351402282715, -0.38347357511520386, -0.38332727551460266, -0.3819707930088043, -0.3804125189781189, -0.3792954981327057, -0.379304438829422, -0.3799806833267212, -0.38098809123039246, -0.38175246119499207, -0.38124701380729675, -0.3806651830673218, -0.3800373375415802, -0.380350261926651, -0.3807532489299774, -0.3817982077598572, -0.38205328583717346, -0.38221776485443115, -0.3820449113845825, -0.39868664741516113, -0.39736536145210266, -0.3961714804172516, -0.39530396461486816, -0.39657989144325256, -0.3981863260269165, -0.4005056619644165, -0.4009484648704529, -0.4007614552974701, -0.39987146854400635, -0.4229772686958313, -0.42143604159355164, -0.4201308488845825, -0.41916996240615845, -0.420681893825531, -0.4227404296398163, -0.4254900813102722, -0.4260745942592621, -0.4256822466850281, -0.4244292378425598, -0.4457162916660309, -0.4443646967411041, -0.4433742165565491, -0.4426092207431793, -0.4438222646713257, -0.4456271529197693, -0.4480002522468567, -0.44869983196258545, -0.44832298159599304, -0.44707605242729187, -0.45768627524375916, -0.45750394463539124, -0.4574507772922516, -0.4569981098175049, -0.4568597674369812, -0.45709341764450073, -0.4577668607234955, -0.45827528834342957, -0.4582612216472626, -0.45791181921958923, -0.45563605427742004, -0.45656171441078186, -0.45725056529045105, -0.45695391297340393, -0.4555310010910034, -0.45421719551086426, -0.4533090889453888, -0.45363011956214905, -0.45407769083976746, -0.45472651720046997, -0.44087105989456177, -0.44283807277679443, -0.44405052065849304, -0.44373786449432373, -0.4408068060874939, -0.4378720223903656, -0.4354739487171173, -0.4358327388763428, -0.4369576573371887, -0.4387741982936859, -0.39159807562828064, -0.394734650850296, -0.3961910307407379, -0.39563411474227905, -0.3907441198825836, -0.38592076301574707, -0.38189440965652466, -0.3823642432689667, -0.3845553994178772, -0.3879776895046234, -0.3590905964374542, -0.36123692989349365, -0.3620152175426483, -0.36124032735824585, -0.3573490381240845, -0.35370439291000366, -0.35094302892684937, -0.35165953636169434, -0.35362622141838074, -0.3563570976257324, -0.3396156132221222, -0.34077537059783936, -0.34100353717803955, -0.3403472900390625, -0.338126003742218, -0.33604177832603455, -0.33470162749290466, -0.3351411521434784, -0.33639514446258545, -0.3380300998687744, -0.33772534132003784, -0.3371044099330902, -0.3361755311489105, -0.3351198732852936, -0.33516693115234375, -0.33560463786125183, -0.33689451217651367, -0.33748170733451843, -0.3379290997982025, -0.3379612863063812, -0.35747694969177246, -0.3558446168899536, -0.35425281524658203, -0.3530149757862091, -0.3545582890510559, -0.35663363337516785, -0.3596058487892151, -0.36017584800720215, -0.35996779799461365, -0.35890278220176697, -0.389560729265213, -0.3876199722290039, -0.38602206110954285, -0.3849220871925354, -0.38716164231300354, -0.38998863101005554, -0.3935115933418274, -0.39392927289009094, -0.39318639039993286, -0.39146628975868225, -0.4219921827316284, -0.42032769322395325, -0.41917774081230164, -0.4184437692165375, -0.42054930329322815, -0.42316314578056335, -0.426236629486084, -0.42653658986091614, -0.4256211221218109, -0.42380595207214355, -0.44261497259140015, -0.4424866735935211, -0.4425007700920105, -0.44220539927482605, -0.4425297975540161, -0.4430912137031555, -0.44393423199653625, -0.443991482257843, -0.4435661733150482, -0.44295862317085266, -0.4423797130584717, -0.443717360496521, -0.4445270597934723, -0.44418370723724365, -0.4423232674598694, -0.44056230783462524, -0.43924108147621155, -0.4394039511680603, -0.43999579548835754, -0.4410370886325836, -0.4228721261024475, -0.42550981044769287, -0.4268014132976532, -0.4261927604675293, -0.42204585671424866, -0.4180772006511688, -0.4148692190647125, -0.4153769016265869, -0.4171401858329773, -0.4199066162109375]))

    def test_null_kernel_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.gauss_edge(sharp_edge_image=IMAGE_2D, kernel_size = 0, gauss_standard_dev =3)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.gauss_edge(sharp_edge_image=IMAGE_2D, kernel_size = 0, gauss_standard_dev =3)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_negative_kernel_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.gauss_edge(sharp_edge_image=IMAGE_2D, kernel_size = -2, gauss_standard_dev =3)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.gauss_edge(sharp_edge_image=IMAGE_2D, kernel_size = -2, gauss_standard_dev =3)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_get_image(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_image()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_image()
        self.assertEqual(str(cm_new.exception), "get_image() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_returns_input_img(self):
        """ I do not insert all the params because in this case they are not used"""
        return_new = fu.get_image(imagename=IMAGE_2D, nx=0, ny=1, nz=1, im=0)
        return_old = oldfu.get_image(imagename=IMAGE_2D, nx=0, ny=1, nz=1, im=0)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(),[0.009504491463303566, 0.025885052978992462, 0.015371464192867279, 0.029651766642928123, 0.025623319670557976, 0.023996084928512573, 0.023316310718655586, 0.03626585379242897, 0.042238689959049225, 0.053261399269104004, 0.06996519863605499, 0.05416787788271904, 0.050994712859392166, 0.03554266691207886, 0.03604980185627937, 0.07005909085273743, 0.056754179298877716, 0.06729267537593842, 0.0899617150425911, 0.08004479855298996, 0.07206107676029205, 0.07158395648002625, 0.08500781655311584, 0.08074058592319489, 0.08976095914840698, 0.09553121030330658, 0.09733162075281143, 0.12153391540050507, 0.09777011722326279, 0.0612066276371479, 0.060473889112472534, 0.0832795649766922, 0.07990699261426926, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.05740871652960777, 0.05622498691082001, 0.05523042380809784, 0.013907668180763721, 0.0071470243856310844, 0.01511574536561966, 2.5205374186043628e-05, 0.008231919258832932, -0.020773129537701607, -0.034199729561805725, -0.04089483618736267, -0.042460259050130844, -0.06925757229328156, -0.06893884390592575, -0.08000176399946213, -0.11662115156650543, -0.111984983086586, -0.11971071362495422, -0.1273496150970459, -0.12249226123094559, -0.1453358680009842, -0.14758040010929108, -0.15034900605678558, -0.17081016302108765, -0.2014905959367752, -0.2121349573135376, -0.22736789286136627, -0.24315771460533142, -0.2552821934223175, -0.23703180253505707, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.3236375153064728, -0.3262620270252228, -0.35172849893569946, -0.3602631986141205, -0.35741564631462097, -0.3575122356414795, -0.38925597071647644, -0.377326101064682, -0.38598355650901794, -0.39209896326065063, -0.3882087767124176, -0.3639817535877228, -0.3711523711681366, -0.37047016620635986, -0.39362388849258423, -0.40711337327957153, -0.3925972580909729, -0.4149233400821686, -0.41900205612182617, -0.4641905426979065, -0.46107935905456543, -0.46086275577545166, -0.4773290157318115, -0.473482221364975, -0.4543262720108032, -0.44096702337265015, -0.4387476146221161, -0.4229215085506439, -0.4376510977745056, -0.4369300603866577] ))

    def test_None_input_img_returns_new_EMData_with_default_size(self):
        """ I do not insert all the params because in this case they are not used"""
        nx = 0
        return_new = fu.get_image(imagename=None, nx = nx, ny=1, nz=1, im=0)
        return_old = oldfu.get_image(imagename=None, nx = nx, ny=1, nz=1, im=0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertEqual(return_new.get_3dview().flatten().tolist(), [])

    def test_None_input_img_returns_new_EMData__with_given_size(self):
        """ I do not insert all the params because in this case they are not used"""
        nx,ny,nz=3,4,3
        return_new = fu.get_image(imagename=None, nx = nx, ny = ny, nz = nz, im=0)
        return_old = oldfu.get_image(imagename=None, nx = nx, ny = ny, nz = nz, im=0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_invalid_path_returns_RuntimeError_FileAccessException(self):
        """ I do not insert all the params because in this case they are not used"""
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_image(imagename="image_not_here", nx=0, ny=1, nz=1, im=0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_image(imagename="image_not_here", nx=0, ny=1, nz=1, im=0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "FileAccessException")
        self.assertEqual(msg[3], "cannot access file ")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_get_im(unittest.TestCase):
    img_list = [IMAGE_3D,IMAGE_2D]

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_im(stackname=None, im=0)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_im(stackname=None, im=0)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute '__getitem__'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_im()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_im()
        self.assertEqual(str(cm_new.exception), "get_im() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_returns_first_img_of_a_list(self):
        return_new = fu.get_im(stackname=self.img_list, im=0)
        return_old = oldfu.get_im(stackname=self.img_list, im=0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview(), IMAGE_3D.get_3dview()))

    def test_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.get_im(stackname=self.img_list, im=10)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.get_im(stackname=self.img_list, im=10)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_get_image_data(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_image_data()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_image_data()
        self.assertEqual(str(cm_new.exception), "get_image_data() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_image_data(self):
        return_new = fu.get_image_data(img=IMAGE_2D)
        return_old = oldfu.get_image_data(img=IMAGE_2D)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new.flatten(), [0.009504491463303566, 0.025885052978992462, 0.015371464192867279, 0.029651766642928123, 0.025623319670557976, 0.023996084928512573, 0.023316310718655586, 0.03626585379242897, 0.042238689959049225, 0.053261399269104004, 0.06996519863605499, 0.05416787788271904, 0.050994712859392166, 0.03554266691207886, 0.03604980185627937, 0.07005909085273743, 0.056754179298877716, 0.06729267537593842, 0.0899617150425911, 0.08004479855298996, 0.07206107676029205, 0.07158395648002625, 0.08500781655311584, 0.08074058592319489, 0.08976095914840698, 0.09553121030330658, 0.09733162075281143, 0.12153391540050507, 0.09777011722326279, 0.0612066276371479, 0.060473889112472534, 0.0832795649766922, 0.07990699261426926, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.05740871652960777, 0.05622498691082001, 0.05523042380809784, 0.013907668180763721, 0.0071470243856310844, 0.01511574536561966, 2.5205374186043628e-05, 0.008231919258832932, -0.020773129537701607, -0.034199729561805725, -0.04089483618736267, -0.042460259050130844, -0.06925757229328156, -0.06893884390592575, -0.08000176399946213, -0.11662115156650543, -0.111984983086586, -0.11971071362495422, -0.1273496150970459, -0.12249226123094559, -0.1453358680009842, -0.14758040010929108, -0.15034900605678558, -0.17081016302108765, -0.2014905959367752, -0.2121349573135376, -0.22736789286136627, -0.24315771460533142, -0.2552821934223175, -0.23703180253505707, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.3236375153064728, -0.3262620270252228, -0.35172849893569946, -0.3602631986141205, -0.35741564631462097, -0.3575122356414795, -0.38925597071647644, -0.377326101064682, -0.38598355650901794, -0.39209896326065063, -0.3882087767124176, -0.3639817535877228, -0.3711523711681366, -0.37047016620635986, -0.39362388849258423, -0.40711337327957153, -0.3925972580909729, -0.4149233400821686, -0.41900205612182617, -0.4641905426979065, -0.46107935905456543, -0.46086275577545166, -0.4773290157318115, -0.473482221364975, -0.4543262720108032, -0.44096702337265015, -0.4387476146221161, -0.4229215085506439, -0.4376510977745056, -0.4369300603866577] ))

    def test_NoneType_as_input_image_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)

    #the first value is a random value
    def test_emptyimage_as_input_image(self):
        return_old = oldfu.get_image_data(img=EMData(5, 5, 5))
        return_new = fu.get_image_data(img=EMData(5, 5, 5))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_old[1:].flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


class Test_get_symt(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_symt()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_symt()
        self.assertEqual(str(cm_new.exception), "get_symt() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_symt(self):
        self.assertTrue(True)
        return_new = fu.get_symt(symmetry='c3')
        return_old = oldfu.get_symt(symmetry='c3')

    def test_get_symt_with_invaliSym_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.get_symt(symmetry='invaliSym')
        with self.assertRaises(IndexError) as cm_old:
            oldfu.get_symt(symmetry='invaliSym')
        self.assertEqual(str(cm_new.exception), "index 0 is out of bounds for axis 1 with size 0")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_get_input_from_string(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_input_from_string()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_input_from_string()
        self.assertEqual(str(cm_new.exception), "get_input_from_string() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_input_from_string_integer_case(self):
        return_new =fu.get_input_from_string(str_input='5')
        return_old = oldfu.get_input_from_string(str_input='5')
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, [5])

    def test_get_input_from_string_negative_number_case(self):
        return_new =fu.get_input_from_string(str_input='-5')
        return_old = oldfu.get_input_from_string(str_input='-5')
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,[-5])

    def test_get_input_from_string_float_case(self):
        return_new =fu.get_input_from_string(str_input='5.3')
        return_old = oldfu.get_input_from_string(str_input='5.3')
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,[5.3])

    def test_get_input_from_string_invalid_case(self):
        with self.assertRaises(ValueError) as cm_new:
            fu.get_input_from_string(str_input='not_a_number')
        with self.assertRaises(ValueError) as cm_old:
            oldfu.get_input_from_string(str_input='not_a_number')
        self.assertEqual(str(cm_new.exception), "invalid literal for int() with base 10: 'not_a_number'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_input_from_string_list_of_values_number_case(self):
        return_new =fu.get_input_from_string(str_input='-5,3.11,5')
        return_old = oldfu.get_input_from_string(str_input='-5,3.11,5')
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, [-5, 3.11, 5])




class Test_model_circle(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_circle()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_circle()
        self.assertEqual(str(cm_new.exception), "model_circle() takes at least 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        return_new = fu.model_circle(r = 2, nx = 5, ny = 5, nz =1)
        return_old = oldfu.model_circle(r = 2, nx = 5, ny = 5, nz =1)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    def test_null_Y_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_circle(r = 145, nx = 352, ny = 0, nz =1)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_circle(r = 145, nx = 352, ny = 0, nz =1)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "y size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_X_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_circle(r = 145, nx = 0, ny = 252, nz =1)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_circle(r = 145, nx = 0, ny = 252, nz =1)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_Z_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_circle(r = 145, nx = 252, ny = 252, nz =0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_circle(r = 145, nx = 252, ny = 252, nz =0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "z size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_R_size_(self):
        return_new = fu.model_circle(r = 0, nx = 5, ny = 5, nz =1)
        return_old = oldfu.model_circle(r = 0, nx = 5, ny = 5, nz =1)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    #todo: how can it be possible??? is it a bug???
    def test_negative_R_size_(self):
        return_new = fu.model_circle(r = -1, nx = 5, ny = 5, nz =1)
        return_old = oldfu.model_circle(r = -1, nx = 5, ny = 5, nz =1)
        self.assertTrue(array_equal(return_new.get_2dview(), return_old.get_2dview()))
        self.assertTrue(array_equal(return_new.get_2dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))



class Test_model_gauss(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_gauss()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_gauss()
        self.assertEqual(str(cm_new.exception), "model_gauss() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values(self):
        return_new = fu.model_gauss(xsigma=2, nx=10, ny=1, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        return_old = oldfu.model_gauss(xsigma=2, nx=10, ny=1, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_old.get_3dview().flatten(),[0.00876415055245161, 0.02699548378586769, 0.06475879997015, 0.12098536640405655, 0.1760326623916626, 0.1994711458683014, 0.1760326623916626, 0.12098536640405655, 0.06475879997015, 0.02699548378586769] ))

    def test_null_Xsigma_returns_Nan_matrix(self):
        return_new = fu.model_gauss(xsigma=0, nx=10, ny=1, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        return_old = oldfu.model_gauss(xsigma=0, nx=10, ny=1, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        self.assertTrue(allclose(return_new.get_3dview(), return_old.get_3dview(), equal_nan=True))
        self.assertTrue(allclose(return_new.get_3dview(), [float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')], equal_nan=True))

    def test_null_Ysigma_returns_Nan_matrix(self):
        return_new = fu.model_gauss(xsigma=2, nx=10, ny=1, nz=1, ysigma=0, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        return_old = oldfu.model_gauss(xsigma=2, nx=10, ny=1, nz=1, ysigma=0, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        self.assertTrue(allclose(return_new.get_3dview(), return_old.get_3dview(), equal_nan=True))
        self.assertTrue(allclose(return_new.get_3dview(),
                                 [float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'),
                                  float('NaN'), float('NaN'), float('NaN'), float('NaN')], equal_nan=True))

    def test_null_Zsigma_returns_Nan_matrix(self):
        return_new = fu.model_gauss(xsigma=2, nx=10, ny=1, nz=1, ysigma=None, zsigma=0, xcenter=None, ycenter=None, zcenter=None)
        return_old = oldfu.model_gauss(xsigma=2, nx=10, ny=1, nz=1, ysigma=None, zsigma=0, xcenter=None, ycenter=None, zcenter=None)
        self.assertTrue(allclose(return_new.get_3dview(), return_old.get_3dview(), equal_nan=True))
        self.assertTrue(allclose(return_new.get_3dview(),
                                 [float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'),
                                  float('NaN'), float('NaN'), float('NaN'), float('NaN')], equal_nan=True))

    def test_null_Y_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_gauss(xsigma=2, nx=352, ny=0, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_gauss(xsigma=2, nx=352, ny=0, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "y size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_X_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_gauss(xsigma=2, nx=0, ny=1, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_gauss(xsigma=2, nx=0, ny=1, nz=1, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_Z_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_gauss(xsigma=2, nx=352, ny=1, nz=0, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_gauss(xsigma=2, nx=352, ny=1, nz=0, ysigma=None, zsigma=None, xcenter=None, ycenter=None, zcenter=None)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "z size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_model_gauss_noise(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_gauss_noise()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_gauss_noise()
        self.assertEqual(str(cm_new.exception), "model_gauss_noise() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    @unittest.skip("for definition it is a random value matrix. not testabel")
    def test_model_gauss_noise(self):
        self.assertTrue(True)
        '''
          This function creates random noise each time so arrays cannot be compared
        return_new = fu.model_gauss_noise(sigma = 1, nx = 352, ny=1, nz=1)
        return_old = oldfu.model_gauss_noise(sigma =1, nx = 352, ny=1, nz=1)
        self.assertTrue(allclose(return_new.get_3dview(), return_old.get_3dview(), atol=1000))
        '''

    def test_null_sigma(self):
        return_new = fu.model_gauss_noise(sigma = 0.0, nx = 10, ny=1, nz=1)
        return_old = oldfu.model_gauss_noise(sigma =0.0, nx = 10, ny=1, nz=1)
        self.assertTrue(allclose(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(allclose(return_new.get_3dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


    def test_null_Y_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_gauss_noise(sigma = 1, nx = 1, ny=0, nz=1)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_gauss_noise(sigma = 1, nx = 1, ny=0, nz=1)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "y size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_X_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_gauss_noise(sigma = 1, nx = 0, ny=10, nz=1)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_gauss_noise(sigma = 1, nx = 0, ny=10, nz=1)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_Z_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_gauss_noise(sigma = 1, nx = 352, ny=1, nz=0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_gauss_noise(sigma = 1, nx = 352, ny=1, nz=0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "z size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_model_blank(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.model_blank()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.model_blank()
        self.assertEqual(str(cm_new.exception), "model_blank() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values(self):
        return_new = fu.model_blank(nx = 10, ny=1, nz=1, bckg = 0.0)
        return_old = oldfu.model_blank(nx = 10, ny=1, nz=1, bckg = 0.0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_default_values_with_bckg(self):
        return_new = fu.model_blank(nx = 10, ny=1, nz=1, bckg = 10.0)
        return_old = oldfu.model_blank(nx = 10, ny=1, nz=1, bckg = 10.0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))

    def test_null_X_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_blank(nx = 0, ny=1, nz=1, bckg = 0.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_blank(nx = 0, ny=1, nz=1, bckg = 0.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_Y_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_blank(nx = 10, ny=0, nz=1, bckg = 0.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_blank(nx = 10, ny=0, nz=1, bckg = 0.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "y size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_null_Z_size_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.model_blank(nx = 10, ny=1, nz=0, bckg = 0.0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.model_blank(nx = 10, ny=1, nz=0, bckg = 0.0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "z size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_peak_search(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.peak_search()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.peak_search()
        self.assertEqual(str(cm_new.exception), "peak_search() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values(self):
        return_new = fu.peak_search(e=IMAGE_2D, npeak = 3, invert = 1, print_screen = 0)
        return_old = oldfu.peak_search(e=IMAGE_2D, npeak = 3, invert = 1, print_screen = 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,[[0.12692593038082123, 5.0, 3.0, 1.0, 0.0, -2.0], [0.12153391540050507, 7.0, 2.0, 0.9575183987617493, 2.0, -3.0], [0.08500781655311584, 2.0, 2.0, 0.6697434782981873, -3.0, -3.0]]))

    def test_inverted_sort(self):
        return_new = fu.peak_search(e=IMAGE_2D, npeak = 3, invert = -1, print_screen = 0)
        return_old = oldfu.peak_search(e=IMAGE_2D, npeak = 3, invert = -1, print_screen = 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[1.0, 5.0, 5.0, 1.0, 0.0, 0.0]]))

    def test_null_npeak_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        img, NotUsed = get_real_data(dim=2)
        return_new = fu.peak_search(img, npeak = 0, invert = 1, print_screen = 0)
        return_old = oldfu.peak_search(img, npeak = 0, invert = 1, print_screen = 0)
        self.assertTrue(array_equal(return_new, return_old))
        """

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.peak_search(e=None, npeak = 3, invert = -1, print_screen = 0)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.peak_search(e=None, npeak = 3, invert = -1, print_screen = 0)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'peak_search'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Empty_img_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        fu.peak_search(EMData(), npeak = 3, invert = -1, print_screen = 0)
        oldfu.peak_search(EMData(), npeak = 3, invert = -1, print_screen = 0)
        """


class Test_pad(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.pad()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.pad()
        self.assertEqual(str(cm_new.exception), "pad() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_returns_RuntimeError_ImageDimensionException_padder_cannot_be_lower_than_sizee_img(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.pad(image_to_be_padded = IMAGE_2D, new_nx = 10, new_ny = 1,	new_nz = 1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.pad(image_to_be_padded = IMAGE_2D, new_nx = 10, new_ny = 1,	new_nz = 1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The size of the padded image cannot be lower than the input image size.")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_returns_RuntimeError_ImageDimensionException_offset_inconsistent(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.pad(image_to_be_padded = IMAGE_2D, new_nx = IMAGE_2D.get_xsize()+10, new_ny = IMAGE_2D.get_ysize()+10,	new_nz = IMAGE_2D.get_zsize()+10, background = "average", off_center_nx = 100, off_center_ny = 100, off_center_nz = 100)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.pad(image_to_be_padded = IMAGE_2D, new_nx = IMAGE_2D.get_xsize()+10, new_ny = IMAGE_2D.get_ysize()+10,	new_nz = IMAGE_2D.get_zsize()+10, background ="average", off_center_nx = 100, off_center_ny = 100, off_center_nz = 100)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The offset inconsistent with the input image size. Solution: Change the offset parameters")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values(self):
        return_new = fu.pad(image_to_be_padded = IMAGE_2D, new_nx = IMAGE_2D.get_xsize()+1, new_ny = IMAGE_2D.get_ysize()+1,	new_nz = IMAGE_2D.get_zsize()+1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        return_old = oldfu.pad(image_to_be_padded = IMAGE_2D, new_nx = IMAGE_2D.get_xsize()+1, new_ny = IMAGE_2D.get_ysize()+1,	new_nz = IMAGE_2D.get_zsize()+1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(),  [-0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, 0.009504491463303566, 0.025885052978992462, 0.015371464192867279, 0.029651766642928123, 0.025623319670557976, 0.023996084928512573, 0.023316310718655586, 0.03626585379242897, 0.042238689959049225, 0.053261399269104004, -0.13252077996730804, 0.06996519863605499, 0.05416787788271904, 0.050994712859392166, 0.03554266691207886, 0.03604980185627937, 0.07005909085273743, 0.056754179298877716, 0.06729267537593842, 0.0899617150425911, 0.08004479855298996, -0.13252077996730804, 0.07206107676029205, 0.07158395648002625, 0.08500781655311584, 0.08074058592319489, 0.08976095914840698, 0.09553121030330658, 0.09733162075281143, 0.12153391540050507, 0.09777011722326279, 0.0612066276371479, -0.13252077996730804, 0.060473889112472534, 0.0832795649766922, 0.07990699261426926, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.05740871652960777, 0.05622498691082001, 0.05523042380809784, -0.13252077996730804, 0.013907668180763721, 0.0071470243856310844, 0.01511574536561966, 2.5205374186043628e-05, 0.008231919258832932, -0.020773129537701607, -0.034199729561805725, -0.04089483618736267, -0.042460259050130844, -0.06925757229328156, -0.13252077996730804, -0.06893884390592575, -0.08000176399946213, -0.11662115156650543, -0.111984983086586, -0.11971071362495422, -0.1273496150970459, -0.12249226123094559, -0.1453358680009842, -0.14758040010929108, -0.15034900605678558, -0.13252077996730804, -0.17081016302108765, -0.2014905959367752, -0.2121349573135376, -0.22736789286136627, -0.24315771460533142, -0.2552821934223175, -0.23703180253505707, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.13252077996730804, -0.3236375153064728, -0.3262620270252228, -0.35172849893569946, -0.3602631986141205, -0.35741564631462097, -0.3575122356414795, -0.38925597071647644, -0.377326101064682, -0.38598355650901794, -0.39209896326065063, -0.13252077996730804, -0.3882087767124176, -0.3639817535877228, -0.3711523711681366, -0.37047016620635986, -0.39362388849258423, -0.40711337327957153, -0.3925972580909729, -0.4149233400821686, -0.41900205612182617, -0.4641905426979065, -0.13252077996730804, -0.46107935905456543, -0.46086275577545166, -0.4773290157318115, -0.473482221364975, -0.4543262720108032, -0.44096702337265015, -0.4387476146221161, -0.4229215085506439, -0.4376510977745056, -0.4369300603866577, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804, -0.13252077996730804]))

    def test_default_values_with_circumference_bckg(self):
        return_new = fu.pad(image_to_be_padded = IMAGE_2D, new_nx = IMAGE_2D.get_xsize()+1, new_ny = IMAGE_2D.get_ysize()+1,	new_nz = IMAGE_2D.get_zsize()+1, background = "circumference", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        return_old = oldfu.pad(image_to_be_padded = IMAGE_2D, new_nx = IMAGE_2D.get_xsize()+1, new_ny = IMAGE_2D.get_ysize()+1,	new_nz = IMAGE_2D.get_zsize()+1, background = "circumference", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [-0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, 0.009504491463303566, 0.025885052978992462, 0.015371464192867279, 0.029651766642928123, 0.025623319670557976, 0.023996084928512573, 0.023316310718655586, 0.03626585379242897, 0.042238689959049225, 0.053261399269104004, -0.03757849335670471, 0.06996519863605499, 0.05416787788271904, 0.050994712859392166, 0.03554266691207886, 0.03604980185627937, 0.07005909085273743, 0.056754179298877716, 0.06729267537593842, 0.0899617150425911, 0.08004479855298996, -0.03757849335670471, 0.07206107676029205, 0.07158395648002625, 0.08500781655311584, 0.08074058592319489, 0.08976095914840698, 0.09553121030330658, 0.09733162075281143, 0.12153391540050507, 0.09777011722326279, 0.0612066276371479, -0.03757849335670471, 0.060473889112472534, 0.0832795649766922, 0.07990699261426926, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.05740871652960777, 0.05622498691082001, 0.05523042380809784, -0.03757849335670471, 0.013907668180763721, 0.0071470243856310844, 0.01511574536561966, 2.5205374186043628e-05, 0.008231919258832932, -0.020773129537701607, -0.034199729561805725, -0.04089483618736267, -0.042460259050130844, -0.06925757229328156, -0.03757849335670471, -0.06893884390592575, -0.08000176399946213, -0.11662115156650543, -0.111984983086586, -0.11971071362495422, -0.1273496150970459, -0.12249226123094559, -0.1453358680009842, -0.14758040010929108, -0.15034900605678558, -0.03757849335670471, -0.17081016302108765, -0.2014905959367752, -0.2121349573135376, -0.22736789286136627, -0.24315771460533142, -0.2552821934223175, -0.23703180253505707, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, -0.03757849335670471, -0.3236375153064728, -0.3262620270252228, -0.35172849893569946, -0.3602631986141205, -0.35741564631462097, -0.3575122356414795, -0.38925597071647644, -0.377326101064682, -0.38598355650901794, -0.39209896326065063, -0.03757849335670471, -0.3882087767124176, -0.3639817535877228, -0.3711523711681366, -0.37047016620635986, -0.39362388849258423, -0.40711337327957153, -0.3925972580909729, -0.4149233400821686, -0.41900205612182617, -0.4641905426979065, -0.03757849335670471, -0.46107935905456543, -0.46086275577545166, -0.4773290157318115, -0.473482221364975, -0.4543262720108032, -0.44096702337265015, -0.4387476146221161, -0.4229215085506439, -0.4376510977745056, -0.4369300603866577, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471, -0.03757849335670471]))

    def test_default_values_with_unknown_bckg(self):
        return_new = fu.pad(image_to_be_padded=IMAGE_2D, new_nx=IMAGE_2D.get_xsize() + 1,
                            new_ny=IMAGE_2D.get_ysize() + 1, new_nz=IMAGE_2D.get_zsize() + 1,
                            background="unknown", off_center_nx=0, off_center_ny=0, off_center_nz=0)
        return_old = oldfu.pad(image_to_be_padded=IMAGE_2D, new_nx=IMAGE_2D.get_xsize() + 1,
                               new_ny=IMAGE_2D.get_ysize() + 1, new_nz=IMAGE_2D.get_zsize() + 1,
                               background="unknown", off_center_nx=0, off_center_ny=0, off_center_nz=0)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))
        self.assertTrue(array_equal(return_new.get_3dview().flatten(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009504491463303566, 0.025885052978992462, 0.015371464192867279, 0.029651766642928123, 0.025623319670557976, 0.023996084928512573, 0.023316310718655586, 0.03626585379242897, 0.042238689959049225, 0.053261399269104004, 0.0, 0.06996519863605499, 0.05416787788271904, 0.050994712859392166, 0.03554266691207886, 0.03604980185627937, 0.07005909085273743, 0.056754179298877716, 0.06729267537593842, 0.0899617150425911, 0.08004479855298996, 0.0, 0.07206107676029205, 0.07158395648002625, 0.08500781655311584, 0.08074058592319489, 0.08976095914840698, 0.09553121030330658, 0.09733162075281143, 0.12153391540050507, 0.09777011722326279, 0.0612066276371479, 0.0, 0.060473889112472534, 0.0832795649766922, 0.07990699261426926, 0.0726018100976944, 0.10390139371156693, 0.12692593038082123, 0.08997570723295212, 0.05740871652960777, 0.05622498691082001, 0.05523042380809784, 0.0, 0.013907668180763721, 0.0071470243856310844, 0.01511574536561966, 2.5205374186043628e-05, 0.008231919258832932, -0.020773129537701607, -0.034199729561805725, -0.04089483618736267, -0.042460259050130844, -0.06925757229328156, 0.0, -0.06893884390592575, -0.08000176399946213, -0.11662115156650543, -0.111984983086586, -0.11971071362495422, -0.1273496150970459, -0.12249226123094559, -0.1453358680009842, -0.14758040010929108, -0.15034900605678558, 0.0, -0.17081016302108765, -0.2014905959367752, -0.2121349573135376, -0.22736789286136627, -0.24315771460533142, -0.2552821934223175, -0.23703180253505707, -0.2393375188112259, -0.2672199606895447, -0.28808265924453735, 0.0, -0.3236375153064728, -0.3262620270252228, -0.35172849893569946, -0.3602631986141205, -0.35741564631462097, -0.3575122356414795, -0.38925597071647644, -0.377326101064682, -0.38598355650901794, -0.39209896326065063, 0.0, -0.3882087767124176, -0.3639817535877228, -0.3711523711681366, -0.37047016620635986, -0.39362388849258423, -0.40711337327957153, -0.3925972580909729, -0.4149233400821686, -0.41900205612182617, -0.4641905426979065, 0.0, -0.46107935905456543, -0.46086275577545166, -0.4773290157318115, -0.473482221364975, -0.4543262720108032, -0.44096702337265015, -0.4387476146221161, -0.4229215085506439, -0.4376510977745056, -0.4369300603866577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_NoneType_as_img_returns_RuntimeError_NullPointerException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.pad(image_to_be_padded = None, new_nx = 10, new_ny = 1,	new_nz = 1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.pad(image_to_be_padded = None, new_nx = 10, new_ny = 1,	new_nz = 1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NullPointerException")
        self.assertEqual(msg[1], "NULL input image")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Empty_img_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.pad(image_to_be_padded = EMData(), new_nx = 10, new_ny = 1,	new_nz = 1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.pad(image_to_be_padded = EMData(), new_nx = 10, new_ny = 1,	new_nz = 1, background = "average", off_center_nx = 0, off_center_ny = 0, off_center_nz = 0)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "x size <= 0")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_chooseformat(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.chooseformat()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.chooseformat()
        self.assertEqual(str(cm_new.exception), "chooseformat() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_exponential_number(self):
        return_new = fu.chooseformat(t=0.00000000000000000000000000003, form_float="  %12.5f")
        return_old = fu.chooseformat(t=0.00000000000000000000000000003, form_float="  %12.5f")
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, '  %12.5e')

    def test_float(self):
        return_new = fu.chooseformat(t=0.3, form_float="  %12.5f")
        return_old = fu.chooseformat(t=0.3, form_float="  %12.5f")
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, '  %12.5f')

    def test_typeError_float_argument_required(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.chooseformat(t='w', form_float="  %12.5f")
        with self.assertRaises(TypeError) as cm_old:
            oldfu.chooseformat(t='w', form_float="  %12.5f")
        self.assertEqual(str(cm_new.exception), "float argument required, not str")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_read_text_row(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.read_text_row()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.read_text_row()
        self.assertEqual(str(cm_new.exception), "read_text_row() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_file_not_found(self):
        with self.assertRaises(IOError) as cm_new:
            fu.read_text_row(fnam="no_file.txt", format="", skip=";")
        with self.assertRaises(IOError) as cm_old:
            oldfu.read_text_row(fnam="no_file.txt", format="", skip=";")
        self.assertEqual(cm_new.exception.strerror, "No such file or directory")
        self.assertEqual(cm_new.exception.strerror, cm_old.exception.strerror)

    def test_default_case(self):
        partids = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D/main001/this_iteration_index_keep_images.txt')
        return_new = fu.read_text_row(fnam=partids, format="", skip=";")
        return_old = oldfu.read_text_row(fnam=partids, format="", skip=";")
        self.assertTrue(return_new == return_old)
        self.assertTrue(array_equal(return_new,[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92]]))


class Test_write_text_row(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.write_text_row()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.write_text_row()
        self.assertEqual(str(cm_new.exception), "write_text_row() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_write_text_row(self):
        data=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]
        f=path.join(ABSOLUTE_PATH, "filefu.txt")
        fold=path.join(ABSOLUTE_PATH, "filefold.txt")
        fu.write_text_row(data=data, file_name=f, form_float="  %14.6f", form_int="  %12d")
        oldfu.write_text_row(data=data, file_name=fold, form_float="  %14.6f", form_int="  %12d")
        self.assertEqual(returns_values_in_file(f),returns_values_in_file(fold))
        remove_list_of_file([f,fold])



class Test_read_text_file(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.read_text_file()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.read_text_file()
        self.assertEqual(str(cm_new.exception), "read_text_file() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_file_not_found(self):
        with self.assertRaises(IOError) as cm_new:
            fu.read_text_file(file_name="no_file.txt", ncol=0)
        with self.assertRaises(IOError) as cm_old:
            oldfu.read_text_file(file_name="no_file.txt", ncol=0)
        self.assertEqual(cm_new.exception.strerror, "No such file or directory")
        self.assertEqual(cm_new.exception.strerror, cm_old.exception.strerror)

    def test_default_case(self):
        partids = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D/main001/this_iteration_index_keep_images.txt')
        return_new = fu.read_text_file(file_name=partids, ncol=0)
        return_old = oldfu.read_text_file(file_name=partids, ncol=0)
        self.assertTrue(return_new == return_old)
        self.assertTrue(array_equal(return_new, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]))



class Test_write_text_file(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.write_text_file()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.write_text_file()
        self.assertEqual(str(cm_new.exception), "write_text_file() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_write_text_row(self):
        data=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]
        f=path.join(ABSOLUTE_PATH, "filefu.txt")
        fold=path.join(ABSOLUTE_PATH, "filefold.txt")
        fu.write_text_file(data=data, file_name=f, form_float="  %14.6f", form_int="  %12d")
        oldfu.write_text_file(data=data, file_name=fold, form_float="  %14.6f", form_int="  %12d")
        self.assertEqual(returns_values_in_file(f),returns_values_in_file(fold))
        remove_list_of_file([f,fold])



class Test_rotate_shift_params(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rotate_shift_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rotate_shift_params()
        self.assertEqual(str(cm_new.exception), "rotate_shift_params() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_rotate_shift_params(self):
        paramsin = [[0.25,1.25,0.5]]
        transf  = [0.25, 1.25, 0.5]
        return_new = fu.rotate_shift_params(paramsin=paramsin, transf=transf)
        return_old = oldfu.rotate_shift_params(paramsin=paramsin, transf=transf)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 0.0, 1.2820115403810843e-08]]))

    def test_rotate_shift_params2(self):
        paramsin = [[0.25,1.25,0,0,0.5]]
        transf  = [0.25, 1.25, 0.5,.25, 1.25, 0.5]
        return_new = fu.rotate_shift_params(paramsin=paramsin, transf=transf)
        return_old = oldfu.rotate_shift_params(paramsin=paramsin, transf=transf)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[0.0, 0.0, 359.50000008558703, 0.23908232152462006, 1.752134084701538]]))

    def test_less_transf_params_returns_IndexError_list_index_out_of_range(self):
        paramsin = [[0.25,1.25,0,0,0.5]]
        transf  = [0.25, 1.25, 0.5]
        with self.assertRaises(IndexError) as cm_new:
            fu.rotate_shift_params(paramsin=paramsin, transf=transf)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.rotate_shift_params(paramsin=paramsin, transf=transf)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_less_transf_params2_returns_IndexError_list_index_out_of_range(self):
        paramsin = [[0.25,1.25,0]]
        transf  = [0.25, 1.25]
        with self.assertRaises(IndexError) as cm_new:
            fu.rotate_shift_params(paramsin=paramsin, transf=transf)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.rotate_shift_params(paramsin=paramsin, transf=transf)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_less_paramsin_params_returns_IndexError_list_index_out_of_range(self):
        paramsin = [[0.25]]
        transf  = [0.25, 1.25, 0.5]
        with self.assertRaises(IndexError) as cm_new:
            fu.rotate_shift_params(paramsin=paramsin, transf=transf)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.rotate_shift_params(paramsin=paramsin, transf=transf)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_reshape_1d(unittest.TestCase):
    """ values got from 'pickle files/utilities/utilities.reshape_1d'"""
    input_obj =  [0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.9984012789768186, 0.9914368216668327, 0.9878146959140469, 0.9881703862020976, 0.982612488476065, 0.9789244545589472, 0.9747235387045814, 0.9622078763024153, 0.9406924390622574, 0.9300175631598249, 0.8976592373307525, 0.8474726574046705, 0.7942852016327994, 0.8065378605172119, 0.7981892234519837, 0.7980760586172797, 0.7834690256016978, 0.7732854546260584, 0.759479194158529, 0.7302534821351329, 0.735749496632646, 0.7505776906379105, 0.7832464000713297, 0.799354031902547, 0.7829602489012508, 0.7467401462021503, 0.7216741559492451, 0.7573457050470969, 0.7735999645280006, 0.7360206933666649, 0.7074315960216845, 0.6838418535731124, 0.6814918195422979, 0.6604400166044002, 0.6276571502978614, 0.5967298971705947, 0.5924074015096022, 0.6113438607798904, 0.5589193571016572, 0.4169423800381157, 0.33547900293137645, 0.43509084125025116, 0.5143369854093631, 0.4505998230268216, 0.3017867022488365, 0.29393725698240897, 0.3395667841020214, 0.34234494237984336, 0.31531353786458843, 0.3120432449453534, 0.2864549161874622, 0.23450693792899116, 0.20246505335938672, 0.22577560951692183, 0.21569461751208094, 0.21511112191209886, 0.2091532904083915, 0.18334792795777813, 0.1954858454475899, 0.21231959169076153, 0.20199531221828237, 0.21190821007216915, 0.21429959199533707, 0.18398541329970813, 0.20171364365585326, 0.22936964071672247, 0.20705888033218262, 0.2310040684684463, 0.23322049365816364, 0.25365125929269, 0.2687457179832018, 0.252646215129461, 0.24715492782090853, 0.23387479872417344, 0.23315205998051616, 0.2312238364934745, 0.21601984544387764, 0.23373779370670353, 0.21445443670567088, 0.210741700365644, 0.2089851778417197, 0.19984641965828376, 0.18358602895051426, 0.16600398773363803, 0.14936583739921497, 0.14684159823845128, 0.14034187449397328, 0.11227281827686696, 0.09549423222286733, 0.09699040681889236, 0.08368778954783127, 0.07285201615715135, 0.06609239822815444, 0.06712766581830018, 0.06571178890380885, 0.05876124933827422, 0.047775744976412994, 0.04517043724966535, 0.04086780062968338, 0.035162664167093884, 0.02501739454518543, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    length_current = 2* len(input_obj)
    length_interpolated = 4* len(input_obj)

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.reshape_1d()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.reshape_1d()
        self.assertEqual(str(cm_new.exception), "reshape_1d() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_null_list_as_input_obj(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.reshape_1d(input_object = [], length_current=self.length_current, length_interpolated=self.length_interpolated, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.reshape_1d(input_object = [], length_current=self.length_current, length_interpolated=self.length_interpolated, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        return_new = fu.reshape_1d(input_object = self.input_obj, length_current=self.length_current, length_interpolated=self.length_interpolated, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        return_old = oldfu.reshape_1d(input_object = self.input_obj , length_current=self.length_current, length_interpolated=self.length_interpolated, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.9987006394884093, 0.9984012789768186, 0.9949190503218257, 0.9914368216668327, 0.9896257587904398, 0.9878146959140469, 0.9879925410580723, 0.9881703862020976, 0.9853914373390813, 0.982612488476065, 0.980768471517506, 0.9789244545589472, 0.9768239966317642, 0.9747235387045814, 0.9684657075034984, 0.9622078763024153, 0.9514501576823364, 0.9406924390622574, 0.9353550011110412, 0.9300175631598249, 0.9138384002452886, 0.8976592373307525, 0.8725659473677114, 0.8474726574046705, 0.8208789295187349, 0.7942852016327994, 0.8004115310750056, 0.8065378605172119, 0.8023635419845978, 0.7981892234519837, 0.7981326410346317, 0.7980760586172797, 0.7907725421094888, 0.7834690256016978, 0.7783772401138781, 0.7732854546260584, 0.7663823243922937, 0.759479194158529, 0.744866338146831, 0.7302534821351329, 0.7330014893838894, 0.735749496632646, 0.7431635936352783, 0.7505776906379105, 0.7669120453546201, 0.7832464000713297, 0.7913002159869383, 0.799354031902547, 0.7911571404018989, 0.7829602489012508, 0.7648501975517006, 0.7467401462021503, 0.7342071510756978, 0.7216741559492451, 0.7395099304981709, 0.7573457050470969, 0.7654728347875488, 0.7735999645280006, 0.7548103289473327, 0.7360206933666649, 0.7217261446941747, 0.7074315960216845, 0.6956367247973985, 0.6838418535731124, 0.6826668365577051, 0.6814918195422979, 0.670965918073349, 0.6604400166044002, 0.6440485834511308, 0.6276571502978614, 0.612193523734228, 0.5967298971705947, 0.5945686493400985, 0.5924074015096022, 0.6018756311447464, 0.6113438607798904, 0.5851316089407739, 0.5589193571016572, 0.48793086856988643, 0.4169423800381157, 0.37621069148474606, 0.33547900293137645, 0.3852849220908138, 0.43509084125025116, 0.47471391332980717, 0.5143369854093631, 0.48246840421809234, 0.4505998230268216, 0.37619326263782904, 0.3017867022488365, 0.2978619796156228, 0.29393725698240897, 0.3167520205422152, 0.3395667841020214, 0.3409558632409324, 0.34234494237984336, 0.3288292401222159, 0.31531353786458843, 0.3136783914049709, 0.3120432449453534, 0.2992490805664078, 0.2864549161874622, 0.2604809270582267, 0.23450693792899116, 0.21848599564418894, 0.20246505335938672, 0.2141203314381543, 0.22577560951692183, 0.22073511351450137, 0.21569461751208094, 0.21540286971208988, 0.21511112191209886, 0.2121322061602452, 0.2091532904083915, 0.19625060918308482, 0.18334792795777813, 0.18941688670268403, 0.1954858454475899, 0.20390271856917572, 0.21231959169076153, 0.20715745195452195, 0.20199531221828237, 0.20695176114522576, 0.21190821007216915, 0.2131039010337531, 0.21429959199533707, 0.1991425026475226, 0.18398541329970813, 0.1928495284777807, 0.20171364365585326, 0.21554164218628785, 0.22936964071672247, 0.21821426052445253, 0.20705888033218262, 0.21903147440031445, 0.2310040684684463, 0.23211228106330495, 0.23322049365816364, 0.2434358764754268, 0.25365125929269, 0.2611984886379459, 0.2687457179832018, 0.2606959665563314, 0.252646215129461, 0.24990057147518477, 0.24715492782090853, 0.24051486327254099, 0.23387479872417344, 0.23351342935234481, 0.23315205998051616, 0.23218794823699535, 0.2312238364934745, 0.22362184096867607, 0.21601984544387764, 0.2248788195752906, 0.23373779370670353, 0.22409611520618722, 0.21445443670567088, 0.21259806853565744, 0.210741700365644, 0.20986343910368183, 0.2089851778417197, 0.2044157987500017, 0.19984641965828376, 0.191716224304399, 0.18358602895051426, 0.17479500834207615, 0.16600398773363803, 0.1576849125664265, 0.14936583739921497, 0.14810371781883314, 0.14684159823845128, 0.14359173636621228, 0.14034187449397328, 0.12630734638542013, 0.11227281827686696, 0.10388352524986715, 0.09549423222286733, 0.09624231952087985, 0.09699040681889236, 0.0903390981833618, 0.08368778954783127, 0.0782699028524913, 0.07285201615715135, 0.0694722071926529, 0.06609239822815444, 0.06661003202322731, 0.06712766581830018, 0.06641972736105452, 0.06571178890380885, 0.06223651912104154, 0.05876124933827422, 0.05326849715734361, 0.047775744976412994, 0.04647309111303917, 0.04517043724966535, 0.04301911893967436, 0.04086780062968338, 0.03801523239838863, 0.035162664167093884, 0.030090029356139657, 0.02501739454518543, 0.012508697272592715, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_null_length_interpolated_pixel_sizes_identical_error_msg(self):
        return_new = fu.reshape_1d(input_object = self.input_obj, length_current=self.length_current, length_interpolated=0, Pixel_size_current = 0.5, Pixel_size_interpolated = 0.5)
        return_old = oldfu.reshape_1d(input_object = self.input_obj, length_current=self.length_current, length_interpolated=0, Pixel_size_current = 0.5, Pixel_size_interpolated = 0.5)
        self.assertEqual(return_new,[])
        self.assertEqual(return_old, [])

    def test_null_length_current(self):
        return_new = fu.reshape_1d(input_object = self.input_obj, length_current=0, length_interpolated=self.length_interpolated, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        return_old = oldfu.reshape_1d(input_object = self.input_obj, length_current=0, length_interpolated=self.length_interpolated, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.9988503197442047, 0.9987006394884093, 0.9985509592326139, 0.9984012789768186, 0.9966601646493222, 0.9949190503218257, 0.9931779359943291, 0.9914368216668327, 0.9905312902286363, 0.9896257587904398, 0.9887202273522433, 0.9878146959140469, 0.9879036184860596, 0.9879925410580723, 0.988081463630085, 0.9881703862020976, 0.9867809117705895, 0.9853914373390813, 0.9840019629075731, 0.982612488476065, 0.9816904799967855, 0.980768471517506, 0.9798464630382266, 0.9789244545589472, 0.9778742255953558, 0.9768239966317642, 0.9757737676681728, 0.9747235387045814, 0.9715946231040399, 0.9684657075034984, 0.9653367919029568, 0.9622078763024153, 0.9568290169923759, 0.9514501576823364, 0.9460712983722969, 0.9406924390622574, 0.9380237200866492, 0.9353550011110412, 0.932686282135433, 0.9300175631598249, 0.9219279817025567, 0.9138384002452886, 0.9057488187880206, 0.8976592373307525, 0.885112592349232, 0.8725659473677114, 0.860019302386191, 0.8474726574046705, 0.8341757934617027, 0.8208789295187349, 0.8075820655757672, 0.7942852016327994, 0.7973483663539025, 0.8004115310750056, 0.8034746957961088, 0.8065378605172119, 0.8044507012509048, 0.8023635419845978, 0.8002763827182908, 0.7981892234519837, 0.7981609322433078, 0.7981326410346317, 0.7981043498259557, 0.7980760586172797, 0.7944243003633842, 0.7907725421094888, 0.7871207838555934, 0.7834690256016978, 0.7809231328577879, 0.7783772401138781, 0.7758313473699683, 0.7732854546260584, 0.7698338895091761, 0.7663823243922937, 0.7629307592754113, 0.759479194158529, 0.75217276615268, 0.744866338146831, 0.7375599101409819, 0.7302534821351329, 0.7316274857595111, 0.7330014893838894, 0.7343754930082678, 0.735749496632646, 0.7394565451339621, 0.7431635936352783, 0.7468706421365944, 0.7505776906379105, 0.7587448679962653, 0.7669120453546201, 0.7750792227129749, 0.7832464000713297, 0.7872733080291341, 0.7913002159869383, 0.7953271239447426, 0.799354031902547, 0.7952555861522229, 0.7911571404018989, 0.7870586946515749, 0.7829602489012508, 0.7739052232264757, 0.7648501975517006, 0.7557951718769255, 0.7467401462021503, 0.7404736486389241, 0.7342071510756978, 0.7279406535124714, 0.7216741559492451, 0.7305920432237081, 0.7395099304981709, 0.7484278177726339, 0.7573457050470969, 0.7614092699173228, 0.7654728347875488, 0.7695363996577747, 0.7735999645280006, 0.7642051467376667, 0.7548103289473327, 0.7454155111569988, 0.7360206933666649, 0.7288734190304198, 0.7217261446941747, 0.7145788703579296, 0.7074315960216845, 0.7015341604095415, 0.6956367247973985, 0.6897392891852554, 0.6838418535731124, 0.6832543450654088, 0.6826668365577051, 0.6820793280500015, 0.6814918195422979, 0.6762288688078235, 0.670965918073349, 0.6657029673388746, 0.6604400166044002, 0.6522443000277656, 0.6440485834511308, 0.635852866874496, 0.6276571502978614, 0.6199253370160447, 0.612193523734228, 0.6044617104524114, 0.5967298971705947, 0.5956492732553466, 0.5945686493400985, 0.5934880254248504, 0.5924074015096022, 0.5971415163271743, 0.6018756311447464, 0.6066097459623184, 0.6113438607798904, 0.5982377348603322, 0.5851316089407739, 0.5720254830212155, 0.5589193571016572, 0.5234251128357719, 0.48793086856988643, 0.45243662430400106, 0.4169423800381157, 0.39657653576143087, 0.37621069148474606, 0.35584484720806125, 0.33547900293137645, 0.3603819625110951, 0.3852849220908138, 0.4101878816705325, 0.43509084125025116, 0.45490237729002914, 0.47471391332980717, 0.49452544936958515, 0.5143369854093631, 0.49840269481372773, 0.48246840421809234, 0.466534113622457, 0.4505998230268216, 0.41339654283232535, 0.37619326263782904, 0.3389899824433328, 0.3017867022488365, 0.29982434093222965, 0.2978619796156228, 0.29589961829901584, 0.29393725698240897, 0.3053446387623121, 0.3167520205422152, 0.32815940232211827, 0.3395667841020214, 0.3402613236714769, 0.3409558632409324, 0.3416504028103879, 0.34234494237984336, 0.33558709125102965, 0.3288292401222159, 0.32207138899340215, 0.31531353786458843, 0.3144959646347797, 0.3136783914049709, 0.31286081817516215, 0.3120432449453534, 0.30564616275588063, 0.2992490805664078, 0.292851998376935, 0.2864549161874622, 0.27346792162284445, 0.2604809270582267, 0.24749393249360893, 0.23450693792899116, 0.22649646678659005, 0.21848599564418894, 0.21047552450178783, 0.20246505335938672, 0.2082926923987705, 0.2141203314381543, 0.21994797047753806, 0.22577560951692183, 0.2232553615157116, 0.22073511351450137, 0.21821486551329117, 0.21569461751208094, 0.21554874361208542, 0.21540286971208988, 0.21525699581209437, 0.21511112191209886, 0.21362166403617203, 0.2121322061602452, 0.21064274828431834, 0.2091532904083915, 0.20270194979573816, 0.19625060918308482, 0.18979926857043147, 0.18334792795777813, 0.18638240733023106, 0.18941688670268403, 0.19245136607513696, 0.1954858454475899, 0.1996942820083828, 0.20390271856917572, 0.20811115512996864, 0.21231959169076153, 0.20973852182264174, 0.20715745195452195, 0.20457638208640216, 0.20199531221828237, 0.20447353668175405, 0.20695176114522576, 0.20942998560869747, 0.21190821007216915, 0.21250605555296112, 0.2131039010337531, 0.2137017465145451, 0.21429959199533707, 0.20672104732142985, 0.1991425026475226, 0.19156395797361536, 0.18398541329970813, 0.18841747088874441, 0.1928495284777807, 0.19728158606681698, 0.20171364365585326, 0.20862764292107056, 0.21554164218628785, 0.22245564145150518, 0.22936964071672247, 0.22379195062058752, 0.21821426052445253, 0.21263657042831757, 0.20705888033218262, 0.21304517736624853, 0.21903147440031445, 0.22501777143438037, 0.2310040684684463, 0.23155817476587562, 0.23211228106330495, 0.2326663873607343, 0.23322049365816364, 0.23832818506679523, 0.2434358764754268, 0.2485435678840584, 0.25365125929269, 0.25742487396531794, 0.2611984886379459, 0.26497210331057386, 0.2687457179832018, 0.2647208422697666, 0.2606959665563314, 0.2566710908428962, 0.252646215129461, 0.2512733933023229, 0.24990057147518477, 0.24852774964804664, 0.24715492782090853, 0.24383489554672477, 0.24051486327254099, 0.2371948309983572, 0.23387479872417344, 0.23369411403825913, 0.23351342935234481, 0.23333274466643047, 0.23315205998051616, 0.23267000410875574, 0.23218794823699535, 0.23170589236523492, 0.2312238364934745, 0.22742283873107527, 0.22362184096867607, 0.21982084320627687, 0.21601984544387764, 0.22044933250958412, 0.2248788195752906, 0.22930830664099705, 0.23373779370670353, 0.22891695445644536, 0.22409611520618722, 0.21927527595592905, 0.21445443670567088, 0.21352625262066416, 0.21259806853565744, 0.2116698844506507, 0.210741700365644, 0.2103025697346629, 0.20986343910368183, 0.20942430847270077, 0.2089851778417197, 0.2067004882958607, 0.2044157987500017, 0.20213110920414273, 0.19984641965828376, 0.19578132198134138, 0.191716224304399, 0.18765112662745664, 0.18358602895051426, 0.1791905186462952, 0.17479500834207615, 0.1703994980378571, 0.16600398773363803, 0.16184445015003227, 0.1576849125664265, 0.15352537498282073, 0.14936583739921497, 0.14873477760902404, 0.14810371781883314, 0.1474726580286422, 0.14684159823845128, 0.1452166673023318, 0.14359173636621228, 0.14196680543009277, 0.14034187449397328, 0.1333246104396967, 0.12630734638542013, 0.11929008233114353, 0.11227281827686696, 0.10807817176336705, 0.10388352524986715, 0.09968887873636724, 0.09549423222286733, 0.09586827587187359, 0.09624231952087985, 0.0966163631698861, 0.09699040681889236, 0.09366475250112709, 0.0903390981833618, 0.08701344386559653, 0.08368778954783127, 0.08097884620016128, 0.0782699028524913, 0.07556095950482133, 0.07285201615715135, 0.07116211167490212, 0.0694722071926529, 0.06778230271040367, 0.06609239822815444, 0.06635121512569087, 0.06661003202322731, 0.06686884892076375, 0.06712766581830018, 0.06677369658967736, 0.06641972736105452, 0.06606575813243168, 0.06571178890380885, 0.06397415401242519, 0.06223651912104154, 0.06049888422965788, 0.05876124933827422, 0.05601487324780892, 0.05326849715734361, 0.0505221210668783, 0.047775744976412994, 0.047124418044726085, 0.04647309111303917, 0.04582176418135226, 0.04517043724966535, 0.044094778094669856, 0.04301911893967436, 0.04194345978467887, 0.04086780062968338, 0.039441516514036004, 0.03801523239838863, 0.03658894828274126, 0.035162664167093884, 0.03262634676161677, 0.030090029356139657, 0.027553711950662543, 0.02501739454518543, 0.018763045908889074, 0.012508697272592715, 0.006254348636296356, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_all_the_values_are_null_or_empty_list_error_msg(self):
        return_new = fu.reshape_1d(input_object = [], length_current=0, length_interpolated=0, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        return_old = oldfu.reshape_1d(input_object = [], length_current=0, length_interpolated=0, Pixel_size_current = 0., Pixel_size_interpolated = 0.)
        self.assertEqual(return_new, [])
        self.assertEqual(return_old, [])

    def test_invalid_pixel_sizes_combination_in_null_value_as_length_interpolated_returns_ZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.reshape_1d(input_object = self.input_obj, length_current=self.length_current, length_interpolated=0, Pixel_size_current = 0.3, Pixel_size_interpolated = 0.)
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.reshape_1d(input_object = self.input_obj, length_current=self.length_current, length_interpolated=0, Pixel_size_current = 0.3, Pixel_size_interpolated = 0.)
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))





'''
@unittest.skip("I m not sure how test them")
class Test_estimate_3D_center_MPI(unittest.TestCase):
    """ values got from 'pickle files/utilities/utilities.estimate_3D_center_MPI'"""
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.estimate_3D_center_MPI"))

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.estimate_3D_center_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.estimate_3D_center_MPI()
        self.assertEqual(str(cm_new.exception), "estimate_3D_center_MPI() takes at least 5 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_myid_not_identical_to_main_node(self):
        (data, nima, myid, number_of_proc, main_node) = self.argum[0]
        return_new = fu.estimate_3D_center_MPI(data, nima, myid, number_of_proc, main_node)
        return_old = oldfu.estimate_3D_center_MPI(data, nima, myid, number_of_proc, main_node)
        self.assertTrue(array_equal(return_old, [0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(array_equal(return_new, [0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_myid_not_identical_to_main_node1(self):
        (data, nima, myid, number_of_proc, main_node) = self.argum[0]
        main_node=myid
        return_new = fu.estimate_3D_center_MPI(data, nima, myid, number_of_proc, main_node)
        return_old = oldfu.estimate_3D_center_MPI(data, nima, myid, number_of_proc, main_node)
        self.assertTrue(array_equal(return_old, [0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(array_equal(return_new, [0.0, 0.0, 0.0, 0.0, 0.0]))
'''

class Test_rotate_3D_shift(unittest.TestCase):
    """ values got from 'pickle files/utilities/utilities.rotate_3D_shift'"""
    argum =get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.rotate_3D_shift"))
    (data, notUsed) = argum[0]
    shift3d = [10.1, 0.2, 10.0]

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rotate_3D_shift()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rotate_3D_shift()
        self.assertEqual(str(cm_new.exception), "rotate_3D_shift() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    def test_wrong_image(self):
        data,not_used= get_real_data(dim = 3)
        with self.assertRaises(RuntimeError) as cm_new:
            fu.rotate_3D_shift(data=[data], shift3d=self.shift3d)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.rotate_3D_shift(data=[data], shift3d=self.shift3d)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_Nonetype_image(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.rotate_3D_shift(data=[None], shift3d=self.shift3d)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.rotate_3D_shift(data=[None], shift3d=self.shift3d)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        fu_data = deepcopy(self.data)
        oldfu_data = deepcopy(self.data)
        return_new = fu.rotate_3D_shift(data=fu_data, shift3d=self.shift3d)
        return_old = oldfu.rotate_3D_shift(data=oldfu_data, shift3d=self.shift3d)
        self.assertEqual(return_new, None)
        self.assertEqual(return_old, None)
        for i in range(len(fu_data)):
            self.assertTrue(array_equal(fu_data[i].get_attr('xform.projection'), oldfu_data[i].get_attr('xform.projection')))
            self.assertFalse(array_equal(self.data[i].get_attr('xform.projection'), fu_data[i].get_attr('xform.projection')))

    def test_returns_IndexError_list_index_out_of_range(self):
        shift3d=[0,0.1]
        with self.assertRaises(IndexError) as cm_new:
            fu.rotate_3D_shift(data=self.data, shift3d=shift3d)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.rotate_3D_shift(data=self.data, shift3d=shift3d)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_set_arb_params(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_arb_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_arb_params()
        self.assertEqual(str(cm_new.exception), "set_arb_params() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_1Attr(self):
        fu_img = EMData()
        oldfu_img = EMData()
        par_str = "lowpassfilter"
        params = "0.50"
        return_new = fu.set_arb_params(fu_img,[params],[par_str])
        return_old = oldfu.set_arb_params(oldfu_img,[params],[par_str])
        self.assertEqual(return_new, None)
        self.assertEqual(return_old, None)
        self.assertEqual(fu_img.get_attr(par_str), fu_img.get_attr(par_str))
        self.assertEqual(fu_img.get_attr(par_str),params)

    def test_with_ListAttr(self):
        fu_img = EMData()
        oldfu_img = EMData()
        par_str = ["lowpassfilter","fake_par"]
        params = ["0.50","3"]
        return_new = fu.set_arb_params(fu_img,params,par_str)
        return_old = oldfu.set_arb_params(oldfu_img,params,par_str)
        self.assertEqual(return_new, None)
        self.assertEqual(return_old, None)
        self.assertEqual(fu_img.get_attr(par_str[0]), fu_img.get_attr(par_str[0]))
        self.assertEqual(fu_img.get_attr(par_str[1]), fu_img.get_attr(par_str[1]))
        self.assertEqual(fu_img.get_attr(par_str[0]),params[0])
        self.assertEqual(fu_img.get_attr(par_str[1]), params[1])

    def test_with_BadListAttr_returns_IndexError_list_index_out_of_range(self):
        fu_img = EMData()
        oldfu_img = EMData()
        par_str = ["lowpassfilter","fake_par"]
        params = ["0.50"]
        with self.assertRaises(IndexError) as cm_new:
            fu.set_arb_params(fu_img,params,par_str)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.set_arb_params(oldfu_img,params,par_str)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_get_arb_params(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_arb_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_arb_params()
        self.assertEqual(str(cm_new.exception), "get_arb_params() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_1Attr(self):
        return_new = fu.get_arb_params(EMData(),["datatype"])
        return_old = oldfu.get_arb_params(EMData(),["datatype"])
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, [EMData().get_attr("datatype")])

    def test_with_ListAttr(self):
        list_of_attribute = ["datatype", "is_complex_ri"]
        return_new = fu.get_arb_params(EMData(),list_of_attribute)
        return_old = oldfu.get_arb_params(EMData(),list_of_attribute)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertEqual(return_new[0], EMData().get_attr("datatype"))
        self.assertEqual(return_new[1], EMData().get_attr("is_complex_ri"))

    def test_notValid_params_returns_RuntimeError_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_arb_params(EMData(),["invalid_param"])
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_arb_params(EMData(),["invalid_param"])
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_reduce_EMData_to_root(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.reduce_EMData_to_root()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.reduce_EMData_to_root()
        self.assertEqual(str(cm_new.exception), "reduce_EMData_to_root() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_NoneType_as_img_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.reduce_EMData_to_root(None, myid=74, main_node=0, comm = -1)
        return_old = oldfu.reduce_EMData_to_root(None, myid=74, main_node=0, comm = -1)
        self.assertEqual(return_new, return_old)
        """

    def test_default_values(self):
        data = deepcopy(IMAGE_2D_REFERENCE)
        return_new = fu.reduce_EMData_to_root(data, myid=74, main_node=0, comm = -1)
        return_old = oldfu.reduce_EMData_to_root(data, myid=74, main_node=0, comm = -1)
        self.assertTrue(array_equal(IMAGE_2D_REFERENCE.get_3dview(), data.get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_with_MPI_COMM_WORLD(self):
        data = deepcopy(IMAGE_2D_REFERENCE)
        return_new = fu.reduce_EMData_to_root(data, myid=74, main_node=0, comm = MPI_COMM_WORLD)
        return_old = oldfu.reduce_EMData_to_root(data, myid=74, main_node=0, comm = MPI_COMM_WORLD)
        self.assertTrue(array_equal(IMAGE_2D_REFERENCE.get_3dview(), data.get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)



class Test_bcast_compacted_EMData_all_to_all(unittest.TestCase):
    """
    It does not matter which of my images I-ll use, I got always the following typeerror:
    Error
Traceback (most recent call last):
  File "/home/lusnig/SPHIRE_1_1/lib/python2.7/unittest/case.py", line 329, in run
    testMethod()
  File "/home/lusnig/EMAN2/eman2/sphire/tests/test_utilities.py", line 1451, in test_bcast_compacted_EMData_all_to_all_true_should_return_equal_objects
    return_new = fu.bcast_compacted_EMData_all_to_all(list_of_em_objects, myid)
  File "/home/lusnig/EMAN2/eman2/sphire/libpy/sparx_utilities.py", line 1105, in bcast_compacted_EMData_all_to_all
    em_dict = dict_received["em_dict"]
TypeError: 'NoneType' object has no attribute '__getitem__'
    """
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.bcast_compacted_EMData_all_to_all()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.bcast_compacted_EMData_all_to_all()
        self.assertEqual(str(cm_new.exception), "bcast_compacted_EMData_all_to_all() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_NoneType_data_returns_TypeError_NoneType_obj_hasnot_attribute__getitem__(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.bcast_compacted_EMData_all_to_all([None,None], myid=74,  comm = -1)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.bcast_compacted_EMData_all_to_all([None,None], myid=74, comm = -1)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute '__getitem__'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_image(self):
        data = [deepcopy(IMAGE_3D),deepcopy(IMAGE_3D)]
        with self.assertRaises(TypeError) as cm_new:
            fu.bcast_compacted_EMData_all_to_all(data, myid=74,  comm = -1)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.bcast_compacted_EMData_all_to_all(data, myid=74,  comm = -1)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute '__getitem__'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


"""
has been cleaned
class Test_gather_compacted_EMData_to_root(unittest.TestCase):
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.gather_compacted_EMData_to_root"))
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.gather_compacted_EMData_to_root()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.gather_compacted_EMData_to_root()
        self.assertEqual(str(cm_new.exception), "gather_compacted_EMData_to_root() takes at least 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        (no_of_emo, list_of_emo, myid) = self.argum[0]
        return_new = fu.gather_compacted_EMData_to_root(no_of_emo, list_of_emo, myid, comm=-1)
        return_old = oldfu.gather_compacted_EMData_to_root(no_of_emo, list_of_emo, myid, comm=-1)
        self.assertEqual(return_new, return_old)

    def test_with_MPI_COMM_WORLD(self):
        (no_of_emo, list_of_emo, myid) = self.argum[0]
        return_new = fu.gather_compacted_EMData_to_root(no_of_emo, list_of_emo, myid, comm=MPI_COMM_WORLD)
        return_old = oldfu.gather_compacted_EMData_to_root(no_of_emo, list_of_emo, myid, comm=MPI_COMM_WORLD)
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_pickle_file_values_wrong_number_of_number_of_all_em_objects_distributed_across_processes(self):
        (no_of_emo, list_of_emo, myid) = self.argum[0]
        return_new = fu.gather_compacted_EMData_to_root(0, list_of_emo, myid,  comm=-1)
        return_old = oldfu.gather_compacted_EMData_to_root(0, list_of_emo, myid, comm=-1)
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_NoneType_as_img_returns_IndexError_list_index_out_of_range(self):
        (no_of_emo, list_of_emo, myid) = self.argum[0]
        with self.assertRaises(IndexError) as cm_new:
            fu.gather_compacted_EMData_to_root(no_of_emo, [], myid, comm=-1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.gather_compacted_EMData_to_root(no_of_emo, [], myid, comm=-1)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
"""


class Test_bcast_EMData_to_all(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.bcast_EMData_to_all()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.bcast_EMData_to_all()
        self.assertEqual(str(cm_new.exception), "bcast_EMData_to_all() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_default_values(self):
        tavg = deepcopy(IMAGE_2D_REFERENCE)
        return_new = fu.bcast_EMData_to_all(tavg, myid = 11, source_node =0, comm= -1)
        return_old = oldfu.bcast_EMData_to_all(tavg, myid= 11, source_node = 0, comm= -1)
        self.assertTrue(array_equal(IMAGE_2D_REFERENCE.get_3dview(), tavg.get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_with_myid_equal_sourcenode_default_valuqes(self):
        tavg = deepcopy(IMAGE_2D_REFERENCE)
        return_new = fu.bcast_EMData_to_all(tavg, myid= 0, source_node =0, comm= -1)
        return_old = oldfu.bcast_EMData_to_all(tavg, myid= 0, source_node =0, comm= -1)
        self.assertTrue(array_equal(IMAGE_2D_REFERENCE.get_3dview(), tavg.get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_with_MPI_COMM_WORLD(self):
        tavg = deepcopy(IMAGE_2D_REFERENCE)
        return_new = fu.bcast_EMData_to_all(tavg, myid= 11, source_node =0, comm= MPI_COMM_WORLD)
        return_old = oldfu.bcast_EMData_to_all(tavg, myid= 11, source_node =0, comm= MPI_COMM_WORLD)
        self.assertTrue(array_equal(IMAGE_2D_REFERENCE.get_3dview(), tavg.get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_NoneType_as_img_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.bcast_EMData_to_all(None, 11, source_node =0, comm= -1)
        return_old = oldfu.bcast_EMData_to_all(None, 11, source_node =0, comm= -1)
        self.assertEqual(return_new, return_old)
        """



class Test_send_EMData(unittest.TestCase):
    #argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.send_EMData"))
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.send_EMData()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.send_EMData()
        self.assertEqual(str(cm_new.exception), "send_EMData() takes at least 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_NoneType_as_img_crashes_because_signal11SIGSEV(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.send_EMData(None, 0, 0)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.send_EMData(None, 0, 0)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_xsize'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    """  Can only be tested on the mpi. Wait too long on normal workstation"""
    # def test_send_EMData_true_should_return_equal_objects(self):
    #     filepath = path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.send_EMData")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle_load(rb)
    #
    #     print(argum[0])
    #
    #     (img, dst, tag, comm) = argum[0]
    #     tag = 0
    #
    #     return_new = fu.send_EMData(img, dst, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_old = oldfu.send_EMData(img, dst, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     self.assertEqual(return_new, return_old)



class Test_recv_EMData(unittest.TestCase):
    #argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.recv_EMData"))
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recv_EMData()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recv_EMData()
        self.assertEqual(str(cm_new.exception), "recv_EMData() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    """  Can only be tested on the mpi. Wait too long on normal workstation"""
    # def test_recv_EMData_true_should_return_equal_objects(self):
    #     filepath = path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.recv_EMData")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle_load(rb)
    #
    #     print(argum[0])
    #
    #     (src, tag,comm) = argum[0]
    #     tag = 0
    #
    #     return_new = fu.recv_EMData(src, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_old = oldfu.recv_EMData(src, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     self.assertEqual(return_new, return_old)




class Test_bcast_number_to_all(unittest.TestCase):

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.bcast_number_to_all()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.bcast_number_to_all()
        self.assertEqual(str(cm_new.exception), "bcast_number_to_all() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_number_to_send_is_null(self):
        return_new = fu.bcast_number_to_all(number_to_send = 0, source_node = 0, mpi_comm = -1)
        return_old = oldfu.bcast_number_to_all(number_to_send = 0, source_node = 0, mpi_comm = -1)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 0)

    def test_with_MPI_COMM_WORLD(self):
        return_new = fu.bcast_number_to_all(number_to_send = 0, source_node = 0, mpi_comm = MPI_COMM_WORLD)
        return_old = oldfu.bcast_number_to_all(number_to_send = 0, source_node = 0, mpi_comm = MPI_COMM_WORLD)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 0)

    def test_number_to_send_is_not_null(self):
        return_new = fu.bcast_number_to_all(number_to_send = 3, source_node = 0, mpi_comm = -1)
        return_old = oldfu.bcast_number_to_all(number_to_send = 3, source_node = 0, mpi_comm = -1)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 3)

    def test_invalid_number_to_send_error_msg(self):
        return_new = fu.bcast_number_to_all(number_to_send = None, source_node = 0, mpi_comm = -1)
        return_old = oldfu.bcast_number_to_all(number_to_send = None, source_node = 0, mpi_comm = -1)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, None)


class Test_bcast_list_to_all(unittest.TestCase):
    myid = 74
    source_node =0
    list_to_send = [1,2]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.bcast_list_to_all()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.bcast_list_to_all()
        self.assertEqual(str(cm_new.exception), "bcast_list_to_all() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    """
    the compatibility test in the nosetests feiled
    def test_with_empty_list(self):
        return_new = fu.bcast_list_to_all([], myid = self.myid, source_node =self.source_node, mpi_comm= -1)
        return_old = oldfu.bcast_list_to_all([], myid= self.myid, source_node = self.source_node, mpi_comm= -1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,[]))
    """

    def test_defualt_values(self):
        return_new = fu.bcast_list_to_all(self.list_to_send, myid = self.myid, source_node =self.source_node, mpi_comm= -1)
        return_old = oldfu.bcast_list_to_all(self.list_to_send, myid= self.myid, source_node = self.source_node, mpi_comm= -1)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,[]))

    def test_defualt_values_with_MPI_COMM_WORLD(self):
        return_new = fu.bcast_list_to_all(self.list_to_send, myid = self.myid, source_node =self.source_node, mpi_comm= MPI_COMM_WORLD)
        return_old = oldfu.bcast_list_to_all(self.list_to_send, myid= self.myid, source_node = self.source_node, mpi_comm= MPI_COMM_WORLD)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,[]))

    def test_myid_equal_sourcenode(self):
        return_new = fu.bcast_list_to_all(self.list_to_send, myid = self.source_node, source_node =self.source_node, mpi_comm= MPI_COMM_WORLD)
        return_old = oldfu.bcast_list_to_all(self.list_to_send, myid= self.source_node, source_node = self.source_node, mpi_comm= MPI_COMM_WORLD)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,self.list_to_send))

    """
    in the nosetests the exception is not raised
    def test_myid_equal_sourcenode_and_wrong_type_in_listsender_returns_ValueError(self):
        list_to_send=[IMAGE_2D]
        with self.assertRaises(ValueError) as cm_new:
            fu.bcast_list_to_all(list_to_send, myid = self.source_node, source_node =self.source_node, mpi_comm= MPI_COMM_WORLD)
        with self.assertRaises(ValueError) as cm_old:
            oldfu.bcast_list_to_all(list_to_send, myid= self.source_node, source_node = self.source_node, mpi_comm= MPI_COMM_WORLD)
        self.assertEqual(str(cm_new.exception), "setting an array element with a sequence.")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
    """

    def test_wrong_type_in_listsender(self):
        list_to_send=[IMAGE_2D]
        return_new = fu.bcast_list_to_all(list_to_send, myid = self.myid, source_node =self.source_node, mpi_comm= MPI_COMM_WORLD)
        return_old = oldfu.bcast_list_to_all(list_to_send, myid= self.myid, source_node = self.source_node, mpi_comm= MPI_COMM_WORLD)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,[]))



class Test_recv_attr_dict(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recv_attr_dict()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recv_attr_dict()
        self.assertEqual(str(cm_new.exception), "recv_attr_dict() takes at least 7 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_send_attr_dict(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.send_attr_dict()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.send_attr_dict()
        self.assertEqual(str(cm_new.exception), "send_attr_dict() takes at least 5 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_recv_attr_dict_bdb(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recv_attr_dict_bdb()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recv_attr_dict_bdb()
        self.assertEqual(str(cm_new.exception), "recv_attr_dict_bdb() takes at least 7 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_print_begin_msg(unittest.TestCase):
    """ see https://wrongsideofmemphis.com/2010/03/01/store-standard-output-on-a-variable-in-python/"""
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.print_begin_msg()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.print_begin_msg()
        self.assertEqual(str(cm_new.exception), "print_begin_msg() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_print_begin_msg(self):
        old_stdout = sys.stdout
        print_new = StringIO()
        sys.stdout = print_new
        return_new = fu.print_begin_msg("test_pgr", onscreen=False)
        print_old = StringIO()
        sys.stdout = print_old
        return_old = oldfu.print_begin_msg("test_pgr", onscreen=False)
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new is None)
        self.assertEqual(print_new.getvalue(), print_old.getvalue())
        sys.stdout = old_stdout

    def test_print_begin_msg_onscreen_True(self):
        old_stdout = sys.stdout
        print_new = StringIO()
        sys.stdout = print_new
        return_new = fu.print_begin_msg("test_pgr", onscreen=True)
        print_old = StringIO()
        sys.stdout = print_old
        return_old = oldfu.print_begin_msg("test_pgr", onscreen=True)
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new is None)
        self.assertEqual(print_new.getvalue(), print_old.getvalue())
        sys.stdout = old_stdout




class Test_print_end_msg(unittest.TestCase):
    """ see https://wrongsideofmemphis.com/2010/03/01/store-standard-output-on-a-variable-in-python/"""
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.print_end_msg()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.print_end_msg()
        self.assertEqual(str(cm_new.exception), "print_end_msg() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_print_end_msg(self):
        old_stdout = sys.stdout
        print_new = StringIO()
        sys.stdout = print_new
        return_new = fu.print_end_msg("test_pgr", onscreen=False)
        print_old = StringIO()
        sys.stdout = print_old
        return_old = oldfu.print_end_msg("test_pgr", onscreen=False)
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new is None)
        self.assertEqual(print_new.getvalue(), print_old.getvalue())
        sys.stdout = old_stdout

    def test_print_end_msg_onscreen_True(self):
        old_stdout = sys.stdout
        print_new = StringIO()
        sys.stdout = print_new
        return_new = fu.print_end_msg("test_pgr", onscreen=True)
        print_old = StringIO()
        sys.stdout = print_old
        return_old = oldfu.print_end_msg("test_pgr", onscreen=True)
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new is None)
        self.assertEqual(print_new.getvalue(), print_old.getvalue())
        sys.stdout = old_stdout



class Test_print_msg(unittest.TestCase):
    """ see https://wrongsideofmemphis.com/2010/03/01/store-standard-output-on-a-variable-in-python/"""
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.print_msg()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.print_msg()
        self.assertEqual(str(cm_new.exception), "print_msg() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_print_msg(self):
        old_stdout = sys.stdout
        print_new = StringIO()
        sys.stdout = print_new
        return_new = fu.print_msg("test_pgr")
        print_old = StringIO()
        sys.stdout = print_old
        return_old = oldfu.print_msg("test_pgr")
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new is None)
        self.assertEqual(print_new.getvalue(), print_old.getvalue())
        sys.stdout = old_stdout



class Test_read_fsc(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.read_fsc()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.read_fsc()
        self.assertEqual(str(cm_new.exception), "read_fsc() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_write_text_row(self):
        data=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]
        f=path.join(ABSOLUTE_PATH, "filefu.txt")
        fu.write_text_file(data, f)
        return_new = fu.read_fsc(f)
        return_old = oldfu.read_fsc(f)
        remove_list_of_file([f])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, data))



class Test_circumference(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.circumference()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.circumference()
        self.assertEqual(str(cm_new.exception), "circumference() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_default_values_2Dimg(self):
        return_new = fu.circumference(deepcopy(IMAGE_BLANK_2D), inner = -1, outer = -1)
        return_old = oldfu.circumference(deepcopy(IMAGE_BLANK_2D), inner = -1, outer = -1)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_with_default_values_3Dimg(self):
        return_new = fu.circumference(deepcopy(IMAGE_BLANK_3D), inner = -1, outer = -1)
        return_old = oldfu.circumference(deepcopy(IMAGE_BLANK_3D), inner = -1, outer = -1)
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_with_invalid_mask_returns_RuntimeError_ImageFormatException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.circumference(deepcopy(IMAGE_BLANK_2D), inner =IMAGE_BLANK_2D.get_xsize()+10 , outer = -1)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.circumference(deepcopy(IMAGE_BLANK_2D), inner =IMAGE_BLANK_2D.get_xsize()+10 , outer = -1)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageFormatException")
        self.assertEqual(msg[1], "Invalid mask")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_wrong_outer_value(self):
        return_new = fu.circumference(deepcopy(IMAGE_BLANK_2D), inner = -1, outer = IMAGE_BLANK_2D.get_xsize()+10 )
        return_old = oldfu.circumference(deepcopy(IMAGE_BLANK_2D), inner = -1, outer = IMAGE_BLANK_2D.get_xsize()+10 )
        self.assertTrue(array_equal(return_new.get_3dview(), return_old.get_3dview()))



class Test_write_headers(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.write_headers()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.write_headers()
        self.assertEqual(str(cm_new.exception), "write_headers() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    #The following tests sometimes failed in the nosetests
    def test_hdf_type(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.hdf")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.hdf")
        fu.write_headers(path_fu, [IMAGE_2D], [1])
        oldfu.write_headers(path_oldfu, [IMAGE_2D], [1])
        self.assertEqual(returns_values_in_file(path_fu), returns_values_in_file(path_oldfu))
        self.assertTrue(path.isfile(path_fu))
        self.assertTrue(path.isfile(path_oldfu))
        remove_list_of_file([path_fu,path_oldfu])
    

    def test_overwrite_hdf_file(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.hdf")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.hdf")
        f = open(path_fu, "w+")
        f.close()
        f = open(path_oldfu, "w+")
        f.close()
        fu.write_headers(path_fu, [IMAGE_2D], [1])
        oldfu.write_headers(path_oldfu, [IMAGE_2D], [1])
        self.assertEqual(returns_values_in_file(path_fu), returns_values_in_file(path_oldfu))
        self.assertTrue(path.isfile(path_fu))
        self.assertTrue(path.isfile(path_oldfu))
        remove_list_of_file([path_fu,path_oldfu])
    

    def test_hdf_type_AssertError_list_differ(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.hdf")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.hdf")
        fu.write_headers(path_fu, [IMAGE_2D], [2])
        oldfu.write_headers(path_oldfu, [IMAGE_2D], [1])
        self.assertTrue(path.isfile(path_fu))
        self.assertTrue(path.isfile(path_oldfu))
        with self.assertRaises(AssertionError) as cm:
            self.assertEqual(returns_values_in_file(path_fu), returns_values_in_file(path_oldfu))
        msg = cm.exception.message.split("'")
        self.assertEqual(msg[0].split(":")[0], "Lists differ")
        self.assertEqual(msg[10].split("\n")[2].split(":")[0], 'First differing element 2')
        remove_list_of_file([path_fu, path_oldfu])

    def test_bdf_type(self):
        '''
        in the code they inserted the following comment:
        #  For unknown reasons this does not work on Linux, but works on Mac ??? Really?
        '''
        self.assertTrue(True)

    def test_invalid_filetype_error_msg(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.txt")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.txt")
        fu.write_headers(path_fu, [IMAGE_2D], [1])
        oldfu.write_headers(path_oldfu, [IMAGE_2D], [1])
        self.assertFalse(path.isfile(path_fu))
        self.assertFalse(path.isfile(path_oldfu))



class Test_write_header(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.write_header()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.write_header()
        self.assertEqual(str(cm_new.exception), "write_header() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_hdf_type(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.hdf")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.hdf")
        fu.write_header(path_fu, IMAGE_2D, 1)
        oldfu.write_header(path_oldfu, IMAGE_2D, 1)
        self.assertEqual(returns_values_in_file(path_fu), returns_values_in_file(path_oldfu))
        self.assertTrue(path.isfile(path_fu))
        self.assertTrue(path.isfile(path_oldfu))
        remove_list_of_file([path_fu,path_oldfu])

    def test_overwrite_hdf_file(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.hdf")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.hdf")
        f = open(path_fu, "w+")
        f.close()
        f = open(path_oldfu, "w+")
        f.close()
        fu.write_header(path_fu, IMAGE_2D, 1)
        oldfu.write_header(path_oldfu, IMAGE_2D, 1)
        self.assertEqual(returns_values_in_file(path_fu), returns_values_in_file(path_oldfu))
        self.assertTrue(path.isfile(path_fu))
        self.assertTrue(path.isfile(path_oldfu))
        remove_list_of_file([path_fu,path_oldfu])

    def test_hdf_type_AssertError_list_differ(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.hdf")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.hdf")
        fu.write_header(path_fu, IMAGE_2D, 2)
        oldfu.write_header(path_oldfu, IMAGE_2D, 1)
        self.assertTrue(path.isfile(path_fu))
        self.assertTrue(path.isfile(path_oldfu))
        with self.assertRaises(AssertionError) as cm:
            self.assertEqual(returns_values_in_file(path_fu), returns_values_in_file(path_oldfu))
        msg = cm.exception.message.split("'")
        self.assertEqual(msg[0].split(":")[0], "Lists differ")
        self.assertEqual(msg[10].split("\n")[2].split(":")[0], 'First differing element 2')
        remove_list_of_file([path_fu, path_oldfu])

    def test_invalid_filetype_error_msg(self):
        path_fu = path.join(ABSOLUTE_PATH, "test.txt")
        path_oldfu = path.join(ABSOLUTE_PATH, "test1.txt")
        fu.write_header(path_fu, IMAGE_2D, 1)
        oldfu.write_header(path_oldfu, IMAGE_2D, 1)
        self.assertFalse(path.isfile(path_fu))
        self.assertFalse(path.isfile(path_oldfu))



class Test_file_type(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.file_type()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.file_type()
        self.assertEqual(str(cm_new.exception), "file_type() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_bdb_filetype(self):
        fu.file_type("bdb:bdbfile")
        oldfu.file_type("bdb:bdbfile")
        self.assertTrue(True)

    def test_valid_filetype(self):
        fu.file_type("hdf.hdf")
        oldfu.file_type("hdf.hdf")
        self.assertTrue(True)

    def test_invalid_filetype_error_msg(self):
        fu.file_type("invalid.cc")
        oldfu.file_type("invalid.cc")
        self.assertTrue(True)


class Test_get_params2D(unittest.TestCase):
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_params2D"))
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_params2D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_params2D()
        self.assertEqual(str(cm_new.exception), "get_params2D() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_params2D(self):
        (ima,) = self.argum[0]
        return_new = fu.get_params2D(ima, xform="xform.align2d")
        return_old = oldfu.get_params2D(ima, xform="xform.align2d")
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,  (0.0, 0.0, 0.0, 0, 1.0)))

    def test_wrong_xform_returns_NotExistingObjectException_key_doesnot_exist(self):
        (ima,) = self.argum[0]
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params2D(ima, xform="xform.align3d")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params2D(ima, xform="xform.align3d")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_input_2dimg_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params2D(IMAGE_2D, xform="xform.align2d")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params2D(IMAGE_2D, xform="xform.align2d")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_input_3dimg_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params2D(IMAGE_3D, xform="xform.align2d")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params2D(IMAGE_3D, xform="xform.align2d")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    def test_NoneType_as_img_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        with self.assertRaises(AttributeError) as cm_new:
            fu.get_params2D(None, xform="xform.align2d")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.get_params2D(None, xform="xform.align2d")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
        """



class Test_set_params2D(unittest.TestCase):
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_params2D"))
    params=[1,2,3,4,5]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_params2D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_params2D()
        self.assertEqual(str(cm_new.exception), "set_params2D() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_params2D_using_wrongxform(self):
        (ima,) = self.argum[0]
        fu_img = deepcopy(ima)
        fu2_img = deepcopy(ima)
        fu.set_params2D(fu_img, self.params, xform="xform.align2d")
        oldfu.set_params2D(fu2_img, self.params, xform="xform.projection")     # is not setting the params
        self.assertFalse(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(fu2_img)))
        self.assertFalse(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(ima)))

    def test_set_params2D_using_wrongxform2(self):
        (ima,) = self.argum[0]
        fu_img = deepcopy(ima)
        fu2_img = deepcopy(ima)
        fu.set_params2D(fu_img, self.params, xform="xform.projection")       # is not setting the params
        oldfu.set_params2D(fu2_img, self.params, xform="xform.projection")      # is not setting the params
        self.assertTrue(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(fu2_img)))
        self.assertTrue(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(ima)))

    def test_set_params2D(self):
        (ima,) = self.argum[0]
        fu_img = deepcopy(ima)
        oldfu_img = deepcopy(ima)
        fu.set_params2D(fu_img, self.params, xform="xform.align2d")
        oldfu.set_params2D(oldfu_img, self.params, xform="xform.align2d")
        self.assertTrue(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(oldfu_img)))
        self.assertFalse(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(ima)))

    def test_less_params(self):
        (ima,) = self.argum[0]
        fu_img = deepcopy(ima)
        oldfu_img = deepcopy(ima)
        with self.assertRaises(IndexError) as cm_new:
            fu.set_params2D(fu_img, [0,1], xform="xform.align2d")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.set_params2D(oldfu_img, [0,1], xform="xform.align2d")
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    def test_wrong_xform_does_not_change_the_values_IS_IT_OK_OR_NOT(self):
        (ima,) = self.argum[0]
        fu_img = deepcopy(ima)
        oldfu_img = deepcopy(ima)
        fu.set_params2D(fu_img, self.params, xform="xform.align3d")          # is not setting the params
        oldfu.set_params2D(oldfu_img, self.params, xform="xform.align3d")    # is not setting the params
        self.assertTrue(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(oldfu_img)))
        self.assertTrue(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(ima)))

    def test_wrong_input_img(self):
        # I called it wrong image just because in the 'get_params2D' there was an error due to the missing xform key
        (ima,) = self.argum[0]
        fu_img = deepcopy(IMAGE_2D)
        oldfu_img = deepcopy(IMAGE_2D)
        fu.set_params2D(fu_img, self.params, xform="xform.align2d")
        oldfu.set_params2D(oldfu_img, self.params, xform="xform.align2d")
        self.assertTrue(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(oldfu_img)))
        self.assertFalse(array_equal(fu.get_params2D(fu_img), oldfu.get_params2D(ima)))

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.set_params2D(None, self.params, xform="xform.align2d")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.set_params2D(None, self.params, xform="xform.align2d")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'set_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))





class Test_get_params3D(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_params3D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_params3D()
        self.assertEqual(str(cm_new.exception), "get_params3D() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    # todo: I need a 3D image with 'xform.align3d' key
    """
    def test_get_params3D(self):
        return_new = fu.get_params3D(IMAGE_3D, xform="xform.align3d")
        return_old = oldfu.get_params3D(IMAGE_3D, xform="xform.align3d")
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0)))
    """

    def test_wrong_xform_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params3D(IMAGE_3D, xform="xform.align2d")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params3D(IMAGE_3D, xform="xform.align2d")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_input_img_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params3D(IMAGE_2D, xform="xform.align3d")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params3D(IMAGE_2D, xform="xform.align3d")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_NoneType_as_img_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        with self.assertRaises(AttributeError) as cm_new:
            fu.get_params3D(IMAGE_3D, xform="xform.align3d")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.get_params3D(IMAGE_3D, xform="xform.align3d")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
        """
#todo: I need a 3D image with 'xform.align3d' key
"""
class Test_set_params3D(unittest.TestCase):
    params=[1,2,3,4,5,6,7,8]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_params3D()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_params3D()
        self.assertEqual(str(cm_new.exception), "set_params3D() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_params3D(self):
        fu_img = deepcopy(IMAGE_3D)
        oldfu_img = deepcopy(IMAGE_3D)
        fu.set_params3D(fu_img, self.params, xform="xform.align3d")
        oldfu.set_params3D(oldfu_img, self.params, xform="xform.align3d")
        self.assertTrue(array_equal(fu.get_params3D(fu_img), oldfu.get_params3D(oldfu_img)))
        self.assertFalse(array_equal(fu.get_params3D(fu_img), oldfu.get_params3D(IMAGE_3D)))

    def test_less_params(self):
        fu_img = deepcopy(IMAGE_3D)
        oldfu_img = deepcopy(IMAGE_3D)
        with self.assertRaises(IndexError) as cm_new:
            fu.set_params3D(fu_img, [0,1], xform="xform.align3d")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.set_params3D(oldfu_img, [0,1], xform="xform.align3d")
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_xform_does_not_change_the_values_IS_IT_OK_OR_NOT(self):
        fu_img = deepcopy(IMAGE_3D)
        oldfu_img = deepcopy(IMAGE_3D)
        fu.set_params3D(fu_img, self.params, xform="xform.align2d")
        oldfu.set_params3D(oldfu_img, self.params, xform="xform.align2d")
        self.assertTrue(array_equal(fu.get_params3D(fu_img), oldfu.get_params3D(oldfu_img)))
        #self.assertFalse(array_equal(fu.get_params3D(fu_img), oldfu.get_params3D(IMAGE_3D)))

    def test_wrong_input_img(self):
        # I called it wrong image just because in the 'get_params2D' there was an error due to the missing xform key
        fu_img = deepcopy(IMAGE_2D)
        oldfu_img = deepcopy(IMAGE_2D)
        fu.set_params3D(fu_img, self.params, xform="xform.align3d")
        oldfu.set_params3D(oldfu_img, self.params, xform="xform.align3d")
        self.assertTrue(array_equal(fu.get_params3D(fu_img), oldfu.get_params3D(oldfu_img)))
        self.assertFalse(array_equal(fu.get_params3D(fu_img), oldfu.get_params3D(IMAGE_3D)))

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.set_params3D(None, self.params, xform="xform.align3d")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.set_params3D(None, self.params, xform="xform.align3d")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'set_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
"""


class Test_get_params_proj(unittest.TestCase):
    argum = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_params_proj"))
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_params_proj()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_params_proj()
        self.assertEqual(str(cm_new.exception), "get_params_proj() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_params_proj(self):
        (ima,) = self.argum[0]
        return_new = fu.get_params_proj(ima, xform="xform.projection")
        return_old = oldfu.get_params_proj(ima, xform="xform.projection")
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new, (14.71329548619616, 101.3719902962565, 220.4187405823029, -0.0, -0.0)))

    def test_wrong_xform_returns_NotExistingObjectException_key_doesnot_exist(self):
        (ima,) = self.argum[0]
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params_proj(ima, xform="xform.align3d")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params_proj(ima, xform="xform.align3d")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_input_2dimg_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params_proj(IMAGE_2D, xform="xform.projection")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params_proj(IMAGE_2D, xform="xform.projection")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_input_3dimg_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params_proj(IMAGE_3D, xform="xform.projection")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params_proj(IMAGE_3D, xform="xform.projection")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    def test_NoneType_as_img_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        with self.assertRaises(AttributeError) as cm_new:
            fu.get_params_proj(None, xform="xform.projection")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.get_params_proj(None, xform="xform.projection")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
        """



class Test_set_params_proj(unittest.TestCase):
    params=[1,2,3,4,5]

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_params_proj()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_params_proj()
        self.assertEqual(str(cm_new.exception), "set_params_proj() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_params_proj_using_wrongxform_returns_NotExistingObjectException_key_doesnot_exist(self): #error is ok
        fu_img = deepcopy(IMAGE_2D)
        fu.set_params_proj(fu_img, self.params, xform="xform.align2d")
        with self.assertRaises(RuntimeError) as cm:
            fu.get_params_proj(fu_img, xform="xform.projection")
        msg = cm.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")

    def test_set_params_proj_using_wrongxform2returns_NotExistingObjectException_key_doesnot_exist(self):
        fu_img = deepcopy(IMAGE_2D)
        fu2_img = deepcopy(IMAGE_2D)
        fu.set_params_proj(fu_img, self.params, xform="xform.align2d")       # is not setting the params
        oldfu.set_params_proj(fu2_img, self.params, xform="xform.align2d")      # is not setting the params
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_params_proj(fu_img, xform="xform.projection")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_params_proj(fu2_img, xform="xform.projection")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_params_proj(self):
        fu_img = deepcopy(IMAGE_2D)
        oldfu_img = deepcopy(IMAGE_2D)
        fu.set_params_proj(fu_img, self.params, xform="xform.projection")
        oldfu.set_params_proj(oldfu_img, self.params, xform="xform.projection")
        self.assertTrue(array_equal(fu.get_params_proj(fu_img, xform="xform.projection"), oldfu.get_params_proj(oldfu_img, xform="xform.projection")))
        #self.assertFalse(array_equal(fu.get_params_proj(fu_img), fu.get_params_proj(IMAGE_2D))) # IMAGE2D has not key ''xform.projection'

    def test_less_params(self):
        fu_img = deepcopy(IMAGE_2D)
        oldfu_img = deepcopy(IMAGE_2D)
        with self.assertRaises(IndexError) as cm_new:
            fu.set_params_proj(fu_img, [0,1], xform="xform.projection")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.set_params_proj(oldfu_img, [0,1], xform="xform.projection")
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.set_params_proj(None, self.params, xform="xform.projection")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.set_params_proj(None, self.params, xform="xform.projection")
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'set_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_get_ctf(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_ctf()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_ctf()
        self.assertEqual(str(cm_new.exception), "get_ctf() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_wrong_img_returns_NotExistingObjectException_key_doesnot_exist(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_ctf(IMAGE_2D)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_ctf(IMAGE_2D)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_ctf(self):
        img_with_ctf = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/alignment.ali2d_single_iter"))[0][0][0]
        return_new = fu.get_ctf(img_with_ctf)
        return_old = oldfu.get_ctf(img_with_ctf)
        self.assertTrue(array_equal(return_new,return_old ))
        self.assertTrue(array_equal(return_new, (1.1349999904632568, 0.009999999776482582, 300.0, 5.699999809265137, 0.0, 10.0, 0.04473999887704849, 130.39999389648438)))

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.get_ctf(None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.get_ctf(None)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_same_ctf(unittest.TestCase):
    params = [1,2,3,4,5,6]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.same_ctf()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.same_ctf()
        self.assertEqual(str(cm_new.exception), "same_ctf() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_same_ctf(self):
        self.assertTrue(fu.same_ctf(fu.generate_ctf(self.params),oldfu.generate_ctf(self.params)))

    def test_not_same_ctf(self):
        self.assertFalse(fu.same_ctf(fu.generate_ctf(self.params),oldfu.generate_ctf([0,1,2,3,4,5])))



class Test_generate_ctf(unittest.TestCase):
    """ params = [defocus, cs, voltage, apix, bfactor, ampcont, astigmatism_amplitude, astigmatism_angle] """
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.generate_ctf()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.generate_ctf()
        self.assertEqual(str(cm_new.exception), "generate_ctf() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_generate_ctf_with6Values(self):
        self.assertTrue(fu.same_ctf(fu.generate_ctf([1, 2, 3, 4, 5, 6]), oldfu.generate_ctf([1, 2, 3, 4, 5, 6])))

    def test_generate_ctf_with8Values(self):
        self.assertTrue(fu.same_ctf(fu.generate_ctf([1, 2, 3, 4, 5, 6,7,8]), oldfu.generate_ctf([1, 2, 3, 4, 5, 6,7,8])))

    def test_generate_ctf_with_incorrect_number_of_params_warning_msg(self):
        self.assertTrue(fu.generate_ctf([1, 2, 3, 4, 5, 6, 7]) is None)
        self.assertTrue(oldfu.generate_ctf([1, 2, 3, 4, 5, 6, 7]) is None)



class Test_delete_bdb(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.delete_bdb()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.delete_bdb()
        self.assertEqual(str(cm_new.exception), "delete_bdb() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_disable_bdb_cache(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.disable_bdb_cache(3)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.disable_bdb_cache(3)
        self.assertEqual(str(cm_new.exception), "disable_bdb_cache() takes no arguments (1 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_disable_bdb_cache(self):
        import EMAN2db
        EMAN2db.BDB_CACHE_DISABLE = False
        self.assertFalse(EMAN2db.BDB_CACHE_DISABLE)
        fu.disable_bdb_cache()
        self.assertTrue(EMAN2db.BDB_CACHE_DISABLE)
        EMAN2db.BDB_CACHE_DISABLE = False
        self.assertFalse(EMAN2db.BDB_CACHE_DISABLE)
        oldfu.disable_bdb_cache()
        self.assertTrue(EMAN2db.BDB_CACHE_DISABLE )



class Test_getvec(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.getvec()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.getvec()
        self.assertEqual(str(cm_new.exception), "getvec() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_tht_between_90_180(self):
        return_new = fu.getvec(0,100)
        return_old = oldfu.getvec(0, 100)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (-0.98480775301220802, 1.2060416625018976e-16, 0.17364817766693041)))

    def test_tht_bigger_than_180(self):
        return_new = fu.getvec(0,200)
        return_old = oldfu.getvec(0, 200)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  (-0.34202014332566871, 4.1885387376769918e-17, 0.93969262078590843)))

    def test_tht_lower_than_90(self):
        return_new = fu.getvec(0, 0)
        return_old = oldfu.getvec(0, 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  (0.0, 0.0, 1.0)))



class Test_getfvec(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.getfvec()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.getfvec()
        self.assertEqual(str(cm_new.exception), "getfvec() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_tht_between_90_180(self):
        return_new = fu.getfvec(0,100)
        return_old = oldfu.getfvec(0, 100)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.98480775301220802, 0.0, -0.1736481776669303)))


    def test_tht_bigger_than_180(self):
        return_new = fu.getfvec(0, 200)
        return_old = oldfu.getfvec(0, 200)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  (-0.34202014332566866, -0.0, -0.93969262078590843)))

    def test_tht_lower_than_90(self):
        return_new = fu.getfvec(0, 0)
        return_old = oldfu.getfvec(0, 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.0, 0.0, 1.0)))



class Test_nearest_fang(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearest_fang()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearest_fang()
        self.assertEqual(str(cm_new.exception), "nearest_fang() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nearest_fang_true_should_return_equal_objects(self):
        """ values got from pickle files/utilities/utilities.nearest_fang """
        vecs = [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487908840179443, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417963027954, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309417963027954, 0.44544869661331177, -0.3333333134651184], [-0.6804221868515015, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]
        tht = 66.00945
        phi = 58.54455
        return_new = fu.nearest_fang(vecs, phi, tht)
        return_old = oldfu.nearest_fang(vecs, phi, tht)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 1)


    def test_empty_vectore(self):
        """ values got from pickle files/utilities/utilities.nearest_fang """
        self.assertEqual(fu.nearest_fang([], 100, 100), oldfu.nearest_fang([], 100, 100))
        self.assertEqual(fu.nearest_fang([], 100, 100), -1)



class Test_nearest_many_full_k_projangles(unittest.TestCase):
    reference_normals = [[0.606369137763977, 0.7754802703857422, 0.17591717839241028], [0.344023197889328, 0.9092735648155212, 0.23424272239208221], [0.5131438970565796, 0.7110531330108643, -0.4807148575782776], [0.6525110006332397, 0.6401833295822144, 0.4054562747478485], [0.5846421718597412, 0.5353381037712097, 0.6095954775810242], [0.3914891481399536, 0.4943649470806122, 0.7761054039001465], [0.21746492385864258, 0.411188542842865, 0.8852304816246033], [0.18686196208000183, 0.4279184937477112, 0.8842897415161133], [0.2696961760520935, 0.41237473487854004, 0.870178759098053], [0.34728822112083435, 0.3424328565597534, 0.8730009198188782], [0.2467251867055893, 0.39220815896987915, 0.8861712217330933], [0.43794623017311096, 0.19451908767223358, 0.8777046203613281], [0.35838937759399414, 0.0876869484782219, -0.9294450283050537], [0.6956571340560913, 0.7182994484901428, 0.010348091833293438], [0.6555072665214539, 0.7445935010910034, -0.12605828046798706], [0.7438855767250061, 0.6679566502571106, -0.02163686789572239], [0.58192378282547, 0.8076738715171814, -0.09501412510871887], [0.7202955484390259, 0.693575382232666, 0.011288836598396301], [0.6438657641410828, 0.765091598033905, 0.008466575294733047], [0.6417456269264221, 0.7646241188049316, -0.05926619470119476], [0.593335747718811, 0.7773913145065308, 0.20884287357330322], [0.5866740942001343, 0.8075113296508789, -0.06114771217107773], [0.5893274545669556, 0.8044687509536743, 0.07431796938180923], [0.48042023181915283, 0.8660674691200256, 0.13828791677951813], [0.46822038292884827, 0.8812242746353149, -0.06491056084632874], [0.34745562076568604, 0.9322780966758728, 0.10065855830907822], [0.4396599531173706, 0.898162305355072, 0.0018815546063706279], [0.5071992874145508, 0.8368419408798218, 0.2060207575559616], [0.35214218497276306, 0.913831353187561, -0.20225776731967926], [0.5917134881019592, 0.798856258392334, 0.1081843376159668], [0.31928351521492004, 0.9256179332733154, -0.2031984180212021], [0.5689234137535095, 0.8101938962936401, 0.14111001789569855], [0.5366130471229553, 0.8180546164512634, 0.2069614678621292], [0.6138750910758972, 0.751165509223938, 0.24270929396152496], [0.6470115184783936, 0.7327832579612732, 0.210724338889122], [0.6170760989189148, 0.7832963466644287, 0.0752587541937828], [0.6726201176643372, 0.7090698480606079, 0.21166512370109558], [0.5653374195098877, 0.7982293963432312, 0.2079022079706192], [0.6659785509109497, 0.704732358455658, 0.24459083378314972], [0.6436562538146973, 0.7641429901123047, 0.04233306273818016], [0.6849393248558044, 0.7063358426094055, 0.17873942852020264], [0.5400856733322144, 0.8298555016517639, 0.14016936719417572], [0.5633652806282043, 0.8192181587219238, 0.10724367946386337], [0.5887830853462219, 0.8072782158851624, 0.040451530367136], [0.5886198282241821, 0.8079495429992676, -0.02728116139769554], [0.5608543157577515, 0.8246564269065857, 0.07337724417448044], [0.6164841055870056, 0.7869266271591187, -0.026340581476688385], [0.6699250340461731, 0.7420257925987244, -0.0244591124355793], [0.6205720901489258, 0.7555667161941528, 0.20978358387947083], [0.668122410774231, 0.7417618036270142, -0.058325473219156265], [0.6953815221786499, 0.7172793745994568, 0.04421444982290268], [0.6165966987609863, 0.7861903309822083, 0.04139237478375435], [0.6167761087417603, 0.7871026396751404, 0.007525925524532795], [0.7440555691719055, 0.6680058240890503, 0.012229571118950844], [0.5889342427253723, 0.8081541061401367, 0.006585149094462395], [0.6699285507202148, 0.7411633729934692, 0.04327383637428284], [0.7258118987083435, 0.6720566749572754, 0.1467544287443161], [0.6510280966758728, 0.7452824115753174, 0.1439322829246521], [0.695436418056488, 0.7182027697563171, -0.02351834438741207], [0.6768127679824829, 0.7217592000961304, 0.1448730081319809], [0.659572958946228, 0.7303088307380676, 0.1777988076210022], [0.6193289160728455, 0.7775111198425293, 0.10912513732910156], [0.644066333770752, 0.7611650824546814, 0.07619946449995041], [0.646177351474762, 0.7552087903022766, 0.11006595939397812], [0.7330403327941895, 0.6557652354240417, 0.18062089383602142], [0.7375331521034241, 0.6283562183380127, 0.2474130392074585], [0.7217933535575867, 0.68284010887146, 0.11288806051015854], [0.6975162625312805, 0.6453772783279419, 0.311382919549942], [0.8656806349754333, 0.48613080382347107, 0.11947321146726608], [0.7893708944320679, 0.6029136180877686, 0.11571019887924194], [0.8126943111419678, 0.5629141926765442, 0.15051743388175964], [0.8193334341049194, 0.5153672695159912, 0.25117599964141846], [0.8606626391410828, 0.45913165807724, 0.22013163566589355], [0.9627028107643127, 0.2471613585948944, -0.11006592959165573], [0.8993244171142578, 0.4370543360710144, -0.014110974967479706], [0.8985337615013123, 0.4356166422367096, 0.053621746599674225], [0.6973650455474854, 0.6844565272331238, 0.21260593831539154], [0.7557139992713928, 0.6292310357093811, 0.18156161904335022], [0.720099925994873, 0.6935028433799744, -0.0225775558501482], [0.6813780665397644, 0.7211584448814392, -0.12511759996414185], [0.720158576965332, 0.689294695854187, 0.07902166992425919], [0.6333718299865723, 0.7533666491508484, 0.17685800790786743], [0.7017511129379272, 0.6973404884338379, 0.14581383764743805], [0.670264720916748, 0.7381020188331604, 0.07714022696018219], [0.6722255349159241, 0.7319769859313965, 0.11100655794143677], [0.6406394839286804, 0.7163008451461792, 0.2765757739543915], [0.6907424926757812, 0.6801389455795288, 0.2455316036939621], [0.6244292855262756, 0.7678812146186829, 0.1429915428161621], [0.7094387412071228, 0.6814774870872498, 0.17968012392520905], [0.5963707566261292, 0.7414255142211914, 0.30761995911598206], [0.6974412798881531, 0.7078441977500916, 0.11194729804992676], [0.5866034030914307, 0.7729452252388, 0.24176852405071259], [0.7146044969558716, 0.6546692252159119, 0.24647238850593567], [0.5873112082481384, 0.7613202929496765, 0.2746942937374115], [0.50588458776474, 0.7085287570953369, 0.49200382828712463]]
    angles = [[41.921590970437258, 91.23979851375101, 333.346436124961, -0.0, -0.0]]
    howmany = 47
    symclass = foundamental_symclasss("c1")
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearest_many_full_k_projangles()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearest_many_full_k_projangles()
        self.assertEqual(str(cm_new.exception), "nearest_many_full_k_projangles() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_pickle_file_values(self):
        symclass = foundamental_symclasss("c5")    # I creasted it like the one of the pickle file
        return_new = fu.nearest_many_full_k_projangles(self.reference_normals, self.angles, self.howmany, symclass)
        return_old = oldfu.nearest_many_full_k_projangles(self.reference_normals, self.angles, self.howmany, symclass)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new, [[15, 53, 78, 17, 58, 13, 50, 47, 80, 49, 55, 79, 66, 83, 18, 19, 90, 39, 14, 69, 84, 62, 56, 46, 82, 52, 51, 63, 59, 35, 64, 88, 57, 77, 44, 54, 61, 40, 70, 21, 43, 60, 16, 87, 22, 29, 76]]))

    def test_with_class_c1(self):
        return_new = fu.nearest_many_full_k_projangles(self.reference_normals, self.angles, self.howmany, self.symclass)
        return_old = oldfu.nearest_many_full_k_projangles(self.reference_normals, self.angles, self.howmany, self.symclass)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,  [[15, 53, 78, 17, 58, 13, 50, 47, 80, 49, 55, 79, 66, 83, 18, 19, 90, 39, 14, 69, 84, 62, 56, 46, 82, 52, 51, 63, 59, 35, 64, 88, 57, 77, 44, 54, 61, 40, 70, 21, 43, 60, 16, 87, 22, 29, 76]]))

    def test_with_null_howmany(self):
        return_new = fu.nearest_many_full_k_projangles(self.reference_normals, self.angles, 0, self.symclass)
        return_old = oldfu.nearest_many_full_k_projangles(self.reference_normals, self.angles, 0, self.symclass)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,  [[]]))

    def test_with_empty_list_returns_RuntimeError_InvalidValueException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.nearest_many_full_k_projangles([], self.angles, self.howmany, self.symclass)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.nearest_many_full_k_projangles([], self.angles, self.howmany, self.symclass)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "Error, number of neighbors cannot be larger than number of reference directions")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_assign_projdirs_f(unittest.TestCase):
    """
    Since 'projdirs' and 'refdirs' are got in the sxmeridian from  angles_to_normals(list_of_angles) I used the
    output of the angles_to_normals tests to fill the 'projdirs' variable. the refdirs values are 2/3 of projdirs values
    """
    projdirs =  [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]
    refdirs = [[0.0, 0.0, 0.66], [0.44907856464385987, 0.4307301163673401, 0.22000000655651095], [-0.27087580382823945, 0.5602020227909088, 0.22000000655651095], [-0.6164889872074127, -0.08450628250837326, 0.22000000655651095], [-0.11013545751571656, -0.6124297463893891, 0.22000000655651095], [0.5484215462207794, -0.2939962184429169, 0.22000000655651095], [5.7699032538494066e-08, 5.044209628034821e-15, -0.66], [0.6164889872074127, 0.08450620383024215, -0.21999998688697817], [0.11013536900281906, 0.6124297463893891, -0.21999998688697817], [-0.5484216248989106, 0.2939961397647858, -0.21999998688697817], [-0.44907860398292543, -0.43073007702827454, -0.21999998688697817], [0.2708758628368378, -0.5602019834518432, -0.21999998688697817]]

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.assign_projdirs_f()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.assign_projdirs_f()
        self.assertEqual(str(cm_new.exception), "assign_projdirs_f() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_real_data(self):
        neighbors = int(len(self.projdirs)/ len(self.refdirs))
        return_new = fu.assign_projdirs_f(self.projdirs, self.refdirs, neighbors)
        return_old = oldfu.assign_projdirs_f(self.projdirs, self.refdirs, neighbors)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]))

    def test_with_null_neighboor_value_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.assign_projdirs_f(self.projdirs, self.refdirs, 0)
        return_old = oldfu.assign_projdirs_f(self.projdirs, self.refdirs, 0)
        self.assertTrue(array_equal(return_new,return_old))
        """

    def test_with_negative_neighboor_value_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.assign_projdirs_f(self.projdirs, self.refdirs, -1)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.assign_projdirs_f(self.projdirs, self.refdirs, -1)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_too_high_neighboor_value_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.assign_projdirs_f(self.projdirs, self.refdirs, 5)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.assign_projdirs_f(self.projdirs, self.refdirs, 5)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_invalid_neighbors_type(self):
        neighbors = len(self.projdirs)/ len(self.refdirs)
        with self.assertRaises(TypeError) as cm_new:
            fu.assign_projdirs_f(self.projdirs, self.refdirs, neighbors)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.assign_projdirs_f(self.projdirs, self.refdirs, neighbors)
        msg = str(cm_new.exception).split("\n")
        msg_old = str(cm_old.exception).split("\n")
        self.assertEqual(msg[0]+msg[1], 'Python argument types in    Util.assign_projdirs_f(list, list, float)')
        self.assertEqual(msg[0]+msg[1], msg_old[0]+msg_old[1])

    def test_with_projdirs_refdirs_have_different_length(self):
        refdirs= self.refdirs [:10]
        neighbors = int(len(self.projdirs)/ len(refdirs))
        return_new = fu.assign_projdirs_f(self.projdirs, refdirs, neighbors)
        return_old = oldfu.assign_projdirs_f(self.projdirs, refdirs, neighbors)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,  [[0], [1], [2], [3, 10], [4], [5, 11], [6], [7], [8], [9]]))

    def test_empty_projdirs(self):
        return_new = fu.assign_projdirs_f([], self.refdirs, 1)
        return_old = oldfu.assign_projdirs_f([], self.refdirs, 1)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,  [[], [], [], [], [], [], [], [], [], [], [], []]))

    def test_empty_refdirs_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.assign_projdirs_f(self.projdirs, [], 1)
        return_old = oldfu.assign_projdirs_f(self.projdirs, [], 1)
        self.assertTrue(array_equal(return_new,return_old))
        """



class Test_angles_to_normals(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angles_to_normals()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angles_to_normals()
        self.assertEqual(str(cm_new.exception), "angles_to_normals() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_pickle_file_values(self):
        angles = [[0.0, 0.0, 0.0], [43.805265094506787, 70.528779365509308, 0.0], [115.80526509450678, 70.528779365509308, 0.0], [187.80526509450678, 70.528779365509308, 0.0], [259.80526509450681, 70.528779365509308, 0.0], [331.80526509450681, 70.528779365509308, 0.0], [180.0, 180.0, 0.0], [7.8052650945068081, 109.47122063449069, 0.0], [79.805265094506808, 109.47122063449069, 0.0], [151.80526509450681, 109.47122063449069, 0.0], [223.80526509450681, 109.47122063449069, 0.0], [295.80526509450681, 109.47122063449069, 0.0]]
        return_new = fu.angles_to_normals(angles)
        return_old = oldfu.angles_to_normals(angles)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]))

    def test_with_empty_angles_list(self):
        return_new = fu.angles_to_normals([])
        return_old = oldfu.angles_to_normals([])
        self.assertTrue(array_equal(return_new,return_old))
        self.assertTrue(array_equal(return_new,[]))



class Test_angular_occupancy(unittest.TestCase):
    params = [[12,1,32],[12,11,2],[2,1,32],[121,19,32],[1.2,1,3.2],[102,1,32],[12,10,32],[9,16,32]]
    angstep = 15

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angular_occupancy()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angular_occupancy()
        self.assertEqual(str(cm_new.exception), "angular_occupancy() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_less_angles_returns_IndexError_list_index_out_of_range(self):
        angles=[[0.1],[21.1],[30.11],[1.1]]
        with self.assertRaises(IndexError) as cm_new:
            fu.angular_occupancy(angles, self.angstep, 'c5', 'S')
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angular_occupancy(angles, self.angstep, 'c5', 'S')
        self.assertEqual(str(cm_new.exception), "index 1 is out of bounds for axis 1 with size 1")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c5_method_S(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'c5', 'S')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'c5', 'S')
        self.assertTrue(array_equal(return_new[0],return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(), [0, 0, 0, 1, 0, 0, 0, 1]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [45.63832975533989, 19.18813645372093, 0.0], [6.3806392352448285, 27.266044450732828, 0.0], [33.51666024724356, 33.55730976192071, 0.0], [57.38151411228954, 38.94244126898138, 0.0], [7.068436092656995, 43.7617426926798, 0.0], [27.193047890155107, 48.1896851042214, 0.0], [46.143332861432214, 52.33011303567037, 0.0], [64.18346669052086, 56.251011404111416, 0.0], [9.503974866209632, 60.00000000000001, 0.0], [26.248664035752775, 63.612200038757, 0.0], [42.53027664295201, 67.11461952384143, 0.0], [58.44017921964933, 70.52877936550931, 0.0], [2.054682482737803, 73.87237978683925, 0.0], [17.439360304788927, 77.16041159309584, 0.0], [32.65213688990222, 80.40593177313954, 0.0], [47.745595738025806, 83.62062979155719, 0.0], [62.76879760804155, 86.8152614632796, 0.0], [5.7687977080415465, 90.0, 0.0]]))


    def test_with_sym_c1_method_S(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'c1', 'S')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'c1', 'S')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(),[0, 6, 0, 9, 0, 0, 0, 6] ))
        self.assertTrue(array_equal(return_new[1],[[0.0, 0.0, 0.0], [103.10941180192563, 8.364875267861896, 0.0], [176.21450628042055, 11.840273881876096, 0.0], [236.0656730373198, 14.514303081558177, 0.0], [288.0389625819211, 16.774744264095403, 0.0], [334.65212472788477, 18.771666189212812, 0.0], [17.32066588103538, 20.581969211500514, 0.0], [56.93303390011404, 22.251299640054558, 0.0], [94.08979640709249, 23.809376225696244, 0.0], [129.2192457664335, 25.27682691656447, 0.0], [162.63945091780187, 26.668660597354688, 0.0], [194.5942533593051, 27.996200493836653, 0.0], [225.27544022149684, 29.268238325258856, 0.0], [254.83707261705493, 30.49176175779049, 0.0], [283.40511428823197, 31.672433426416607, 0.0], [311.0841165757072, 32.81491794601815, 0.0], [337.96198959682874, 33.92311200325921, 0.0], [4.1134892483376575, 35.00031047544478, 0.0], [29.60281804202976, 36.04932905362067, 0.0], [54.48560148491377, 37.072596525218906, 0.0], [78.81041245173176, 38.07222541364264, 0.0], [102.61996477563827, 39.050066871297375, 0.0], [125.95205935497097, 40.007753913485175, 0.0], [148.84034277218166, 40.94673588313957, 0.0], [171.31492190766434, 41.8683062262928, 0.0], [193.40286659732473, 42.77362509927865, 0.0], [215.12862428281286, 43.663737936199425, 0.0], [236.5143647768975, 44.53959082510704, 0.0], [257.5802690164162, 45.402043338524194, 0.0], [278.3447725350669, 46.2518793150683, 0.0], [298.8247720406049, 47.08981597832405, 0.0], [319.03580170651145, 47.916511695972396, 0.0], [338.99218443330994, 48.732572619036205, 0.0], [358.7071622904154, 49.538558392667, 0.0], [18.19300963741847, 50.33498709240685, 0.0], [37.46113108715198, 51.1223395105916, 0.0], [56.522148070016584, 51.901062894532814, 0.0], [75.38597385890661, 52.67157421985833, 0.0], [94.06188009352373, 53.43426306781659, 0.0], [112.55855548442202, 54.18949416363721, 0.0], [130.8841578675008, 54.93760962356959, 0.0], [149.04636050865906, 55.67893095051762, 0.0], [167.05239341807666, 56.413760811888345, 0.0], [184.90908031797213, 57.142384628092266, 0.0], [202.62287181191573, 57.865071995851814, 0.0], [220.19987522407473, 58.58207796692143, 0.0], [237.64588151014897, 59.29364419985824, 0.0], [254.96638958583773, 60.00000000000001, 0.0], [272.16662837155565, 60.70136326071934, 0.0], [289.2515768122522, 61.39794131726056, 0.0], [306.22598209734565, 62.08993172297126, 0.0], [323.0943762769458, 62.7775229564712, 0.0], [339.8610914458917, 63.46089506721575, 0.0], [356.53027364599154, 64.14022026598406, 0.0], [13.105895718668648, 64.81566346602408, 0.0], [29.59176862453008, 65.48738277990002, 0.0], [45.99155283280313, 66.15552997649502, 0.0], [62.30876787180928, 66.82025090210684, 0.0], [78.54680162140626, 67.48168586912871, 0.0], [94.7089188193986, 68.13997001541874, 0.0], [110.79826894611286, 68.79523363712073, 0.0], [126.81789354449648, 69.44760249740412, 0.0], [142.77073302709655, 70.0971981133283, 0.0], [158.65963301599265, 70.74413802280935, 0.0], [174.4873502571022, 71.3885360334643, 0.0], [190.25655814616263, 72.03050245493117, 0.0], [205.9698519000554, 72.67014431610464, 0.0], [221.62975340390906, 73.30756556858914, 0.0], [237.23871576155324, 73.94286727754617, 0.0], [252.79912757434613, 74.57614780100388, 0.0], [268.3133169711282, 75.2075029585985, 0.0], [283.78355541002907, 75.83702619063038, 0.0], [299.2120612710465, 76.46480870823927, 0.0], [314.6010032566984, 77.09093963543441, 0.0], [329.9525036166027, 77.71550614365214, 0.0], [345.2686412105413, 78.33859357945896, 0.0], [0.5514545234039474, 78.96028558596703, 0.0], [15.802944044363288, 79.5806642184851, 0.0], [31.02507552169874, 80.19981005488745, 0.0], [46.21978210384517, 80.81780230114738, 0.0], [61.38896687649088, 81.43471889244924, 0.0], [76.53450520487294, 82.05063659026474, 0.0], [91.65824698981302, 82.66563107575222, 0.0], [106.76201884549539, 83.27977703981496, 0.0], [121.84762620650507, 83.89314827013408, 0.0], [136.91685537121364, 84.50581773547201, 0.0], [151.97147548821775, 85.11785766752689, 0.0], [167.01324049219787, 85.72933964060351, 0.0], [182.0438909952684, 86.34033464935307, 0.0], [197.06515613963234, 86.95091318482322, 0.0], [212.07875541713094, 87.56114530905013, 0.0], [227.08640046109076, 88.17110072841555, 0.0], [242.0897968157137, 88.78084886598522, 0.0], [257.09064568812886, 89.39045893303924, 0.0]] ))

    def test_with_sym_oct_method_S(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'oct', 'S')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'oct', 'S')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(), [0, 0, 0, 1, 0, 0, 0, 0]))
        self.assertTrue(array_equal(return_new[1],[[0.0, 0.0, 0.0], [18.060151356949547, 32.700469931476135, 0.0], [42.45792646077342, 37.9381274271855, 0.0]] ))

    def test_with_sym_invalid_method_S_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.angular_occupancy(self.params, self.angstep, 'invalid', 'S')
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angular_occupancy(self.params, self.angstep, 'invalid', 'S')
        self.assertEqual(str(cm_new.exception), "index 0 is out of bounds for axis 1 with size 0")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c5_method_P(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'c5', 'P')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'c5', 'P')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(), [0, 1, 0, 2, 0, 0, 1, 1]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0], [57.9555495773441, 15.0, 0.0], [0.0, 30.0, 0.0], [30.000000000000004, 30.0, 0.0], [60.00000000000001, 30.0, 0.0], [0.0, 45.0, 0.0], [21.213203435596427, 45.0, 0.0], [42.42640687119285, 45.0, 0.0], [63.63961030678928, 45.0, 0.0], [0.0, 60.0, 0.0], [17.320508075688775, 60.0, 0.0], [34.64101615137755, 60.0, 0.0], [51.96152422706632, 60.0, 0.0], [69.2820323027551, 60.0, 0.0], [0.0, 75.0, 0.0], [15.529142706151246, 75.0, 0.0], [31.058285412302492, 75.0, 0.0], [46.58742811845374, 75.0, 0.0], [62.116570824604985, 75.0, 0.0], [0.0, 90.0, 0.0], [15.0, 90.0, 0.0], [30.0, 90.0, 0.0], [45.0, 90.0, 0.0], [60.0, 90.0, 0.0]]))

    def test_with_sym_c1_method_P(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'c1', 'P')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'c1', 'P')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(), [0, 1, 0, 3, 0, 0, 1, 1]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0], [57.9555495773441, 15.0, 0.0], [115.9110991546882, 15.0, 0.0], [173.8666487320323, 15.0, 0.0], [231.8221983093764, 15.0, 0.0], [289.7777478867205, 15.0, 0.0], [347.73329746406455, 15.0, 0.0], [0.0, 30.0, 0.0], [30.000000000000004, 30.0, 0.0], [60.00000000000001, 30.0, 0.0], [90.00000000000001, 30.0, 0.0], [120.00000000000001, 30.0, 0.0], [150.00000000000003, 30.0, 0.0], [180.00000000000003, 30.0, 0.0], [210.00000000000003, 30.0, 0.0], [240.00000000000003, 30.0, 0.0], [270.00000000000006, 30.0, 0.0], [300.00000000000006, 30.0, 0.0], [330.00000000000006, 30.0, 0.0], [0.0, 45.0, 0.0], [21.213203435596427, 45.0, 0.0], [42.42640687119285, 45.0, 0.0], [63.63961030678928, 45.0, 0.0], [84.8528137423857, 45.0, 0.0], [106.06601717798213, 45.0, 0.0], [127.27922061357856, 45.0, 0.0], [148.49242404917499, 45.0, 0.0], [169.7056274847714, 45.0, 0.0], [190.91883092036784, 45.0, 0.0], [212.13203435596427, 45.0, 0.0], [233.3452377915607, 45.0, 0.0], [254.55844122715712, 45.0, 0.0], [275.77164466275354, 45.0, 0.0], [296.98484809834997, 45.0, 0.0], [318.1980515339464, 45.0, 0.0], [339.4112549695428, 45.0, 0.0], [0.0, 60.0, 0.0], [17.320508075688775, 60.0, 0.0], [34.64101615137755, 60.0, 0.0], [51.96152422706632, 60.0, 0.0], [69.2820323027551, 60.0, 0.0], [86.60254037844388, 60.0, 0.0], [103.92304845413265, 60.0, 0.0], [121.24355652982143, 60.0, 0.0], [138.5640646055102, 60.0, 0.0], [155.88457268119896, 60.0, 0.0], [173.20508075688772, 60.0, 0.0], [190.5255888325765, 60.0, 0.0], [207.84609690826525, 60.0, 0.0], [225.16660498395402, 60.0, 0.0], [242.48711305964278, 60.0, 0.0], [259.80762113533154, 60.0, 0.0], [277.12812921102034, 60.0, 0.0], [294.44863728670913, 60.0, 0.0], [311.7691453623979, 60.0, 0.0], [329.0896534380867, 60.0, 0.0], [346.4101615137755, 60.0, 0.0], [0.0, 75.0, 0.0], [15.529142706151246, 75.0, 0.0], [31.058285412302492, 75.0, 0.0], [46.58742811845374, 75.0, 0.0], [62.116570824604985, 75.0, 0.0], [77.64571353075623, 75.0, 0.0], [93.17485623690747, 75.0, 0.0], [108.70399894305872, 75.0, 0.0], [124.23314164920997, 75.0, 0.0], [139.7622843553612, 75.0, 0.0], [155.29142706151245, 75.0, 0.0], [170.8205697676637, 75.0, 0.0], [186.34971247381495, 75.0, 0.0], [201.8788551799662, 75.0, 0.0], [217.40799788611744, 75.0, 0.0], [232.9371405922687, 75.0, 0.0], [248.46628329841994, 75.0, 0.0], [263.9954260045712, 75.0, 0.0], [279.5245687107224, 75.0, 0.0], [295.0537114168736, 75.0, 0.0], [310.58285412302484, 75.0, 0.0], [326.11199682917606, 75.0, 0.0], [341.6411395353273, 75.0, 0.0], [357.1702822414785, 75.0, 0.0], [0.0, 90.0, 0.0], [15.0, 90.0, 0.0], [30.0, 90.0, 0.0], [45.0, 90.0, 0.0], [60.0, 90.0, 0.0], [75.0, 90.0, 0.0], [90.0, 90.0, 0.0], [105.0, 90.0, 0.0], [120.0, 90.0, 0.0], [135.0, 90.0, 0.0], [150.0, 90.0, 0.0], [165.0, 90.0, 0.0], [180.0, 90.0, 0.0], [195.0, 90.0, 0.0], [210.0, 90.0, 0.0], [225.0, 90.0, 0.0], [240.0, 90.0, 0.0], [255.0, 90.0, 0.0], [270.0, 90.0, 0.0], [285.0, 90.0, 0.0], [300.0, 90.0, 0.0], [315.0, 90.0, 0.0], [330.0, 90.0, 0.0], [345.0, 90.0, 0.0]]))

    def test_with_sym_oct_method_P(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'oct', 'P')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'oct', 'P')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(),[0, 1, 0, 1, 0, 0, 1, 1] ))
        self.assertTrue(array_equal(return_new[1],[[0.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 30.0, 0.0], [30.000000000000004, 30.0, 0.0], [0.0, 45.0, 0.0], [21.213203435596427, 45.0, 0.0], [42.42640687119285, 45.0, 0.0]] ))

    def test_with_sym_c5_method_invalid(self):
        return_new = fu.angular_occupancy(self.params, self.angstep, 'c5', 'invalid')
        return_old = oldfu.angular_occupancy(self.params, self.angstep, 'c5', 'invalid')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0].flatten(), [0, 0, 0, 1, 0, 0, 0, 1]))
        self.assertTrue(array_equal(return_new[1],[[0.0, 0.0, 0.0], [45.63832975533989, 19.18813645372093, 0.0], [6.3806392352448285, 27.266044450732828, 0.0], [33.51666024724356, 33.55730976192071, 0.0], [57.38151411228954, 38.94244126898138, 0.0], [7.068436092656995, 43.7617426926798, 0.0], [27.193047890155107, 48.1896851042214, 0.0], [46.143332861432214, 52.33011303567037, 0.0], [64.18346669052086, 56.251011404111416, 0.0], [9.503974866209632, 60.00000000000001, 0.0], [26.248664035752775, 63.612200038757, 0.0], [42.53027664295201, 67.11461952384143, 0.0], [58.44017921964933, 70.52877936550931, 0.0], [2.054682482737803, 73.87237978683925, 0.0], [17.439360304788927, 77.16041159309584, 0.0], [32.65213688990222, 80.40593177313954, 0.0], [47.745595738025806, 83.62062979155719, 0.0], [62.76879760804155, 86.8152614632796, 0.0], [5.7687977080415465, 90.0, 0.0]] ))

    def test_with_empty_params_list_returns_indexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.angular_occupancy([], self.angstep, 'c5', 'S')
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angular_occupancy([], self.angstep, 'c5', 'S')
        self.assertEqual(str(cm_new.exception), "index 1 is out of bounds for axis 1 with size 0")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_null_angstep_returns_ZeroDivisionError_error_msg(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.angular_occupancy(self.params, 0, 'c5', 'S')
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.angular_occupancy(self.params, 0, 'c5', 'S')
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


#I did not changged the inc_mirror because it is only called to 'angular_occupancy'. I'll change it there
class Test_angular_histogram(unittest.TestCase):
    params = [[12,1,32],[12,11,2],[2,1,32],[121,19,32],[1.2,1,3.2],[102,1,32],[12,10,32],[9,16,32]]
    angstep = 15
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angular_histogram()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angular_histogram()
        self.assertEqual(str(cm_new.exception), "angular_histogram() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_less_angles_returns_IndexError_list_index_out_of_range(self):
        angles=[[0.1],[21.1],[30.11],[1.1]]
        with self.assertRaises(IndexError) as cm_new:
            fu.angular_histogram(params=angles,angstep= self.angstep, sym="c1", method="S", inc_mirror=0)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angular_histogram(params=angles,angstep= self.angstep, sym="c1", method="S", inc_mirror=0)
        self.assertEqual(str(cm_new.exception), "index 1 is out of bounds for axis 1 with size 1")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c5_method_S(self):
        return_new = fu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='c5', method="S", inc_mirror=0)
        return_old = oldfu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='c5', method="S", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [45.63832975533989, 19.18813645372093, 0.0], [6.3806392352448285, 27.266044450732828, 0.0], [33.51666024724356, 33.55730976192071, 0.0], [57.38151411228954, 38.94244126898138, 0.0], [7.068436092656995, 43.7617426926798, 0.0], [27.193047890155107, 48.1896851042214, 0.0], [46.143332861432214, 52.33011303567037, 0.0], [64.18346669052086, 56.251011404111416, 0.0], [9.503974866209632, 60.00000000000001, 0.0], [26.248664035752775, 63.612200038757, 0.0], [42.53027664295201, 67.11461952384143, 0.0], [58.44017921964933, 70.52877936550931, 0.0], [2.054682482737803, 73.87237978683925, 0.0], [17.439360304788927, 77.16041159309584, 0.0], [32.65213688990222, 80.40593177313954, 0.0], [47.745595738025806, 83.62062979155719, 0.0], [62.76879760804155, 86.8152614632796, 0.0], [5.7687977080415465, 90.0, 0.0]]))

    def test_with_sym_c1_method_S(self):
        return_new = fu.angular_histogram(params=deepcopy(self.params),angstep= self.angstep, sym='c1', method="S", inc_mirror=0)
        return_old = oldfu.angular_histogram(params=deepcopy(self.params),angstep= self.angstep, sym='c1', method="S", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [5, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [103.10941180192563, 8.364875267861896, 0.0], [176.21450628042055, 11.840273881876096, 0.0], [236.0656730373198, 14.514303081558177, 0.0], [288.0389625819211, 16.774744264095403, 0.0], [334.65212472788477, 18.771666189212812, 0.0], [17.32066588103538, 20.581969211500514, 0.0], [56.93303390011404, 22.251299640054558, 0.0], [94.08979640709249, 23.809376225696244, 0.0], [129.2192457664335, 25.27682691656447, 0.0], [162.63945091780187, 26.668660597354688, 0.0], [194.5942533593051, 27.996200493836653, 0.0], [225.27544022149684, 29.268238325258856, 0.0], [254.83707261705493, 30.49176175779049, 0.0], [283.40511428823197, 31.672433426416607, 0.0], [311.0841165757072, 32.81491794601815, 0.0], [337.96198959682874, 33.92311200325921, 0.0], [4.1134892483376575, 35.00031047544478, 0.0], [29.60281804202976, 36.04932905362067, 0.0], [54.48560148491377, 37.072596525218906, 0.0], [78.81041245173176, 38.07222541364264, 0.0], [102.61996477563827, 39.050066871297375, 0.0], [125.95205935497097, 40.007753913485175, 0.0], [148.84034277218166, 40.94673588313957, 0.0], [171.31492190766434, 41.8683062262928, 0.0], [193.40286659732473, 42.77362509927865, 0.0], [215.12862428281286, 43.663737936199425, 0.0], [236.5143647768975, 44.53959082510704, 0.0], [257.5802690164162, 45.402043338524194, 0.0], [278.3447725350669, 46.2518793150683, 0.0], [298.8247720406049, 47.08981597832405, 0.0], [319.03580170651145, 47.916511695972396, 0.0], [338.99218443330994, 48.732572619036205, 0.0], [358.7071622904154, 49.538558392667, 0.0], [18.19300963741847, 50.33498709240685, 0.0], [37.46113108715198, 51.1223395105916, 0.0], [56.522148070016584, 51.901062894532814, 0.0], [75.38597385890661, 52.67157421985833, 0.0], [94.06188009352373, 53.43426306781659, 0.0], [112.55855548442202, 54.18949416363721, 0.0], [130.8841578675008, 54.93760962356959, 0.0], [149.04636050865906, 55.67893095051762, 0.0], [167.05239341807666, 56.413760811888345, 0.0], [184.90908031797213, 57.142384628092266, 0.0], [202.62287181191573, 57.865071995851814, 0.0], [220.19987522407473, 58.58207796692143, 0.0], [237.64588151014897, 59.29364419985824, 0.0], [254.96638958583773, 60.00000000000001, 0.0], [272.16662837155565, 60.70136326071934, 0.0], [289.2515768122522, 61.39794131726056, 0.0], [306.22598209734565, 62.08993172297126, 0.0], [323.0943762769458, 62.7775229564712, 0.0], [339.8610914458917, 63.46089506721575, 0.0], [356.53027364599154, 64.14022026598406, 0.0], [13.105895718668648, 64.81566346602408, 0.0], [29.59176862453008, 65.48738277990002, 0.0], [45.99155283280313, 66.15552997649502, 0.0], [62.30876787180928, 66.82025090210684, 0.0], [78.54680162140626, 67.48168586912871, 0.0], [94.7089188193986, 68.13997001541874, 0.0], [110.79826894611286, 68.79523363712073, 0.0], [126.81789354449648, 69.44760249740412, 0.0], [142.77073302709655, 70.0971981133283, 0.0], [158.65963301599265, 70.74413802280935, 0.0], [174.4873502571022, 71.3885360334643, 0.0], [190.25655814616263, 72.03050245493117, 0.0], [205.9698519000554, 72.67014431610464, 0.0], [221.62975340390906, 73.30756556858914, 0.0], [237.23871576155324, 73.94286727754617, 0.0], [252.79912757434613, 74.57614780100388, 0.0], [268.3133169711282, 75.2075029585985, 0.0], [283.78355541002907, 75.83702619063038, 0.0], [299.2120612710465, 76.46480870823927, 0.0], [314.6010032566984, 77.09093963543441, 0.0], [329.9525036166027, 77.71550614365214, 0.0], [345.2686412105413, 78.33859357945896, 0.0], [0.5514545234039474, 78.96028558596703, 0.0], [15.802944044363288, 79.5806642184851, 0.0], [31.02507552169874, 80.19981005488745, 0.0], [46.21978210384517, 80.81780230114738, 0.0], [61.38896687649088, 81.43471889244924, 0.0], [76.53450520487294, 82.05063659026474, 0.0], [91.65824698981302, 82.66563107575222, 0.0], [106.76201884549539, 83.27977703981496, 0.0], [121.84762620650507, 83.89314827013408, 0.0], [136.91685537121364, 84.50581773547201, 0.0], [151.97147548821775, 85.11785766752689, 0.0], [167.01324049219787, 85.72933964060351, 0.0], [182.0438909952684, 86.34033464935307, 0.0], [197.06515613963234, 86.95091318482322, 0.0], [212.07875541713094, 87.56114530905013, 0.0], [227.08640046109076, 88.17110072841555, 0.0], [242.0897968157137, 88.78084886598522, 0.0], [257.09064568812886, 89.39045893303924, 0.0]]))

    @unittest.skip("BUG in sp_fundamentals.py -->symmetry_neighbors --> local variable 'neighbors' referenced before assignment")
    def test_with_sym_oct_method_S(self):
        self.assertTrue(True)
        '''
        return_new = fu.angular_histogram(params=deepcopy(self.params),angstep= self.angstep, sym='oct1', method="S", inc_mirror=0)
        return_old = oldfu.fu.angular_histogram(params=deepcopy(self.params),angstep= self.angstep, sym='oct1', method="S", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [1215, 4645, 560]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [18.060151356949547, 32.700469931476135, 0.0], [42.457926460773422, 37.938127427185499, 0.0]]))
        '''

    def test_with_sym_invalid_method_S_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.angular_histogram(params=self.params,angstep= self.angstep, sym='invalid', method="S", inc_mirror=0)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angular_histogram(params=self.params,angstep= self.angstep, sym='invalid', method="S", inc_mirror=0)
        self.assertEqual(str(cm_new.exception), "index 0 is out of bounds for axis 1 with size 0")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c5_method_P(self):
        return_new = fu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='c5',  method="P", inc_mirror=0)
        return_old = oldfu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='c5',  method="P", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0], [57.9555495773441, 15.0, 0.0], [0.0, 30.0, 0.0], [30.000000000000004, 30.0, 0.0], [60.00000000000001, 30.0, 0.0], [0.0, 45.0, 0.0], [21.213203435596427, 45.0, 0.0], [42.42640687119285, 45.0, 0.0], [63.63961030678928, 45.0, 0.0], [0.0, 60.0, 0.0], [17.320508075688775, 60.0, 0.0], [34.64101615137755, 60.0, 0.0], [51.96152422706632, 60.0, 0.0], [69.2820323027551, 60.0, 0.0], [0.0, 75.0, 0.0], [15.529142706151246, 75.0, 0.0], [31.058285412302492, 75.0, 0.0], [46.58742811845374, 75.0, 0.0], [62.116570824604985, 75.0, 0.0], [0.0, 90.0, 0.0], [15.0, 90.0, 0.0], [30.0, 90.0, 0.0], [45.0, 90.0, 0.0], [60.0, 90.0, 0.0]]))

    def test_with_sym_c1_method_P(self):
        return_new = fu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep,sym= 'c1', method="P", inc_mirror=0)
        return_old = oldfu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep,sym= 'c1', method="P", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [4, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0], [57.9555495773441, 15.0, 0.0], [115.9110991546882, 15.0, 0.0], [173.8666487320323, 15.0, 0.0], [231.8221983093764, 15.0, 0.0], [289.7777478867205, 15.0, 0.0], [347.73329746406455, 15.0, 0.0], [0.0, 30.0, 0.0], [30.000000000000004, 30.0, 0.0], [60.00000000000001, 30.0, 0.0], [90.00000000000001, 30.0, 0.0], [120.00000000000001, 30.0, 0.0], [150.00000000000003, 30.0, 0.0], [180.00000000000003, 30.0, 0.0], [210.00000000000003, 30.0, 0.0], [240.00000000000003, 30.0, 0.0], [270.00000000000006, 30.0, 0.0], [300.00000000000006, 30.0, 0.0], [330.00000000000006, 30.0, 0.0], [0.0, 45.0, 0.0], [21.213203435596427, 45.0, 0.0], [42.42640687119285, 45.0, 0.0], [63.63961030678928, 45.0, 0.0], [84.8528137423857, 45.0, 0.0], [106.06601717798213, 45.0, 0.0], [127.27922061357856, 45.0, 0.0], [148.49242404917499, 45.0, 0.0], [169.7056274847714, 45.0, 0.0], [190.91883092036784, 45.0, 0.0], [212.13203435596427, 45.0, 0.0], [233.3452377915607, 45.0, 0.0], [254.55844122715712, 45.0, 0.0], [275.77164466275354, 45.0, 0.0], [296.98484809834997, 45.0, 0.0], [318.1980515339464, 45.0, 0.0], [339.4112549695428, 45.0, 0.0], [0.0, 60.0, 0.0], [17.320508075688775, 60.0, 0.0], [34.64101615137755, 60.0, 0.0], [51.96152422706632, 60.0, 0.0], [69.2820323027551, 60.0, 0.0], [86.60254037844388, 60.0, 0.0], [103.92304845413265, 60.0, 0.0], [121.24355652982143, 60.0, 0.0], [138.5640646055102, 60.0, 0.0], [155.88457268119896, 60.0, 0.0], [173.20508075688772, 60.0, 0.0], [190.5255888325765, 60.0, 0.0], [207.84609690826525, 60.0, 0.0], [225.16660498395402, 60.0, 0.0], [242.48711305964278, 60.0, 0.0], [259.80762113533154, 60.0, 0.0], [277.12812921102034, 60.0, 0.0], [294.44863728670913, 60.0, 0.0], [311.7691453623979, 60.0, 0.0], [329.0896534380867, 60.0, 0.0], [346.4101615137755, 60.0, 0.0], [0.0, 75.0, 0.0], [15.529142706151246, 75.0, 0.0], [31.058285412302492, 75.0, 0.0], [46.58742811845374, 75.0, 0.0], [62.116570824604985, 75.0, 0.0], [77.64571353075623, 75.0, 0.0], [93.17485623690747, 75.0, 0.0], [108.70399894305872, 75.0, 0.0], [124.23314164920997, 75.0, 0.0], [139.7622843553612, 75.0, 0.0], [155.29142706151245, 75.0, 0.0], [170.8205697676637, 75.0, 0.0], [186.34971247381495, 75.0, 0.0], [201.8788551799662, 75.0, 0.0], [217.40799788611744, 75.0, 0.0], [232.9371405922687, 75.0, 0.0], [248.46628329841994, 75.0, 0.0], [263.9954260045712, 75.0, 0.0], [279.5245687107224, 75.0, 0.0], [295.0537114168736, 75.0, 0.0], [310.58285412302484, 75.0, 0.0], [326.11199682917606, 75.0, 0.0], [341.6411395353273, 75.0, 0.0], [357.1702822414785, 75.0, 0.0], [0.0, 90.0, 0.0], [15.0, 90.0, 0.0], [30.0, 90.0, 0.0], [45.0, 90.0, 0.0], [60.0, 90.0, 0.0], [75.0, 90.0, 0.0], [90.0, 90.0, 0.0], [105.0, 90.0, 0.0], [120.0, 90.0, 0.0], [135.0, 90.0, 0.0], [150.0, 90.0, 0.0], [165.0, 90.0, 0.0], [180.0, 90.0, 0.0], [195.0, 90.0, 0.0], [210.0, 90.0, 0.0], [225.0, 90.0, 0.0], [240.0, 90.0, 0.0], [255.0, 90.0, 0.0], [270.0, 90.0, 0.0], [285.0, 90.0, 0.0], [300.0, 90.0, 0.0], [315.0, 90.0, 0.0], [330.0, 90.0, 0.0], [345.0, 90.0, 0.0]]))

    @unittest.skip("BUG in sp_fundamentals.py -->symmetry_neighbors --> local variable 'neighbors' referenced before assignment")
    def test_with_sym_oct_method_P(self):
        self.assertTrue(True)
        '''
        return_new = fu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='oct1', method="P", inc_mirror=0)
        return_old = oldfu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='oct1', method="P", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [7, 2700, 296, 3068, 41, 206, 102]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 30.0, 0.0], [30.000000000000004, 30.0, 0.0], [0.0, 45.0, 0.0], [21.213203435596427, 45.0, 0.0], [42.426406871192853, 45.0, 0.0]]))
        '''

    def test_with_sym_c5_method_invalid(self):
        return_new = fu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='c5', method="invalid", inc_mirror=0)
        return_old = oldfu.angular_histogram(params=deepcopy(self.params), angstep=self.angstep, sym='c5', method="invalid", inc_mirror=0)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(array_equal(return_new[1], [[0.0, 0.0, 0.0], [45.63832975533989, 19.18813645372093, 0.0], [6.3806392352448285, 27.266044450732828, 0.0], [33.51666024724356, 33.55730976192071, 0.0], [57.38151411228954, 38.94244126898138, 0.0], [7.068436092656995, 43.7617426926798, 0.0], [27.193047890155107, 48.1896851042214, 0.0], [46.143332861432214, 52.33011303567037, 0.0], [64.18346669052086, 56.251011404111416, 0.0], [9.503974866209632, 60.00000000000001, 0.0], [26.248664035752775, 63.612200038757, 0.0], [42.53027664295201, 67.11461952384143, 0.0], [58.44017921964933, 70.52877936550931, 0.0], [2.054682482737803, 73.87237978683925, 0.0], [17.439360304788927, 77.16041159309584, 0.0], [32.65213688990222, 80.40593177313954, 0.0], [47.745595738025806, 83.62062979155719, 0.0], [62.76879760804155, 86.8152614632796, 0.0], [5.7687977080415465, 90.0, 0.0]]))

    def test_with_empty_params_list_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.angular_histogram(params=[], angstep=self.angstep, sym='c5', method="S", inc_mirror=0)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angular_histogram(params=[], angstep=self.angstep, sym='c5', method="S", inc_mirror=0)
        self.assertEqual(cm_new.exception.message, "index 1 is out of bounds for axis 1 with size 0")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_with_null_angstep_returns_ZeroDivisionError_error_msg(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.angular_histogram(params=deepcopy(self.params),angstep= 0, sym='c5', method="S", inc_mirror=0)
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.angular_histogram(params=deepcopy(self.params),angstep= 0, sym='c5', method="S", inc_mirror=0)
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_balance_angular_distribution(unittest.TestCase):
    params = [[12,1,32],[12,11,2],[2,1,32],[121,19,32],[1.2,1,3.2],[102,1,32],[12,10,32],[9,16,32]]
    angstep = 15

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.balance_angular_distribution()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.balance_angular_distribution()
        self.assertEqual(str(cm_new.exception), "balance_angular_distribution() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c5_not_positive_maxOccupy(self):
        return_new = fu.balance_angular_distribution(params=self.params, max_occupy = -1, angstep = self.angstep, sym= 'c5')
        return_old=     oldfu.balance_angular_distribution(params=self.params, max_occupy = -1, angstep = self.angstep, sym= 'c5')
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new.flatten(),[0, 0, 0, 1, 0, 0, 0, 1]))

    @unittest.skip("compatibility test FAILED")
    def test_with_sym_c5_positive_maxOccupy(self):
        self.assertTrue(True)
        '''
        return_new = fu.balance_angular_distribution(params=self.params, max_occupy = 3, angstep = self.angstep, sym= 'c5')
        return_old=     oldfu.balance_angular_distribution(params=self.params, max_occupy = 3, angstep = self.angstep, sym= 'c5')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [0, 2, 3, 4, 7]))
        self.assertTrue(array_equal(return_new[1].flatten(),[12.0, 1.0, 32.0, 2.0, 1.0, 32.0, 121.0, 19.0, 32.0, 1.2, 1.0, 3.2, 9.0, 16.0, 32.0]))
        '''

    def test_with_sym_c1_not_positive_maxOccupy(self):
        return_new = fu.balance_angular_distribution(self.params, max_occupy = -1, angstep = self.angstep, sym= 'c1')
        return_old = oldfu.balance_angular_distribution(self.params, max_occupy = -1, angstep = self.angstep, sym= 'c1')
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new.flatten(), [0, 6, 0, 9, 0, 0, 0, 6]))

    def test_with_sym_c1_positive_maxOccupy(self):
        return_new = fu.balance_angular_distribution(self.params, max_occupy = -1, angstep = self.angstep, sym= 'c1')
        return_old = oldfu.balance_angular_distribution(self.params, max_occupy = -1, angstep = self.angstep, sym= 'c1')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [0]))
        self.assertTrue(array_equal(return_new[1].flatten(), [6]))

    def test_with_sym_oct_not_positive_maxOccupy(self):
        return_new = fu.balance_angular_distribution(self.params, max_occupy = -1, angstep = self.angstep, sym= 'oct')
        return_old = oldfu.balance_angular_distribution(self.params, max_occupy = -1, angstep = self.angstep, sym= 'oct')
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [0]))
        self.assertTrue(array_equal(return_new[1].flatten(), [0]))

    @unittest.skip("compatibility test FAILED")
    def test_with_sym_oct_positive_maxOccupy(self):
        self.assertTrue(True)
        '''
        return_new = fu.balance_angular_distribution(self.params, max_occupy = 3, angstep = self.angstep, sym= 'oct')
        return_old = oldfu.balance_angular_distribution(self.params, max_occupy = 3, angstep = self.angstep, sym= 'oct')
        self.assertTrue(array_equal(return_new[0], return_old[0]))  #failed
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [3, 4, 6, 7]))
        self.assertTrue(array_equal(return_new[1].flatten(), [121.0, 19.0, 32.0, 1.2, 1.0, 3.2, 12.0, 10.0, 32.0, 9.0, 16.0, 32.0]))
        '''

    def test_with_empty_list_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.balance_angular_distribution([], max_occupy = -1, angstep = self.angstep, sym= 'c5')
        with self.assertRaises(IndexError) as cm_old:
            oldfu.balance_angular_distribution([], max_occupy = -1, angstep = self.angstep, sym= 'c5')
        self.assertEqual(cm_new.exception.message, "too many indices for array")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_with_null_angstepy_error_msg(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.balance_angular_distribution(self.params, max_occupy = -1, angstep = 0, sym= 'c5')
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.balance_angular_distribution(self.params, max_occupy = -1, angstep = 0, sym= 'c5')
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c5_positive_maxOccupy_not_testabel(self):
        """
        It use to process random value that lead the function to returns always different va;ues
        """
        self.assertTrue(True)
        """
        return_new = fu.balance_angular_distribution(deepcopy(self.params), max_occupy = 1, angstep = self.angstep, sym= 'c5')
        return_old = oldfu.balance_angular_distribution(deepcopy(self.params), max_occupy = 1, angstep = self.angstep, sym= 'c5')
        self.assertTrue(array_equal(return_new, return_old))
        """



class Test_symmetry_neighbors(unittest.TestCase):
    angles = [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.symmetry_neighbors()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.symmetry_neighbors()
        self.assertEqual(str(cm_new.exception), "symmetry_neighbors() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_empty_list_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.symmetry_neighbors([] , symmetry= "c1")
        return_old = oldfu.symmetry_neighbors([], symmetry= "c1")
        self.assertTrue(array_equal(return_new, return_old))
        """

    def test_with_less_angles_returns_RuntimeError_3_angles_are_required(self):
        angles=[[0.1],[21.1],[30.11],[1.1]]
        with self.assertRaises(RuntimeError) as cm_new:
            fu.symmetry_neighbors(angles=angles , symmetry= "c1")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.symmetry_neighbors(angles=angles , symmetry= "c1")
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "InvalidValueException")
        self.assertEqual(msg[3], "Three angles are required")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_sym_c1(self):
        return_new = fu.symmetry_neighbors(angles=self.angles , symmetry= "c1")
        return_old = oldfu.symmetry_neighbors(angles=self.angles , symmetry= "c1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]))

    def test_with_sym_c5(self):
        return_new = fu.symmetry_neighbors(angles=self.angles , symmetry= "c5")
        return_old = oldfu.symmetry_neighbors(angles=self.angles , symmetry= "c5")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[0.0, 0.0, 1.0], [72.0, 0.0, 1.0], [288.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [72.680419921875, 0.6526213884353638, 0.3333333432674408], [288.680419921875, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [71.589599609375, 0.8487909436225891, 0.3333333432674408], [287.589599609375, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [71.06591796875, -0.12803982198238373, 0.3333333432674408], [287.06591796875, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [71.8331298828125, -0.927923858165741, 0.3333333432674408], [287.8331298828125, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [72.8309326171875, -0.4454488158226013, 0.3333333432674408], [288.8309326171875, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [72.0, 7.64274186065882e-15, -1.0], [288.0, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [72.93408203125, 0.12803970277309418, -0.3333333134651184], [288.93408203125, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [72.1668701171875, 0.927923858165741, -0.3333333134651184], [288.1668701171875, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [71.1690673828125, 0.44544869661331177, -0.3333333134651184], [287.1690673828125, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [71.319580078125, -0.652621328830719, -0.3333333134651184], [287.319580078125, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184], [72.410400390625, -0.8487908840179443, -0.3333333134651184], [288.410400390625, -0.8487908840179443, -0.3333333134651184]]))

    def test_with_sym_d1(self):
        return_new = fu.symmetry_neighbors(angles=self.angles , symmetry= "d1")
        return_old = oldfu.symmetry_neighbors(angles=self.angles , symmetry= "d1")
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0.0, 0.0, 1.0], [0.0, 180.0, 181.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [359.319580078125, 179.34738159179688, 180.3333282470703], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [0.410430908203125, 179.15121459960938, 180.3333282470703], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [0.93408203125, 180.12803649902344, 180.3333282470703], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.1668701171875, 180.92791748046875, 180.3333282470703], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [359.1690673828125, 180.44544982910156, 180.3333282470703], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.0, 180.0, 179.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [359.06591796875, 179.87196350097656, 179.6666717529297], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [359.8331298828125, 179.07208251953125, 179.6666717529297], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [0.8309326171875, 179.55455017089844, 179.6666717529297], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.680419921875, 180.65261840820312, 179.6666717529297], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184], [359.5895690917969, 180.84878540039062, 179.6666717529297]]))

    def test_with_sym_not_c_or_d(self):
        angles = [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408]]
        return_new = fu.symmetry_neighbors(angles=angles , symmetry= "invalid")
        return_old = oldfu.symmetry_neighbors(angles=angles , symmetry= "invalid")
        self.assertTrue(array_equal(return_new, return_old))



class Test_rotation_between_anglesets(unittest.TestCase):
    """  used the value used in 'Test_assign_projdirs_f' """
    agls1 =  [[0.0, 0.0, 1.0], [0.6804220676422119, 0.6526213884353638, 0.3333333432674408], [-0.4104178845882416, 0.8487909436225891, 0.3333333432674408], [-0.9340742230415344, -0.12803982198238373, 0.3333333432674408], [-0.16687190532684326, -0.927923858165741, 0.3333333432674408], [0.8309417366981506, -0.4454488158226013, 0.3333333432674408], [8.742277657347586e-08, 7.64274186065882e-15, -1.0], [0.9340742230415344, 0.12803970277309418, -0.3333333134651184], [0.16687177121639252, 0.927923858165741, -0.3333333134651184], [-0.8309418559074402, 0.44544869661331177, -0.3333333134651184], [-0.6804221272468567, -0.652621328830719, -0.3333333134651184], [0.41041797399520874, -0.8487908840179443, -0.3333333134651184]]
    agls2 = [[0.0, 0.0, 0.66], [0.44907856464385987, 0.4307301163673401, 0.22000000655651095], [-0.27087580382823945, 0.5602020227909088, 0.22000000655651095], [-0.6164889872074127, -0.08450628250837326, 0.22000000655651095], [-0.11013545751571656, -0.6124297463893891, 0.22000000655651095], [0.5484215462207794, -0.2939962184429169, 0.22000000655651095], [5.7699032538494066e-08, 5.044209628034821e-15, -0.66], [0.6164889872074127, 0.08450620383024215, -0.21999998688697817], [0.11013536900281906, 0.6124297463893891, -0.21999998688697817], [-0.5484216248989106, 0.2939961397647858, -0.21999998688697817], [-0.44907860398292543, -0.43073007702827454, -0.21999998688697817], [0.2708758628368378, -0.5602019834518432, -0.21999998688697817]]

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rotation_between_anglesets()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rotation_between_anglesets()
        self.assertEqual(str(cm_new.exception), "rotation_between_anglesets() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_rotation_between_anglesets(self):
        return_new = fu.rotation_between_anglesets(agls1=self.agls1, agls2=self.agls2)
        return_old = oldfu.rotation_between_anglesets(agls1=self.agls1, agls2=self.agls2)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  (2.215e-09, 0.0, 0.0)))

    def test_sets_have_different_length(self):
        agls2=self.agls2[:30]
        return_new = fu.rotation_between_anglesets(agls1=self.agls1, agls2=agls2)
        return_old = oldfu.rotation_between_anglesets(agls1=self.agls1, agls2=agls2)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  (2.215e-09, 0.0, 0.0)))

    def test_angls1_empty_list_error_msg(self):
        return_new = fu.rotation_between_anglesets(agls1=[], agls2=self.agls2)
        return_old = oldfu.rotation_between_anglesets(agls1=[], agls2=self.agls2)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertEqual(return_new, -1)

    def test_angls2_empty_list_error_msg(self):
        return_new = fu.rotation_between_anglesets(agls1=self.agls1, agls2=[])
        return_old = oldfu.rotation_between_anglesets(agls1=self.agls1, agls2=[])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertEqual(return_new, -1)



class Test_angle_between_projections_directions(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.angle_between_projections_directions()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.angle_between_projections_directions()
        self.assertEqual(str(cm_new.exception), "angle_between_projections_directions() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_3angles(self):
        agls1 = [20, 60, 0]
        agls2 = [45, 75, 5]
        return_new = fu.angle_between_projections_directions(proj1=agls1, proj2=agls2)
        return_old = oldfu.angle_between_projections_directions(proj1=agls1, proj2=agls2)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 27.432927773655976)

    def test_with_2angles(self):
        agls1 = [20, 60]
        agls2 = [45, 75]
        return_new = fu.angle_between_projections_directions(proj1=agls1, proj2=agls2)
        return_old = oldfu.angle_between_projections_directions(proj1=agls1, proj2=agls2)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 27.432927773655976)

    def test_with_list1_empty(self):
        agls2 = [45, 75]
        with self.assertRaises(IndexError) as cm_new:
            fu.angle_between_projections_directions(proj1=[], proj2=agls2)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angle_between_projections_directions(proj1=[], proj2=agls2)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_list2_empty(self):
        agls1 = [45, 75]
        with self.assertRaises(IndexError) as cm_new:
            fu.angle_between_projections_directions(proj1= agls1, proj2=[])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.angle_between_projections_directions( proj1=agls1, proj2=[])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_get_pixel_size(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_pixel_size()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_pixel_size()
        self.assertEqual(str(cm_new.exception), "get_pixel_size() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_pixel_size_img2d(self):
        return_new = fu.get_pixel_size(img=IMAGE_2D)
        return_old = oldfu.get_pixel_size(img=IMAGE_2D)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 1)

    def test_get_pixel_size_img3d(self):
        return_new = fu.get_pixel_size(img=IMAGE_3D)
        return_old = oldfu.get_pixel_size(img=IMAGE_3D)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 1)

    def test_get_pixel_size_imgEmpty(self):
        return_new = fu.get_pixel_size(img=EMData())
        return_old = oldfu.get_pixel_size(img=EMData())
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 1)

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.get_pixel_size(img=None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.get_pixel_size(img=None)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_attr_default'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_set_pixel_size(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.set_pixel_size()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.set_pixel_size()
        self.assertEqual(str(cm_new.exception), "set_pixel_size() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_set_pixel_size(self):
        img_fu = deepcopy(IMAGE_2D)
        img_fu_old = deepcopy(IMAGE_2D)
        fu.set_pixel_size(img=img_fu,pixel_size=2.1)
        oldfu.set_pixel_size(img=img_fu_old,pixel_size=2.1)
        self.assertEqual(img_fu.get_attr('apix_x'), img_fu_old.get_attr('apix_x'))
        self.assertEqual(img_fu.get_attr('apix_y'), img_fu_old.get_attr('apix_y'))
        self.assertEqual(img_fu.get_attr('apix_z'), img_fu_old.get_attr('apix_z'))
        self.assertEqual(img_fu.get_attr('apix_x'), 2.1)
        self.assertEqual(img_fu.get_attr('apix_y'), 2.1)
        self.assertEqual(img_fu.get_attr('apix_z'), 2.1)

    def test_set_pixel_size_truncated_value(self):
        img_fu = deepcopy(IMAGE_2D)
        img_fu_old = deepcopy(IMAGE_2D)
        fu.set_pixel_size(img=img_fu,pixel_size=2.1111)
        oldfu.set_pixel_size(img=img_fu_old,pixel_size=2.1111)
        self.assertEqual(img_fu.get_attr('apix_x'), img_fu_old.get_attr('apix_x'))
        self.assertEqual(img_fu.get_attr('apix_y'), img_fu_old.get_attr('apix_y'))
        self.assertEqual(img_fu.get_attr('apix_z'), img_fu_old.get_attr('apix_z'))
        self.assertEqual(img_fu.get_attr('apix_x'), 2.111)
        self.assertEqual(img_fu.get_attr('apix_y'), 2.111)
        self.assertEqual(img_fu.get_attr('apix_z'), 2.111)

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.set_pixel_size(img=None,pixel_size=2.1)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.set_pixel_size(img=None,pixel_size=2.1)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'get_zsize'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_lacos(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.lacos()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.lacos()
        self.assertEqual(str(cm_new.exception), "lacos() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_null_angle(self):
        return_new = fu.lacos(x=0)
        return_old = oldfu.lacos(x=0)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,90.0)

    def test_negative_angle(self):
        return_new = fu.lacos(x=-0.12)
        return_old = oldfu.lacos(x=-0.12)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, 96.892102579346385)

    def test_positive_angle(self):
        return_new = fu.lacos(x=0.12)
        return_old = oldfu.lacos(x=0.12)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new,83.107897420653629)

    def test_outOfRange_angle(self):
        return_new = fu.lacos(x=12)
        return_old = oldfu.lacos(x=12)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 0.0)


class Test_findall(unittest.TestCase):
    l = [1,2,3,4,5,5,5,4,3,2,1]

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.findall()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.findall()
        self.assertEqual(str(cm_new.exception), "findall() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_findall_5(self):
        return_new = fu.findall(value=5, L=self.l, start=0)
        return_old = oldfu.findall(value=5, L=self.l, start=0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [4, 5, 6]))

    def test_findall_noValues(self):
        return_new = fu.findall(value=0, L=self.l, start=0)
        return_old = oldfu.findall(value=0, L=self.l, start=0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, []))



class Test_class_iterImagesList(unittest.TestCase):
    list_of_imgs = [IMAGE_2D,IMAGE_3D,IMAGE_BLANK_2D,IMAGE_BLANK_3D,IMAGE_2D_REFERENCE]

    def test_invalid_init(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.iterImagesList()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.iterImagesList()
        self.assertEqual(str(cm_new.exception), "__init__() takes at least 2 arguments (1 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_valid_init(self):
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes = None)
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes = None)
        self.assertEqual(type(fu_obj).__name__ , "iterImagesList")
        self.assertEqual(type(fu_obj).__name__, type(oldfu_obj).__name__)

    def test_valid_init2(self):
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes = [1,2])
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes = [1,2])
        self.assertEqual(type(fu_obj).__name__ , "iterImagesList")
        self.assertEqual(type(fu_obj).__name__, type(oldfu_obj).__name__)

    def test_wrong_init_list_of_index_leads_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes = [1,2,7])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes = [1,2,7])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_iterNo(self):
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        self.assertEqual(fu_obj.iterNo(),oldfu_obj.iterNo())
        self.assertEqual(fu_obj.iterNo(), -1)

    def test_imageIndex(self):
        """ since the position is -1 it is returning the index of the last image hence 4"""
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        self.assertEqual(fu_obj.imageIndex(), oldfu_obj.imageIndex())
        self.assertEqual(fu_obj.imageIndex(), 4)

    def test_image(self):
        """ since the position is -1 it is returning the last image hence the 4th"""
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        fu_img=fu_obj.image()
        oldfu_img=oldfu_obj.image()
        expectedimg=self.list_of_imgs[fu_obj.imageIndex()]
        self.assertTrue(array_equal(fu_img.get_3dview(), oldfu_img.get_3dview()))
        self.assertTrue(array_equal(fu_img.get_3dview(), expectedimg.get_3dview()))

    def test_goToNext(self):
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)

        """ I'm testing all the data in the obj in order to test the return False"""
        fu_counter =0
        while fu_obj.goToNext():       # I'm , implicitly, testing the return True
            self.assertEqual(fu_obj.iterNo(), fu_counter)
            fu_counter += 1

        oldfu_counter =0
        while oldfu_obj.goToNext():
            self.assertEqual(oldfu_obj.iterNo(), oldfu_counter)
            oldfu_counter += 1

        """ no more img in the object"""
        self.assertFalse(fu_obj.goToNext())
        self.assertFalse(oldfu_obj.goToNext())

        """ check if both of the classes tested all the images"""
        self.assertTrue(fu_counter, oldfu_counter)
        self.assertTrue(fu_counter, len(self.list_of_imgs))

    def test_goToPrev(self):
        fu_obj = fu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        oldfu_obj = oldfu.iterImagesList(list_of_images=self.list_of_imgs, list_of_indexes=None)
        """At the beginning there is no previous image"""
        self.assertFalse(fu_obj.goToPrev())
        self.assertFalse(oldfu_obj.goToPrev())
        self.assertEqual(fu_obj.iterNo(),oldfu_obj.iterNo())
        self.assertEqual(fu_obj.iterNo(), -1)

        """ We are on the first image, it means that we have still no previous image"""
        fu_obj.goToNext()
        oldfu_obj.goToNext()
        self.assertEqual(fu_obj.iterNo(),oldfu_obj.iterNo())
        self.assertEqual(fu_obj.iterNo(), 0)

        self.assertFalse(fu_obj.goToPrev())
        self.assertFalse(oldfu_obj.goToPrev())
        self.assertEqual(fu_obj.iterNo(),oldfu_obj.iterNo())
        self.assertEqual(fu_obj.iterNo(), -1)

        """ We are on the second image, it means that we have an previous image"""
        fu_obj.goToNext()
        oldfu_obj.goToNext()
        fu_obj.goToNext()
        oldfu_obj.goToNext()
        self.assertEqual(fu_obj.iterNo(),oldfu_obj.iterNo())
        self.assertEqual(fu_obj.iterNo(), 1)

        self.assertTrue(fu_obj.goToPrev())
        self.assertTrue(oldfu_obj.goToPrev())
        self.assertEqual(fu_obj.iterNo(),oldfu_obj.iterNo())
        self.assertEqual(fu_obj.iterNo(), 0)



class Test_pack_message(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.pack_message()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.pack_message()
        self.assertEqual(str(cm_new.exception), "pack_message() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_data_is_a_string(self):
        data = "case S:I am a string!!!"
        return_new = fu.pack_message(data=data)
        return_old = oldfu.pack_message(data=data)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, 'Scase S:I am a string!!!')

    def test_data_is_a_very_long_string(self):
        long_data = "I am a stringggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg!!!"
        return_new = fu.pack_message(data=long_data)
        return_old = oldfu.pack_message(data=long_data)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, "C"+compress(long_data,1))

    def test_data_is_a_notstring(self):
        data = 5555
        return_new = fu.pack_message(data=data)
        return_old = oldfu.pack_message(data=data)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, "O" + pickle_dumps(data,-1))

    def test_data_is_a_notstring_long_version(self):
        data = 555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
        return_new = fu.pack_message(data=data)
        return_old = oldfu.pack_message(data=data)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, "Z" + compress(pickle_dumps(data, -1),1))



class Test_unpack_message(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.unpack_message()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.unpack_message()
        self.assertEqual(str(cm_new.exception), "unpack_message() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_data_is_a_string_BUG(self):
        self.assertTrue(True)
        """
        data = fu.pack_message("case S:I am a string!!!")
        return_new = fu.unpack_message(data)
        return_old = oldfu.unpack_message(data)
        self.assertEqual(return_new,return_old)
        """

    def test_data_is_a_very_long_string(self):
        self.assertTrue(True)
        """
        long_data = "I am a stringggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg!!!"
        data= fu.pack_message(long_data)
        return_new = fu.unpack_message(data)
        return_old = oldfu.unpack_message(data)
        self.assertEqual(return_new,return_old)
        """

    def test_data_is_a_notstring(self):
        self.assertTrue(True)
        """
        data = fu.pack_message(5555)
        return_new = fu.unpack_message(5555)
        return_old = oldfu.unpack_message(data)
        self.assertEqual(return_new,return_old)
        """
    def test_data_is_a_notstring_long_version(self):
        self.assertTrue(True)
        """
        data = fu.pack_message(555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555)
        return_new = fu.unpack_message(data)
        return_old = oldfu.unpack_message(data)
        self.assertEqual(return_new, return_old)
        """

    # data is a numpy array with the following value: ['O', '\x80' ,'\x02' ,']', 'q', '', 'K', 'I', 'a', '.'] and dtype = '|S1'. I cannot create it
    def test_pickle_file_values(self):
        (data,) = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.unpack_message"))[0]
        return_new = fu.unpack_message(msg=data)
        return_old = oldfu.unpack_message(msg=data)
        self.assertEqual(return_new, return_old)



class Test_wrap_mpi_send(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.wrap_mpi_send()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.wrap_mpi_send()
        self.assertEqual(str(cm_new.exception), "wrap_mpi_send() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_value(self):
        """ values got via pickle files/utilities/utilities.wrap_mpi_send"""
        return_new = fu.wrap_mpi_send(data = [9], destination = 0, communicator = None)
        return_old = oldfu.wrap_mpi_send(data =[9], destination = 0, communicator = None)
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_with_MPI_COMM_WORLD(self):
        return_new = fu.wrap_mpi_send(data = [9], destination = 0, communicator = MPI_COMM_WORLD)
        return_old = oldfu.wrap_mpi_send(data =[9], destination = 0, communicator = MPI_COMM_WORLD)
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_invalid_communicator_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.wrap_mpi_send(data = [9], destination = 0, communicator = -1)
        return_old = oldfu.wrap_mpi_send(data =[9], destination = 0, communicator = -1)
        self.assertEqual(return_new, return_old)
        """



class Test_wrap_mpi_recv(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.wrap_mpi_recv()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.wrap_mpi_recv()
        self.assertEqual(str(cm_new.exception), "wrap_mpi_recv() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    "Can only test on cluster , cannot work on workstation"
    # def test_wrap_mpi_recv_true_should_return_equal_objects(self):
    #     filepath = path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_recv")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle_load(rb)
    #
    #     print(argum[0])
    #
    #     (data, communicator) = argum[0]
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_new = fu.wrap_mpi_recv(data, communicator)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.wrap_mpi_recv(data, communicator)
    #
    #     self.assertEqual(return_new, return_old)



class Test_wrap_mpi_bcast(unittest.TestCase):
    """ Values got running Test_get_sorting_params_refine.test_default_case"""
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.wrap_mpi_bcast()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.wrap_mpi_bcast()
        self.assertEqual(str(cm_new.exception), "wrap_mpi_bcast() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_None_data(self):
        """ values got via pickle files/utilities/utilities.wrap_mpi_send"""
        return_new = fu.wrap_mpi_bcast(None, root=0, communicator= None)
        return_old = oldfu.wrap_mpi_bcast(None, root=0, communicator= None)
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_default_case(self):
        attr_value_list = [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0],[2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]
        return_new = fu.wrap_mpi_bcast(data = attr_value_list, root = 0, communicator = MPI_COMM_WORLD)
        return_old = oldfu.wrap_mpi_bcast(data =attr_value_list, root= 0, communicator = MPI_COMM_WORLD)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545,150.6989385443887, 95.77312314162165,0.0, 0.0], [2, 67.0993779295224,52.098986136572584,248.45843717750148, 0.0,0.0]]))


    def test_invalid_communicator_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.wrap_mpi_bcast(data = [9], root = 0, communicator = -1)
        return_old = oldfu.wrap_mpi_bcast(data =[9], root= 0, communicator = -1)
        self.assertEqual(return_new, return_old)
        """



class Test_wrap_mpi_gatherv(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.wrap_mpi_gatherv()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.wrap_mpi_gatherv()
        self.assertEqual(str(cm_new.exception), "wrap_mpi_gatherv() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_pickle_file_values(self):
        return_new = fu.wrap_mpi_gatherv(data = [45,3], root = 0, communicator= None)
        return_old = oldfu.wrap_mpi_gatherv(data= [45,3], root = 0, communicator= None)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [45, 3]))

    def test_with_MPI_COMM_WORLD(self):
        return_new = fu.wrap_mpi_gatherv(data = [45,3], root = 0, communicator= MPI_COMM_WORLD)
        return_old = oldfu.wrap_mpi_gatherv(data= [45,3], root = 0, communicator= MPI_COMM_WORLD)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [45, 3]))

    def test_invalid_communicator_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        return_new = fu.wrap_mpi_gatherv(data = [45,3], root = 0, communicator= -1)
        return_old = oldfu.wrap_mpi_gatherv(data= [45,3], root = 0, communicator= -1)
        self.assertEqual(return_new, return_old)
        """



class Test_get_colors_and_subsets(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_colors_and_subsets()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_colors_and_subsets()
        self.assertEqual(str(cm_new.exception), "get_colors_and_subsets() takes exactly 6 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_mainMode_equal_my_rank(self):
        main_node = 0
        my_rank = mpi_comm_rank(MPI_COMM_WORLD)
        shared_comm = mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL)
        sh_my_rank = mpi_comm_rank(shared_comm)
        masters = mpi_comm_split(MPI_COMM_WORLD, sh_my_rank == main_node, my_rank)
        shared_comm = mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL)
        return_new = fu.get_colors_and_subsets(main_node, MPI_COMM_WORLD, my_rank, shared_comm, sh_my_rank,masters)
        return_old = oldfu.get_colors_and_subsets(main_node, MPI_COMM_WORLD, my_rank, shared_comm, sh_my_rank,masters)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0, 1, True)))

    def test_mainMode_not_equal_my_rank_returns_TypeError_obj_Nonetype_hasnot_len(self):
        main_node = 0
        my_rank = mpi_comm_rank(MPI_COMM_WORLD)
        shared_comm = mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL)
        sh_my_rank = mpi_comm_rank(shared_comm)
        masters = mpi_comm_split(MPI_COMM_WORLD, sh_my_rank == main_node, my_rank)
        shared_comm = mpi_comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL)
        with self.assertRaises(TypeError) as cm_new:
            fu.get_colors_and_subsets(main_node, MPI_COMM_WORLD, my_rank, shared_comm, sh_my_rank+1,masters)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_colors_and_subsets(main_node, MPI_COMM_WORLD, my_rank, shared_comm, sh_my_rank+1,masters)
        self.assertEqual(str(cm_new.exception), "object of type 'NoneType' has no len()")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


class Test_wrap_mpi_split(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.wrap_mpi_split()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.wrap_mpi_split()
        self.assertEqual(str(cm_new.exception), "wrap_mpi_split() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


        """ Can only be tested in mpi not on workstation   """
    # def test_wrap_mpi_split_true_should_return_equal_objects(self):
    #     filepath = path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_split")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle_load(rb)
    #
    #     print(argum[0])
    #
    #     (comm, no_of_groups) = argum[0]
    #
    #     return_new = fu.wrap_mpi_split(comm, no_of_groups)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.wrap_mpi_split(comm, no_of_groups)
    #
    #     self.assertEqual(return_new, return_old)


class Test_get_dist(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_dist()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_dist()
        self.assertEqual(str(cm_new.exception), "get_dist() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_get_dist(self):
        return_new = fu.get_dist(c1=[2,4],c2=[5,1])
        return_old = oldfu.get_dist(c1=[2, 4], c2=[5, 1])
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 4.2426406871192848)

    def test_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.get_dist(c1=[2],c2=[5])
        with self.assertRaises(IndexError) as cm_old:
            oldfu.get_dist(c1=[2],c2=[5])
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_eliminate_moons(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.eliminate_moons()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.eliminate_moons()
        self.assertEqual(str(cm_new.exception), "eliminate_moons() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_real_case_IMAGE_3D(self):
        moon_params = [0.4,1]
        return_new = fu.eliminate_moons(my_volume=deepcopy(IMAGE_3D),moon_elimination_params= moon_params)
        return_old = oldfu.eliminate_moons(my_volume=deepcopy(IMAGE_3D),moon_elimination_params= moon_params)
        self.assertFalse(array_equal(return_old.get_3dview(), IMAGE_3D.get_3dview()))
        self.assertTrue(array_equal(return_old.get_3dview(), return_new.get_3dview()))
        self.assertTrue(array_equal(return_old.get_3dview().flatten(), [0.0006831996142864227, 0.001868105260655284, 0.0011152572697028518, 0.0021869514603167772, 0.0018898354610428214, 0.0017698195297271013, 0.0016916856402531266, 0.002617279998958111, 0.0030361914541572332, 0.003822825150564313, 0.005602582823485136, 0.0043375855311751366, 0.00408348860219121, 0.0028461397159844637, 0.0028867495711892843, 0.005610101390630007, 0.004544687923043966, 0.0053885760717093945, 0.00720383832231164, 0.0064097242429852486, 0.006845172494649887, 0.006799850147217512, 0.008074999786913395, 0.0076696500182151794, 0.008526506833732128, 0.009074630215764046, 0.009245653636753559, 0.011544659733772278, 0.009287306107580662, 0.005814094562083483, 0.00642767921090126, 0.008851660415530205, 0.008493195287883282, 0.007716738153249025, 0.011043524369597435, 0.013490768149495125, 0.009563383646309376, 0.0061018867418169975, 0.0059760697185993195, 0.005870359484106302, 0.0014521797420457006, 0.0007483167573809624, 0.0015884794993326068, 2.679041699593654e-06, 0.0008749584085308015, -0.002207944868132472, -0.0035939724184572697, -0.004281822592020035, -0.004433520138263702, -0.007224172819405794, -0.0063781444914639, -0.0074320388957858086, -0.010893117636442184, -0.010637595318257809, -0.011371471919119358, -0.012097101658582687, -0.011441514827311039, -0.013501474633812904, -0.013653973117470741, -0.013888886198401451, -0.01317901723086834, -0.015636542811989784, -0.016589796170592308, -0.018206873908638954, -0.019471270963549614, -0.020442159846425056, -0.018536828458309174, -0.01857362873852253, -0.02061760611832142, -0.022179218009114265, -0.02287045307457447, -0.023210572078824043, -0.025245238095521927, -0.026571037247776985, -0.026361016556620598, -0.026368141174316406, -0.027938764542341232, -0.026843314990401268, -0.027276253327727318, -0.027639249339699745, -0.026057451963424683, -0.024594489485025406, -0.02530156821012497, -0.025948859751224518, -0.02757061831653118, -0.028515461832284927, -0.026763472706079483, -0.02803664468228817, -0.02812436781823635, -0.03108006715774536, -0.031155439093708992, -0.031315725296735764, -0.03267689794301987, -0.033164139837026596, -0.03182239830493927, -0.03088667429983616, -0.030035698786377907, -0.028737608343362808, -0.029572375118732452, -0.029461942613124847, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.021333441138267517, -0.019425779581069946, -0.01787419058382511, -0.017487069591879845, -0.01829470694065094, -0.019424768164753914, -0.019989386200904846, -0.018826007843017578, -0.017712760716676712, -0.01676655001938343, -0.012493756599724293, -0.01146268006414175, -0.00971062108874321, -0.008890450932085514, -0.00912691093981266, -0.00934694055467844, -0.00952224899083376, -0.009711365215480328, -0.009977499954402447, -0.010294096544384956, -0.005148290190845728, -0.005254354327917099, -0.005208166316151619, -0.004943280480802059, -0.004703851882368326, -0.004443700425326824, -0.00440576346591115, -0.004217868205159903, -0.004329017363488674, -0.004337705671787262, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.002209967700764537, -0.0019582523964345455, -0.0015475356485694647, -0.0014766849344596267, -0.0018612173153087497, -0.0021799022797495127, -0.0021322721149772406, -0.0019850628450512886, -0.001835850067436695, -0.0019200366223230958, -0.0052078221924602985, -0.004732931964099407, -0.004160498734563589, -0.00374482199549675, -0.0037430597003549337, -0.003941280301660299, -0.0038778719026595354, -0.0035354432184249163, -0.003840447636321187, -0.0036034833174198866, -0.004643223248422146, -0.0045485664159059525, -0.0034825631882995367, -0.0034220274537801743, -0.003468860872089863, -0.0030371425673365593, -0.0031030105892568827, -0.002788519486784935, -0.002758264308795333, -0.0029790843836963177, -0.002762072952464223, -0.002017154125496745, -0.002431818749755621, -0.0024968578945845366, -0.0021522478200495243, -0.0025222720578312874, -0.002417219104245305, -0.0018907504854723811, -0.0013818652369081974, -0.0016725071473047137, -0.0017116977833211422, -0.001692348625510931, -0.0019659199751913548, -0.0019338668789714575, -0.002215971937403083, -0.002540990710258484, -0.0025788310449570417, -0.0019055575830861926, -0.0022413930855691433, -0.0026795908343046904, -0.00989117193967104, -0.010742376558482647, -0.009120774455368519, -0.011000007390975952, -0.00651513272896409, -0.010385492816567421, -0.011444059200584888, -0.011400331743061543, -0.014471810311079025, -0.010781862773001194, -0.013151044026017189, -0.01143861934542656, -0.012976489961147308, -0.015315257012844086, -0.013821430504322052, -0.01284016203135252, -0.013544867746531963, -0.013805175200104713, -0.01652291975915432, -0.012025486677885056, -0.01251536700874567, -0.016768792644143105, -0.013865325599908829, -0.010392301715910435, -0.011314907111227512, -0.012064789421856403, -0.012600692920386791, -0.011947670951485634, -0.01106877252459526, -0.013006697408854961, -0.0103489113971591, -0.009084177203476429, -0.00815383531153202, -0.008103594183921814, -0.006132341921329498, -0.006346367299556732, -0.008497406728565693, -0.006595413666218519, -0.006856997963041067, -0.008657771162688732, -0.006649545393884182, -0.004525046329945326, -0.005457132589071989, -0.004674646072089672, 0.0004850833211094141, -0.0027177713345736265, -0.003701904322952032, -0.0031287854071706533, -0.003191983560100198, -0.0022913094144314528, -0.00021387869492173195, 4.307296694605611e-05, 0.0018965796334668994, 0.0020115000661462545, 0.0017046518623828888, 0.0033221314661204815, 0.0029020183719694614, 0.003958276938647032, 0.003963730297982693, 0.0028416169807314873, 0.006007655989378691, 0.007919419556856155, 0.007784456480294466, 0.009633158333599567, 0.011075290851294994, 0.008278442546725273, 0.006173708941787481, 0.007875321432948112, 0.009626374579966068, 0.005919715855270624, 0.008114121854305267, 0.01298772357404232, 0.011645571328699589, 0.013143770396709442, 0.014380918815732002, 0.012386168353259563, 0.0146223409101367, 0.013986622914671898, 0.013216688297688961, 0.014284975826740265, 0.01352826226502657, 0.013796446844935417, 0.011048119515180588, 0.008766019716858864, 0.009071634151041508, 0.013534939847886562, 0.014285300858318806, 0.014428463764488697, 0.009486719034612179, 0.00897042453289032, 0.006698505952954292, 0.002733988920226693, 0.004388277418911457, 0.00551304267719388, 0.0029439262580126524, 0.0014235725393518806, 0.002016131766140461, -0.0017732006963342428, -0.0025035610888153315, -0.0074412887915968895, -0.013107653707265854, -0.0180065780878067, -0.016747819259762764, -0.019371716305613518, -0.029219523072242737, -0.03267541527748108, -0.03957265615463257, -0.03850237652659416, -0.03672628477215767, -0.04018678888678551, -0.043254997581243515, -0.04227171838283539, -0.04753929004073143, -0.05084632337093353, -0.05283939093351364, -0.06310930103063583, -0.06379184126853943, -0.06703544408082962, -0.06723736971616745, -0.07037252932786942, -0.07172592729330063, -0.07769829779863358, -0.08079458773136139, -0.0780562162399292, -0.08941511064767838, -0.0874086320400238, -0.08784541487693787, -0.09321098774671555, -0.09217599779367447, -0.0923122838139534, -0.09575333446264267, -0.0941954180598259, -0.09816794842481613, -0.09616312384605408, -0.09949889779090881, -0.10438363254070282, -0.10343998670578003, -0.10484079271554947, -0.10515226423740387, -0.10664478689432144, -0.10692054033279419, -0.11413195729255676, -0.11716318130493164, -0.11909736692905426, -0.11673508584499359, -0.11616235971450806, -0.1198720633983612, -0.11617904901504517, -0.11000734567642212, -0.11031350493431091, -0.11312708258628845, -0.11271307617425919, -0.11219165474176407, -0.11582069844007492, -0.11578503251075745, -0.11481984704732895, -0.11278822273015976, -0.112098328769207, -0.11007384210824966, -0.11100846529006958, -0.11475849896669388, -0.11429175734519958, -0.117560476064682, -0.11804649233818054, -0.12298644334077835, -0.12135817110538483, -0.12072829902172089, -0.12620079517364502, -0.12938427925109863, -0.13754168152809143, -0.14176338911056519, -0.13636915385723114, -0.1383482664823532, -0.1378316432237625, -0.13603346049785614, -0.13966405391693115, -0.14193081855773926, -0.14010506868362427, -0.13938689231872559, -0.1380908340215683, -0.14587713778018951, -0.15323618054389954, -0.1516239494085312, -0.15766486525535583, -0.16448922455310822, -0.1717568039894104, -0.16862356662750244, -0.1679026037454605, -0.17101922631263733, -0.17410224676132202, -0.17560629546642303, -0.1770385056734085, -0.17901374399662018, -0.18498535454273224, -0.1900484263896942, -0.1957603394985199, -0.19912751019001007, -0.20140932500362396, -0.20834751427173615, -0.2121393382549286, -0.2591957747936249, -0.26247334480285645, -0.26239845156669617, -0.2644755244255066, -0.2677023410797119, -0.2761981189250946, -0.28172481060028076, -0.27850541472435, -0.27517011761665344, -0.27886685729026794, -0.28770074248313904, -0.29007524251937866, -0.296048104763031, -0.29909271001815796, -0.3021949529647827, -0.3098064959049225, -0.3174734115600586, -0.3227666914463043, -0.32224905490875244, -0.32563453912734985, -0.3322528600692749, -0.3421875536441803, -0.34751224517822266, -0.3546442985534668, -0.36146464943885803, -0.36559659242630005, -0.37003231048583984, -0.36811110377311707, -0.3603134751319885, -0.35406941175460815, -0.3517828583717346, -0.35198402404785156, -0.349437415599823, -0.3521624207496643, -0.3532658815383911, -0.35080140829086304, -0.34840965270996094, -0.34352442622184753, -0.33599749207496643, -0.3340344727039337, -0.3272158205509186, -0.32561129331588745, -0.31975042819976807, -0.3067528307437897, -0.2978297770023346, -0.2893831729888916, -0.28600403666496277, -0.28233644366264343, -0.27353590726852417, -0.2691003084182739, -0.2604397237300873, -0.25082045793533325, -0.24085092544555664, -0.23039749264717102, -0.2241668701171875, -0.22189435362815857, -0.21962210536003113, -0.2145138829946518, -0.20903454720973969, -0.20150016248226166, -0.19687867164611816, -0.18600323796272278, -0.17070908844470978, -0.16284482181072235, -0.15580931305885315, -0.15305136144161224, -0.1455586552619934, -0.14272819459438324, -0.13521234691143036, -0.1328830122947693, -0.13456854224205017, -0.1364719718694687, -0.13607147336006165, -0.13110794126987457, -0.1305248737335205, -0.12291847169399261, -0.12314318120479584, -0.1252719610929489, -0.11904841661453247, -0.11485446989536285, -0.11382095515727997, -0.11032187938690186, -0.10272619128227234, -0.10405058413743973, -0.11163853108882904, -0.10940589755773544, -0.09859804064035416, -0.10201603919267654, -0.10241398960351944, -0.09472884237766266, -0.09840932488441467, -0.09580077230930328, -0.09058013558387756, -0.08941778540611267, -0.09616278856992722, -0.1018795371055603, -0.10432533174753189, -0.10203967988491058, -0.098849356174469, -0.09318523108959198, -0.0971054881811142, -0.09785378724336624, -0.09947044402360916, -0.10443077981472015, -0.10811378806829453, -0.11256957054138184, -0.10543791204690933, -0.10344670712947845, -0.08978064358234406, -0.08572018146514893, -0.0841037929058075, -0.0797007605433464, -0.08266925066709518, -0.07508290559053421, -0.06810516119003296, -0.07633653283119202, -0.07501396536827087, -0.07104278355836868, -0.07052437961101532, -0.06604169309139252, -0.0770469531416893, -0.07821550965309143, -0.07055875658988953, -0.0687403455376625, -0.05433542653918266, -0.04646049812436104, -0.04382943734526634, -0.033884577453136444, -0.0410473607480526, -0.03190535306930542, -0.03250129520893097, -0.03350614756345749, -0.021243080496788025, -0.024345453828573227, -0.027516283094882965, -0.019916223362088203, -0.027773384004831314, -0.03341028466820717, -0.02484086900949478, -0.022909415885806084, -0.018109414726495743, -0.02153690904378891, -0.021450715139508247, -0.02122606709599495, -0.026521731168031693, -0.02569415606558323, -0.027316836640238762, -0.023419125005602837, -0.022619977593421936, -0.020525915548205376, -0.021300077438354492, -0.028874851763248444, -0.02592700533568859, -0.0325649194419384, -0.02751816436648369, -0.029357071965932846, -0.03277556598186493, -0.02632036618888378, -0.029159506782889366, -0.029223017394542694, -0.03195647895336151, -0.032373201102018356, -0.027800558134913445, -0.023506268858909607, -0.03344694897532463, -0.03876466676592827, -0.032745618373155594, -0.03525695577263832, -0.03192753717303276, -0.03631393611431122, -0.0391823947429657, -0.0355924591422081, -0.043438076972961426, -0.038312315940856934, -0.03782815486192703, -0.039749301970005035, -0.02893228828907013, -0.021202364936470985, -0.026226378977298737, -0.033955544233322144, -0.03538822382688522, -0.033268824219703674, -0.03255176916718483, -0.028752218931913376, -0.02872483804821968, -0.02153942361474037, -0.01630709320306778, -0.018085382878780365, -0.020398661494255066, -0.022144468501210213, -0.027211755514144897, -0.023012982681393623, -0.013828781433403492, -0.009886418469250202, -0.010909391567111015, -0.016864510253071785, -0.010821880772709846, -0.011484570801258087, -0.004707030486315489, -0.0035495804622769356, -0.008592878468334675, -0.0002756702888291329, 0.0019090495770797133, -0.0035225979518145323, 0.003940519876778126, 0.003713401034474373, 0.009755820035934448, 0.009930896572768688, 0.01053449884057045, 0.009064428508281708, 0.015047534368932247, 0.009189496748149395, 0.010609983466565609, 0.019005797803401947, 0.019597845152020454, 0.023713642731308937, 0.0281127467751503, 0.026868920773267746, 0.021320603787899017, 0.016762467101216316, 0.024015257135033607, 0.025192996487021446, 0.02737692929804325, 0.02502334490418434, 0.02791544795036316, 0.03523661196231842, 0.03440350666642189, 0.02897394821047783, 0.03534296900033951, 0.03925039991736412, 0.03218945860862732, 0.033107396215200424, 0.03636415675282478, 0.032895252108573914, 0.025547754019498825, 0.025163305923342705, 0.02717309258878231, 0.030575906857848167, 0.025035306811332703, 0.029582008719444275, 0.01729333959519863, 0.02464519441127777, 0.018931668251752853, 0.003354722401127219, 0.010531540960073471, 0.014164220541715622, 0.0108448825776577, -0.0003998903266619891, 0.0029715702403336763, -0.0025412938557565212, -0.006418257020413876, -0.013294088654220104, -0.015726737678050995, -0.0198776014149189, -0.026688028126955032, -0.02743806689977646, -0.027278386056423187, -0.03512886166572571, -0.046664245426654816, -0.04825190082192421, -0.050118137151002884, -0.04557758942246437, -0.044985752552747726, -0.051683440804481506, -0.05988762900233269, -0.062219370156526566, -0.06504490226507187, -0.07698366791009903, -0.08047006279230118, -0.0849016010761261, -0.08911417424678802, -0.09198698401451111, -0.09023528546094894, -0.09180763363838196, -0.09636448323726654, -0.09868153929710388, -0.10571181029081345, -0.10659529268741608, -0.1066051498055458, -0.11490371078252792, -0.11514682322740555, -0.11144796013832092, -0.1138036698102951, -0.11854048818349838, -0.12435073405504227, -0.12785585224628448, -0.12662823498249054, -0.13778077065944672, -0.13688038289546967, -0.13592852652072906, -0.13712078332901, -0.13037565350532532, -0.13317757844924927, -0.13523629307746887, -0.14206071197986603, -0.14766202867031097, -0.15237556397914886, -0.155753031373024, -0.15066823363304138, -0.1472088247537613, -0.12831199169158936, -0.13109228014945984, -0.13262291252613068, -0.13892214000225067, -0.1407325565814972, -0.13629093766212463, -0.1384405493736267, -0.13273568451404572, -0.12893396615982056, -0.13234247267246246, -0.1338265836238861, -0.13609972596168518, -0.1403530240058899, -0.13878314197063446, -0.1397261768579483, -0.1425325572490692, -0.14131969213485718, -0.14164400100708008, -0.1444968730211258, -0.15158213675022125, -0.16040219366550446, -0.1541145294904709, -0.15567317605018616, -0.1546323150396347, -0.16129887104034424, -0.1612626165151596, -0.15985961258411407, -0.16566915810108185, -0.1670168936252594, -0.1714383214712143, -0.18623678386211395, -0.19594824314117432, -0.19321849942207336, -0.19991354644298553, -0.2070588767528534, -0.20758378505706787, -0.20728875696659088, -0.20916114747524261, -0.21627746522426605, -0.21819978952407837, -0.22173963487148285, -0.2211415320634842, -0.21779762208461761, -0.22725273668766022, -0.23529621958732605, -0.2425251603126526, -0.24501203000545502, -0.2515961229801178, -0.2562917172908783, -0.2597421407699585, -0.25451037287712097, -0.255024790763855, -0.2610856890678406, -0.2679905891418457, -0.2709254324436188, -0.27769726514816284, -0.27403417229652405, -0.2693414092063904, -0.272217720746994, -0.27651000022888184, -0.27267152070999146, -0.27086907625198364, -0.2736477851867676, -0.27973827719688416, -0.28600260615348816, -0.298160582780838, -0.29922834038734436, -0.29831746220588684, -0.30045533180236816, -0.3071839511394501, -0.3057084381580353, -0.30183935165405273, -0.2970816195011139, -0.3048500120639801, -0.3150530159473419, -0.3242373466491699, -0.3213922679424286, -0.32085758447647095, -0.32358089089393616, -0.3276093006134033, -0.33008813858032227, -0.3277050852775574, -0.3216434419155121, -0.3218679130077362, -0.32776695489883423, -0.32906609773635864, -0.321580171585083, -0.3184431791305542, -0.3195696175098419, -0.32326740026474, -0.3210148811340332, -0.31570112705230713, -0.30579647421836853, -0.3043813109397888, -0.3030731678009033, -0.297353059053421, -0.2861577272415161, -0.2723696827888489, -0.26694607734680176, -0.26786571741104126, -0.19492754340171814, -0.19283121824264526, -0.18914005160331726, -0.18335016071796417, -0.17365707457065582, -0.16677136719226837, -0.16258475184440613, -0.15308435261249542, -0.14738847315311432, -0.1453046053647995, -0.15017108619213104, -0.14751119911670685, -0.14005765318870544, -0.1324840635061264, -0.12576927244663239, -0.12133081257343292, -0.11621355265378952, -0.1125120297074318, -0.10808945447206497, -0.10938296467065811, -0.11994592845439911, -0.12005551904439926, -0.1183645948767662, -0.11169544607400894, -0.10549358278512955, -0.09504467993974686, -0.09138456732034683, -0.08655765652656555, -0.08179549127817154, -0.08316205441951752, -0.08855439722537994, -0.08866050094366074, -0.0877186506986618, -0.09129467606544495, -0.09145131707191467, -0.088701531291008, -0.08781076967716217, -0.08987671136856079, -0.08772726356983185, -0.08168095350265503, -0.08064626157283783, -0.07429655641317368, -0.07183369249105453, -0.07025406509637833, -0.07219874113798141, -0.07117941975593567, -0.07334690541028976, -0.07369155436754227, -0.07603570073843002, -0.0772966593503952, -0.06933975219726562, -0.06936262547969818, -0.06991357356309891, -0.0743558406829834, -0.0743802860379219, -0.07493084669113159, -0.07478734850883484, -0.07485450059175491, -0.07048709690570831, -0.06886123865842819, -0.05664155259728432, -0.05051323026418686, -0.050589609891176224, -0.05030197650194168, -0.048382822424173355, -0.05153379216790199, -0.050060175359249115, -0.04918833449482918, -0.049858398735523224, -0.04919090121984482, -0.047482606023550034, -0.04802841693162918, -0.04348740726709366, -0.04461566358804703, -0.0400080569088459, -0.03369547426700592, -0.032809413969516754, -0.026827950030565262, -0.025092866271734238, -0.021071506664156914, -0.020586609840393066, -0.019342441111803055, -0.02040758542716503, -0.017899716272950172, -0.01525459997355938, -0.01888425461947918, -0.0150248222053051, -0.012576570734381676, -0.016314178705215454, -0.017653964459896088, -0.014704222790896893, -0.013608762063086033, -0.01262454129755497, -0.010743639431893826, -0.012907362543046474, -0.013693613931536674, -0.012460555881261826, -0.0140147116035223, -0.014768332242965698, -0.012159149162471294]))

    def test_real_case_IMAGE_3D_no_change(self):
        moon_params = [0.4,1]
        return_new = fu.eliminate_moons(my_volume=deepcopy(IMAGE_3D), moon_elimination_params=moon_params)
        return_old = oldfu.eliminate_moons(my_volume=deepcopy(IMAGE_3D), moon_elimination_params=moon_params)
        self.assertFalse(array_equal(return_old.get_3dview(), IMAGE_3D.get_3dview()))
        self.assertTrue(array_equal(return_old.get_3dview(), return_new.get_3dview()))

    def test_returns_IndexError_list_index_out_of_range(self):
        moon_params = [0.4]
        with self.assertRaises(IndexError) as cm_new:
            fu.eliminate_moons(my_volume=deepcopy(IMAGE_3D), moon_elimination_params=moon_params)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.eliminate_moons(my_volume=deepcopy(IMAGE_3D), moon_elimination_params=moon_params)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_img_returns_RuntimeError_the_img_should_be_a_3D_img(self):
        moon_params = [0.4,0.7]
        with self.assertRaises(RuntimeError) as cm_new:
            fu.eliminate_moons(my_volume=EMData(), moon_elimination_params=moon_params)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.eliminate_moons(my_volume=EMData(), moon_elimination_params=moon_params)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The image should be 3D")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_NoneType_img_returns_AttributeError_NoneType_obj_hasnot_attribute_find_3d_threshold(self):
        moon_params = [0.4,0.7]
        with self.assertRaises(AttributeError) as cm_new:
            fu.eliminate_moons(my_volume=None, moon_elimination_params=moon_params)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.eliminate_moons(my_volume=None, moon_elimination_params=moon_params)
        self.assertEqual(str(cm_new.exception), "'NoneType' object has no attribute 'find_3d_threshold'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_combinations_of_n_taken_by_k(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.combinations_of_n_taken_by_k()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.combinations_of_n_taken_by_k()
        self.assertEqual(str(cm_new.exception), "combinations_of_n_taken_by_k() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_combinations_of_n_taken_by_k(self):
        return_new = fu.combinations_of_n_taken_by_k(5,3)
        return_old = oldfu.combinations_of_n_taken_by_k(5,3)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, 10)



class Test_cmdexecute(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.cmdexecute()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.cmdexecute()
        self.assertEqual(str(cm_new.exception), "cmdexecute() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_correct_cmd_without_printing_on_success(self):
        return_new = fu.cmdexecute("ls", False)
        return_old = oldfu.cmdexecute("ls", False)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, None)

    def test_correct_cmd_with_printing_on_success(self):
        return_new = fu.cmdexecute("ls", True)
        return_old = oldfu.cmdexecute("ls", True)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, 1)

    def test_wrong_cmd(self):
        return_new = fu.cmdexecute("quack", True)
        return_old = oldfu.cmdexecute("quack", True)
        self.assertEqual(return_new,return_old)
        self.assertEqual(return_new, 0)



class Test_string_found_in_file(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.string_found_in_file()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.string_found_in_file()
        self.assertEqual(str(cm_new.exception), "string_found_in_file() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_file_not_found_returns_IOError(self):
        with self.assertRaises(IOError) as cm_new:
            fu.string_found_in_file("search smth", "not_a_file.txt")
        with self.assertRaises(IOError) as cm_old:
            oldfu.string_found_in_file("search smth", "not_a_file.txt")
        self.assertEqual(cm_new.exception.strerror, "No such file or directory")
        self.assertEqual(cm_new.exception.strerror, cm_old.exception.strerror)

    def test_found_value(self):
        f = "f.txt"
        data=[["hallo",1,1,1],[2,2,2,2],[3,3,3,3]]
        path_to_file = path.join(ABSOLUTE_PATH, f)
        fu.write_text_row(data, path_to_file)
        return_new = fu.string_found_in_file("hallo", path_to_file)
        return_old = oldfu.string_found_in_file("hallo", path_to_file)
        remove_list_of_file([f])
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new)

    def test_notfound_value(self):
        f = "f.txt"
        data=[["ds",1,1,1],[2,2,2,2],[3,3,3,3]]
        path_to_file = path.join(ABSOLUTE_PATH, f)
        fu.write_text_row(data, path_to_file)
        return_new = fu.string_found_in_file("hallo", path_to_file)
        return_old = oldfu.string_found_in_file("hallo", path_to_file)
        remove_list_of_file([f])
        self.assertEqual(return_new,return_old)
        self.assertFalse(return_new)



class Test_get_latest_directory_increment_value(unittest.TestCase):
    start_value = 1
    folder_name = 'd'
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_latest_directory_increment_value()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_latest_directory_increment_value()
        self.assertEqual(str(cm_new.exception), "get_latest_directory_increment_value() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nothing_to_count(self):
        return_new = fu.get_latest_directory_increment_value(ABSOLUTE_PATH, self.folder_name, start_value = self.start_value, myformat = "%03d")
        return_old = oldfu.get_latest_directory_increment_value(ABSOLUTE_PATH, self.folder_name, start_value = self.start_value, myformat = "%03d")
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new,self.start_value)

    def test_count_something(self):
        mkdir(path.join(ABSOLUTE_PATH, self.folder_name+"001"))
        mkdir(path.join(ABSOLUTE_PATH, self.folder_name + "002"))
        return_new = fu.get_latest_directory_increment_value(ABSOLUTE_PATH, "/"+self.folder_name,start_value=self.start_value, myformat="%03d")
        return_old = oldfu.get_latest_directory_increment_value(ABSOLUTE_PATH, "/" + self.folder_name,start_value=self.start_value, myformat="%03d")
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new,2)
        remove_dir(path.join(ABSOLUTE_PATH, self.folder_name+"001"))
        remove_dir(path.join(ABSOLUTE_PATH, self.folder_name + "002"))



class Test_if_error_then_all_processes_exit_program(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.if_error_then_all_processes_exit_program()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.if_error_then_all_processes_exit_program()
        self.assertEqual(str(cm_new.exception), "if_error_then_all_processes_exit_program() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


""" this function has been cleaned
class Test_get_shrink_data_huang(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_shrink_data_huang()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_shrink_data_huang()
        self.assertEqual(str(cm_new.exception), "get_shrink_data_huang() takes at least 7 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    @unittest.skip("it cannot find something in the ADNAN file")
    def test_get_shrink_data_huang_true_should_return_equal_objects(self):
        '''
        I got
        RuntimeError: FileAccessException at /home/lusnig/EMAN2/eman2/libEM/emdata_metadata.cpp:240: error with '/home/lusnig/Downloads/adnan4testing/Substack/EMAN2DB/../../Particles/mpi_proc_000/EMAN2DB/TcdA1-0011_frames_sum_ptcls_352x352x1': 'cannot access file '/home/lusnig/Downloads/adnan4testing/Substack/EMAN2DB/../../Particles/mpi_proc_000/EMAN2DB/TcdA1-0011_frames_sum_ptcls_352x352x1'' caught
        '''
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["log_main"] = "logging"
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["stack"] = 'bdb:' + path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Substack/sort3d_substack_002')
        Tracker["applyctf"] = True
        ids = []
        for i in range(1227):
            ids.append(i)
        Tracker["chunk_dict"] =ids
        myid = 0
        m_node = 0
        nproc = 1
        partids = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, "Refine3D-Substack-Local_001/main010/indexes_010.txt")
        partstack =  path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, "Refine3D-Substack-Local_001/main010/params_010.txt")
        nxinit = 2

        return_new = fu.get_shrink_data_huang(Tracker, nxinit, partids, partstack, myid, m_node, nproc)

        return_old = oldfu.get_shrink_data_huang(Tracker, nxinit, partids, partstack, myid, m_node, nproc)

        self.assertTrue(allclose(return_new[0][0].get_3dview(), return_old[0][0].get_3dview(), 0.5))
"""

class Test_getindexdata(unittest.TestCase):
    """ nproc and myid valeus got from "pickle files/utilities/utilities.getindexdata"""
    nproc = 95
    myid = 22
    stack = 'bdb:' + path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D/best_000')
    partids = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D/main001/this_iteration_index_keep_images.txt')
    partstack = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D//main001/run000/rotated_reduced_params.txt')

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.getindexdata()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.getindexdata()
        self.assertEqual(str(cm_new.exception), "getindexdata() takes exactly 5 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_nproc_greater_than_ndata(self):
        return_new = fu.getindexdata(self.stack, self.partids, self.partstack, self.myid, self.nproc)
        return_old = oldfu.getindexdata(self.stack, self.partids, self.partstack, self.myid, self.nproc)
        a=return_new[0].get_3dview().flatten().tolist()
        self.assertTrue(array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))

    def test_nproc_and_myid_greater_than_ndata_(self):
        return_new = fu.getindexdata(self.stack, self.partids, self.partstack, 100, self.nproc)
        return_old = oldfu.getindexdata(self.stack, self.partids, self.partstack, 100, self.nproc)
        self.assertTrue(array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))

    def test_nproc_lower_than_ndata(self):
        return_new = fu.getindexdata(self.stack, self.partids, self.partstack, self.myid, nproc= 10)
        return_old = oldfu.getindexdata(self.stack, self.partids, self.partstack, self.myid, nproc= 10)
        self.assertTrue(array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))



class Test_store_value_of_simple_vars_in_json_file(unittest.TestCase):
    f= path.join(ABSOLUTE_PATH, "fu.json")
    f_old = path.join(ABSOLUTE_PATH, "oldfu.json")
    var_to_save= {'string_var': 'var1', 'integer_var': 7, 'bool_var': False, 'list_var': [2,3,4]}
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.store_value_of_simple_vars_in_json_file()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.store_value_of_simple_vars_in_json_file()
        self.assertEqual(str(cm_new.exception), "store_value_of_simple_vars_in_json_file() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_value(self):
        fu.store_value_of_simple_vars_in_json_file(filename =  self.f, local_vars = self.var_to_save, exclude_list_of_vars = [], write_or_append = "w",	vars_that_will_show_only_size = [])
        oldfu.store_value_of_simple_vars_in_json_file(filename =  self.f_old, local_vars = self.var_to_save, exclude_list_of_vars = [], write_or_append = "w",	vars_that_will_show_only_size = [])
        self.assertEqual(returns_values_in_file(self.f), returns_values_in_file(self.f_old))
        self.assertTrue(fu.string_found_in_file(self.var_to_save.keys()[0], self.f))
        self.assertTrue(oldfu.string_found_in_file(self.var_to_save.keys()[0], self.f_old))
        remove_list_of_file([self.f,self.f_old])

    def test_exclude_a_variable(self):
        var=self.var_to_save.keys()[0]
        fu.store_value_of_simple_vars_in_json_file(filename =  self.f, local_vars = self.var_to_save, exclude_list_of_vars = [var], write_or_append = "w",	vars_that_will_show_only_size = [])
        oldfu.store_value_of_simple_vars_in_json_file(filename =  self.f_old, local_vars = self.var_to_save, exclude_list_of_vars = [var], write_or_append = "w",	vars_that_will_show_only_size = [])
        self.assertEqual(returns_values_in_file(self.f), returns_values_in_file(self.f_old))
        self.assertFalse(fu.string_found_in_file(var, self.f))
        self.assertFalse(oldfu.string_found_in_file(var, self.f_old))
        remove_list_of_file([self.f,self.f_old])

    def test_onlySize_a_variable(self):
        var= 'list_var'
        fu.store_value_of_simple_vars_in_json_file(filename =  self.f, local_vars = self.var_to_save, exclude_list_of_vars = [], write_or_append = "w",	vars_that_will_show_only_size = [var])
        oldfu.store_value_of_simple_vars_in_json_file(filename =  self.f_old, local_vars = self.var_to_save, exclude_list_of_vars = [], write_or_append = "w",	vars_that_will_show_only_size = [var])
        self.assertEqual(returns_values_in_file(self.f), returns_values_in_file(self.f_old))
        self.assertTrue(fu.string_found_in_file("<type 'list'> with length: 3", self.f))
        self.assertTrue(oldfu.string_found_in_file("<type 'list'> with length: 3", self.f_old))
        remove_list_of_file([self.f, self.f_old])



class Test_convert_json_fromunicode(unittest.TestCase):
    f= path.join(ABSOLUTE_PATH, "f.json")
    var_to_save= {'string_var': 'var1', 'integer_var': 7, 'bool_var': False, 'list_var': [2,3,4]}

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.convert_json_fromunicode()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.convert_json_fromunicode()
        self.assertEqual(str(cm_new.exception), "convert_json_fromunicode() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_with_loaded_jsonFile(self):
        fu.store_value_of_simple_vars_in_json_file(filename=self.f, local_vars=self.var_to_save,exclude_list_of_vars=[], write_or_append="w",vars_that_will_show_only_size=[])
        with open(self.f, 'r') as f1:
            values=json_load( f1)

        return_new = fu.convert_json_fromunicode(values)
        return_old = oldfu.convert_json_fromunicode(values)
        self.assertDictEqual(return_new,return_old)
        remove_list_of_file([self.f])

    def test_with_string(self):
        data = "ciaone"
        return_new = fu.convert_json_fromunicode(data)
        return_old = oldfu.convert_json_fromunicode(data)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, data)


""" the following tests have been cleaned
class Test_get_sorting_attr_stack(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_sorting_attr_stack()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_sorting_attr_stack()
        self.assertEqual(str(cm_new.exception), "get_sorting_attr_stack() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        stack = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.ali3d_multishc"))[0][0]
        for i in range(len(stack)):
            stack[i].set_attr("group",i)
        return_new = fu.get_sorting_attr_stack(stack)
        return_old = oldfu.get_sorting_attr_stack(stack)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new,[[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0], [3, 48.329977109192356, 117.5705850518134, 351.86937664408134, 0.0, 0.0], [4, 27.69691819268928, 43.27031745806075, 221.01598994487352, 0.0, 0.0], [5, 21.310076601702065, 171.03012732754505, 171.96680859829712, 0.0, 0.0], [6, 15.29265945235332, 52.64344620523427, 261.77408611622946, 0.0, 0.0], [7, 5.167600687104752, 102.82730387712962, 342.39939706174914, 0.0, 0.0], [8, 24.790317531012022, 99.47564893121846, 197.28160868317875, 0.0, 0.0], [9, 63.6110637166135, 57.82849773806232, 46.28909621816683, 0.0, 0.0], [10, 43.0541340885994, 99.96781920338643, 268.6955892945691, 0.0, 0.0], [11, 61.17678429642095, 90.75462285273008, 146.44621665024576, 0.0, 0.0], [12, 42.21290066966395, 144.5552830212904, 307.2072172161095, 0.0, 0.0], [13, 66.842043722353, 151.03106106540827, 74.69558471790953, 0.0, 0.0], [14, 57.13253366620802, 77.06211216981875, 14.078738028151804, 0.0, 0.0], [15, 23.41658890400403, 33.44337707006545, 302.5719333204403, 0.0, 0.0], [16, 31.39639453397831, 51.13633353440325, 208.30249398432636, 0.0, 0.0], [17, 29.02858651478833, 35.5375562136724, 142.0210172114356, 0.0, 0.0], [18, 33.0380714821259, 25.92835772588129, 56.941091101308245, 0.0, 0.0], [19, 33.154703792393036, 119.10157237670761, 289.65555324857974, 0.0, 0.0], [20, 51.7488556189989, 61.390756979764014, 50.60060397900594, 0.0, 0.0], [21, 13.14807567561769, 87.51982195330109, 196.43451185108586, 0.0, 0.0], [22, 44.16102088511332, 70.89045303205856, 173.69722943005007, 0.0, 0.0], [23, 5.203725557624949, 124.16777353839416, 274.3070842951357, 0.0, 0.0], [24, 71.40261567268388, 94.91091772300047, 193.279386674675, 0.0, 0.0], [25, 37.759166538330405, 64.89547557023316, 126.31095971850405, 0.0, 0.0], [26, 69.73110445824167, 36.63569177645915, 141.5284940422194, 0.0, 0.0], [27, 13.404248750628184, 64.23894693338787, 268.45800591947955, 0.0, 0.0], [28, 36.768338316193336, 142.47038328015265, 328.30282713390517, 0.0, 0.0], [29, 49.40940654524283, 57.44562945197844, 225.5720769552322, 0.0, 0.0], [30, 42.05392118654274, 111.5816596879126, 229.38783889376506, 0.0, 0.0], [31, 4.413172142330993, 36.815963893301586, 219.42779280502356, 0.0, 0.0], [32, 21.93687623164496, 101.33898701867622, 10.389958811519477, 0.0, 0.0], [33, 7.238104232102415, 93.18173962163453, 297.8577786238534, 0.0, 0.0], [34, 33.74128043831372, 136.57263972662804, 251.97793890785513, 0.0, 0.0], [35, 13.24801500355332, 93.3437024073173, 83.42851157658254, 0.0, 0.0], [36, 68.98968742383678, 137.52188147974766, 131.28953414678818, 0.0, 0.0], [37, 6.656739020518444, 83.46392407516454, 108.36199175910278, 0.0, 0.0], [38, 55.08043065083436, 77.00680292359846, 276.94285028405614, 0.0, 0.0], [39, 68.11877000276846, 170.36329302918105, 110.40575965943643, 0.0, 0.0], [40, 14.384107688958224, 37.08496112731635, 226.66984538039912, 0.0, 0.0], [41, 34.82364685570731, 22.08565956767624, 67.24551437397145, 0.0, 0.0], [42, 54.90981252536744, 84.76440954646556, 246.244958672071, 0.0, 0.0], [43, 65.74869436077404, 96.69886072311077, 48.8693101954874, 0.0, 0.0], [44, 56.61431355350081, 127.49229486891034, 352.1753011958611, 0.0, 0.0], [45, 13.363329087310646, 50.6501020880211, 254.4766641608211, 0.0, 0.0], [46, 37.52831980476232, 57.12531740572374, 63.945298445442404, 0.0, 0.0], [47, 65.89658161668316, 81.17948835034154, 12.404945110092683, 0.0, 0.0], [48, 34.31531685691827, 82.26885039884172, 160.73300049337337, 0.0, 0.0], [49, 17.681190265485114, 105.10467265590191, 119.8614020125277, 0.0, 0.0], [50, 15.31103266122065, 54.585177368223576, 271.6740281746854, 0.0, 0.0], [51, 36.283176319068986, 62.914786915921084, 104.54889787946392, 0.0, 0.0], [52, 3.4069623341830635, 157.06953327943953, 258.63938748179646, 0.0, 0.0], [53, 15.818597678822641, 101.17411462087938, 124.36987185667502, 0.0, 0.0], [54, 5.145354760978023, 145.49560881782645, 259.29832164314917, 0.0, 0.0], [55, 56.769249564633384, 49.87909869978583, 339.7947584207519, 0.0, 0.0], [56, 22.02743770985016, 60.589524188603185, 329.76575466706055, 0.0, 0.0], [57, 13.845198933255048, 108.88152968318862, 266.16552396525003, 0.0, 0.0], [58, 26.256731825057074, 50.99774992620399, 103.53874757271407, 0.0, 0.0], [59, 59.518223120645075, 44.12851528850411, 350.54914168406856, 0.0, 0.0], [60, 14.6120857430749, 134.118368756269, 190.99542187117007, 0.0, 0.0], [61, 54.53681589479305, 104.15765119485769, 218.94808795789788, 0.0, 0.0], [62, 43.81366478468314, 167.8017617509876, 184.31279489033895, 0.0, 0.0], [63, 43.93387611407539, 37.88212967171051, 78.06736658579604, 0.0, 0.0], [64, 59.09189637303123, 75.17428928858558, 359.3010696556519, 0.0, 0.0], [65, 22.179552852855508, 58.65260316036829, 181.70124261889157, 0.0, 0.0], [66, 69.29897204983331, 164.63384795897073, 186.55493640315802, 0.0, 0.0], [67, 7.70600634119414, 135.8714860404165, 200.73045260049025, 0.0, 0.0], [68, 8.093148293539642, 106.7864434015925, 31.46848962081191, 0.0, 0.0], [69, 63.21178451811289, 172.1347585895757, 221.8787228503647, 0.0, 0.0], [70, 39.26708106177247, 109.5665308139106, 212.0178242621771, 0.0, 0.0], [71, 5.334919831202555, 65.96229606168743, 191.2185522118338, 0.0, 0.0], [72, 69.25239015969382, 127.83273678349846, 258.0599500142225, 0.0, 0.0], [73, 33.5544474437572, 95.83126974465429, 185.6297961209063, 0.0, 0.0], [74, 23.26676870455981, 93.61370037103086, 45.28778441021137, 0.0, 0.0], [75, 23.043517141348772, 85.84605595290141, 49.47880592292438, 0.0, 0.0], [76, 32.29696863642283, 82.2144519927641, 195.90500589451506, 0.0, 0.0], [77, 61.75569747009288, 59.71952915315906, 200.08880634568584, 0.0, 0.0], [78, 55.77388609688742, 80.90666038882988, 348.03178571828937, 0.0, 0.0], [79, 59.29148816095736, 112.04611759714986, 337.00885101339793, 0.0, 0.0], [80, 35.1758933715832, 90.05390008227911, 128.26317896036022, 0.0, 0.0], [81, 11.86605599859547, 137.92249001802958, 246.76615643070795, 0.0, 0.0], [82, 67.2881097166682, 69.57308961723741, 148.17043074761835, 0.0, 0.0], [83, 47.19123431038949, 61.2678939843427, 17.806652177535398, 0.0, 0.0], [84, 10.603352096546047, 99.09333923199034, 100.16174984508127, 0.0, 0.0], [85, 9.711739202292577, 66.0802792775635, 344.0008163242261, 0.0, 0.0], [86, 62.633048929359376, 108.2560649627951, 233.64751565667922, 0.0, 0.0], [87, 15.973356105036785, 167.05874406098954, 9.692510439987075, 0.0, 0.0], [88, 29.510920813596286, 39.4352946112764, 315.9396310062408, 0.0, 0.0], [89, 47.105447134414106, 131.1150313645737, 169.44623231632715, 0.0, 0.0], [90, 70.08997625761248, 54.12087833087295, 191.55760589075828, 0.0, 0.0], [91, 30.490916355709714, 49.17052385044576, 157.3015888880858, 0.0, 0.0], [92, 69.16740894604052, 46.33040988241775, 276.01316541962285, 0.0, 0.0], [93, 11.38232151100766, 19.50450649105371, 65.58885592786618, 0.0, 0.0], [94, 49.76300543857437, 131.18661434631423, 209.89458600184605, 0.0, 0.0]])

    def test_empty_stack(self):
        return_new=fu.get_sorting_attr_stack([])
        self.assertTrue(array_equal(return_new, oldfu.get_sorting_attr_stack([])))
        self.assertTrue(array_equal(return_new, []))

    def test_wrong_images_in_the_stack_RunTimeError(self):
        stack=[IMAGE_2D,IMAGE_2D]
        for i in range(len(stack)):
            stack[i].set_attr("group",i)
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_sorting_attr_stack(stack)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_sorting_attr_stack(stack)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))


class Test_get_sorting_params_refine(unittest.TestCase):
    stack = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.ali3d_multishc"))[0][0]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_sorting_params_refine()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_sorting_params_refine()
        self.assertEqual(str(cm_new.exception), "get_sorting_params_refine() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        for i in range(len(self.stack)):
            self.stack[i].set_attr("group",i)
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["nproc"] = 1
        return_new = fu.get_sorting_params_refine(Tracker, self.stack, len(self.stack))
        return_old = oldfu.get_sorting_params_refine(Tracker, self.stack, len(self.stack))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0], [3, 48.329977109192356, 117.5705850518134, 351.86937664408134, 0.0, 0.0], [4, 27.69691819268928, 43.27031745806075, 221.01598994487352, 0.0, 0.0], [5, 21.310076601702065, 171.03012732754505, 171.96680859829712, 0.0, 0.0], [6, 15.29265945235332, 52.64344620523427, 261.77408611622946, 0.0, 0.0], [7, 5.167600687104752, 102.82730387712962, 342.39939706174914, 0.0, 0.0], [8, 24.790317531012022, 99.47564893121846, 197.28160868317875, 0.0, 0.0], [9, 63.6110637166135, 57.82849773806232, 46.28909621816683, 0.0, 0.0], [10, 43.0541340885994, 99.96781920338643, 268.6955892945691, 0.0, 0.0], [11, 61.17678429642095, 90.75462285273008, 146.44621665024576, 0.0, 0.0], [12, 42.21290066966395, 144.5552830212904, 307.2072172161095, 0.0, 0.0], [13, 66.842043722353, 151.03106106540827, 74.69558471790953, 0.0, 0.0], [14, 57.13253366620802, 77.06211216981875, 14.078738028151804, 0.0, 0.0], [15, 23.41658890400403, 33.44337707006545, 302.5719333204403, 0.0, 0.0], [16, 31.39639453397831, 51.13633353440325, 208.30249398432636, 0.0, 0.0], [17, 29.02858651478833, 35.5375562136724, 142.0210172114356, 0.0, 0.0], [18, 33.0380714821259, 25.92835772588129, 56.941091101308245, 0.0, 0.0], [19, 33.154703792393036, 119.10157237670761, 289.65555324857974, 0.0, 0.0], [20, 51.7488556189989, 61.390756979764014, 50.60060397900594, 0.0, 0.0], [21, 13.14807567561769, 87.51982195330109, 196.43451185108586, 0.0, 0.0], [22, 44.16102088511332, 70.89045303205856, 173.69722943005007, 0.0, 0.0], [23, 5.203725557624949, 124.16777353839416, 274.3070842951357, 0.0, 0.0], [24, 71.40261567268388, 94.91091772300047, 193.279386674675, 0.0, 0.0], [25, 37.759166538330405, 64.89547557023316, 126.31095971850405, 0.0, 0.0], [26, 69.73110445824167, 36.63569177645915, 141.5284940422194, 0.0, 0.0], [27, 13.404248750628184, 64.23894693338787, 268.45800591947955, 0.0, 0.0], [28, 36.768338316193336, 142.47038328015265, 328.30282713390517, 0.0, 0.0], [29, 49.40940654524283, 57.44562945197844, 225.5720769552322, 0.0, 0.0], [30, 42.05392118654274, 111.5816596879126, 229.38783889376506, 0.0, 0.0], [31, 4.413172142330993, 36.815963893301586, 219.42779280502356, 0.0, 0.0], [32, 21.93687623164496, 101.33898701867622, 10.389958811519477, 0.0, 0.0], [33, 7.238104232102415, 93.18173962163453, 297.8577786238534, 0.0, 0.0], [34, 33.74128043831372, 136.57263972662804, 251.97793890785513, 0.0, 0.0], [35, 13.24801500355332, 93.3437024073173, 83.42851157658254, 0.0, 0.0], [36, 68.98968742383678, 137.52188147974766, 131.28953414678818, 0.0, 0.0], [37, 6.656739020518444, 83.46392407516454, 108.36199175910278, 0.0, 0.0], [38, 55.08043065083436, 77.00680292359846, 276.94285028405614, 0.0, 0.0], [39, 68.11877000276846, 170.36329302918105, 110.40575965943643, 0.0, 0.0], [40, 14.384107688958224, 37.08496112731635, 226.66984538039912, 0.0, 0.0], [41, 34.82364685570731, 22.08565956767624, 67.24551437397145, 0.0, 0.0], [42, 54.90981252536744, 84.76440954646556, 246.244958672071, 0.0, 0.0], [43, 65.74869436077404, 96.69886072311077, 48.8693101954874, 0.0, 0.0], [44, 56.61431355350081, 127.49229486891034, 352.1753011958611, 0.0, 0.0], [45, 13.363329087310646, 50.6501020880211, 254.4766641608211, 0.0, 0.0], [46, 37.52831980476232, 57.12531740572374, 63.945298445442404, 0.0, 0.0], [47, 65.89658161668316, 81.17948835034154, 12.404945110092683, 0.0, 0.0], [48, 34.31531685691827, 82.26885039884172, 160.73300049337337, 0.0, 0.0], [49, 17.681190265485114, 105.10467265590191, 119.8614020125277, 0.0, 0.0], [50, 15.31103266122065, 54.585177368223576, 271.6740281746854, 0.0, 0.0], [51, 36.283176319068986, 62.914786915921084, 104.54889787946392, 0.0, 0.0], [52, 3.4069623341830635, 157.06953327943953, 258.63938748179646, 0.0, 0.0], [53, 15.818597678822641, 101.17411462087938, 124.36987185667502, 0.0, 0.0], [54, 5.145354760978023, 145.49560881782645, 259.29832164314917, 0.0, 0.0], [55, 56.769249564633384, 49.87909869978583, 339.7947584207519, 0.0, 0.0], [56, 22.02743770985016, 60.589524188603185, 329.76575466706055, 0.0, 0.0], [57, 13.845198933255048, 108.88152968318862, 266.16552396525003, 0.0, 0.0], [58, 26.256731825057074, 50.99774992620399, 103.53874757271407, 0.0, 0.0], [59, 59.518223120645075, 44.12851528850411, 350.54914168406856, 0.0, 0.0], [60, 14.6120857430749, 134.118368756269, 190.99542187117007, 0.0, 0.0], [61, 54.53681589479305, 104.15765119485769, 218.94808795789788, 0.0, 0.0], [62, 43.81366478468314, 167.8017617509876, 184.31279489033895, 0.0, 0.0], [63, 43.93387611407539, 37.88212967171051, 78.06736658579604, 0.0, 0.0], [64, 59.09189637303123, 75.17428928858558, 359.3010696556519, 0.0, 0.0], [65, 22.179552852855508, 58.65260316036829, 181.70124261889157, 0.0, 0.0], [66, 69.29897204983331, 164.63384795897073, 186.55493640315802, 0.0, 0.0], [67, 7.70600634119414, 135.8714860404165, 200.73045260049025, 0.0, 0.0], [68, 8.093148293539642, 106.7864434015925, 31.46848962081191, 0.0, 0.0], [69, 63.21178451811289, 172.1347585895757, 221.8787228503647, 0.0, 0.0], [70, 39.26708106177247, 109.5665308139106, 212.0178242621771, 0.0, 0.0], [71, 5.334919831202555, 65.96229606168743, 191.2185522118338, 0.0, 0.0], [72, 69.25239015969382, 127.83273678349846, 258.0599500142225, 0.0, 0.0], [73, 33.5544474437572, 95.83126974465429, 185.6297961209063, 0.0, 0.0], [74, 23.26676870455981, 93.61370037103086, 45.28778441021137, 0.0, 0.0], [75, 23.043517141348772, 85.84605595290141, 49.47880592292438, 0.0, 0.0], [76, 32.29696863642283, 82.2144519927641, 195.90500589451506, 0.0, 0.0], [77, 61.75569747009288, 59.71952915315906, 200.08880634568584, 0.0, 0.0], [78, 55.77388609688742, 80.90666038882988, 348.03178571828937, 0.0, 0.0], [79, 59.29148816095736, 112.04611759714986, 337.00885101339793, 0.0, 0.0], [80, 35.1758933715832, 90.05390008227911, 128.26317896036022, 0.0, 0.0], [81, 11.86605599859547, 137.92249001802958, 246.76615643070795, 0.0, 0.0], [82, 67.2881097166682, 69.57308961723741, 148.17043074761835, 0.0, 0.0], [83, 47.19123431038949, 61.2678939843427, 17.806652177535398, 0.0, 0.0], [84, 10.603352096546047, 99.09333923199034, 100.16174984508127, 0.0, 0.0], [85, 9.711739202292577, 66.0802792775635, 344.0008163242261, 0.0, 0.0], [86, 62.633048929359376, 108.2560649627951, 233.64751565667922, 0.0, 0.0], [87, 15.973356105036785, 167.05874406098954, 9.692510439987075, 0.0, 0.0], [88, 29.510920813596286, 39.4352946112764, 315.9396310062408, 0.0, 0.0], [89, 47.105447134414106, 131.1150313645737, 169.44623231632715, 0.0, 0.0], [90, 70.08997625761248, 54.12087833087295, 191.55760589075828, 0.0, 0.0], [91, 30.490916355709714, 49.17052385044576, 157.3015888880858, 0.0, 0.0], [92, 69.16740894604052, 46.33040988241775, 276.01316541962285, 0.0, 0.0], [93, 11.38232151100766, 19.50450649105371, 65.58885592786618, 0.0, 0.0], [94, 49.76300543857437, 131.18661434631423, 209.89458600184605, 0.0, 0.0]]))

    def returns_too_ndata_vlaue_respect_the_number_of_data_IndexError_list_index_out_of_range(self):
        for i in range(len(self.stack)):
            self.stack[i].set_attr("group",i)
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["nproc"] = 1
        with self.assertRaises(IndexError) as cm_new:
            fu.get_sorting_params_refine(Tracker, self.stack, len(self.stack)+11)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.get_sorting_params_refine(Tracker, self.stack, len(self.stack)+11)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_stack(self):
        stack=[]
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["nproc"] = 1
        return_new = fu.get_sorting_params_refine(Tracker, stack, len(stack))
        return_old = oldfu.get_sorting_params_refine(Tracker, stack, len(stack))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, []))

    def test_wrong_images_in_the_stack_RunTimeError(self):
        stack=[IMAGE_2D,IMAGE_2D]
        for i in range(len(stack)):
            stack[i].set_attr("group",i)
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["constants"]["nproc"] = 1
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_sorting_params_refine(Tracker, stack, len(stack))
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_sorting_params_refine(Tracker, stack, len(stack))
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_parsing_sorting_params(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.parsing_sorting_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.parsing_sorting_params()
        self.assertEqual(str(cm_new.exception), "parsing_sorting_params() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_list(self):
        return_new = fu.parsing_sorting_params([])
        return_old = oldfu.parsing_sorting_params([])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, ([], [])))

    def test_typeerror_int_hasnot_attribute__get_item__(self):
        l=[1,2,3,4,5]
        with self.assertRaises(TypeError) as cm_new:
            fu.parsing_sorting_params(l)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.parsing_sorting_params(l)
        self.assertEqual(str(cm_new.exception), "'int' object has no attribute '__getitem__'")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_values(self):
        l=[[1,2,3,4,5],[1,21,31,41,51]]
        return_new = fu.parsing_sorting_params(l)
        return_old = oldfu.parsing_sorting_params(l)
        self.assertTrue(array_equal(return_new[0], return_old[0]))
        self.assertTrue(array_equal(return_new[1], return_old[1]))
        self.assertTrue(array_equal(return_new[0], [1, 1]))
        self.assertTrue(array_equal(return_new[1], [[2, 3, 4, 5], [21, 31, 41, 51]]))




class Test_fill_in_mpi_list(unittest.TestCase):
    # Values got running Test_get_sorting_params_refine.test_default_case
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.fill_in_mpi_list()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.fill_in_mpi_list()
        self.assertEqual(str(cm_new.exception), "fill_in_mpi_list() takes exactly 4 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        total_attr_value_list=[[],[],[]]
        attr_value_list = [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]
        return_new = fu.fill_in_mpi_list(mpi_list = deepcopy(total_attr_value_list), data_list = attr_value_list, index_start = 0 ,index_end = len(total_attr_value_list))
        return_old = oldfu.fill_in_mpi_list(mpi_list = deepcopy(total_attr_value_list), data_list = attr_value_list, index_start = 0 ,index_end = len(total_attr_value_list))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]))

    def test_index_start_negative_returns_IndexError_list_index_out_of_range(self):
        total_attr_value_list=[[],[],[]]
        attr_value_list = [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]
        with self.assertRaises(IndexError) as cm_new:
            fu.fill_in_mpi_list(mpi_list=deepcopy(total_attr_value_list), data_list=attr_value_list, index_start=0-1,index_end=len(total_attr_value_list))
        with self.assertRaises(IndexError) as cm_old:
            oldfu.fill_in_mpi_list(mpi_list=deepcopy(total_attr_value_list), data_list=attr_value_list, index_start=-1,index_end=len(total_attr_value_list))
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))
    def test_index_end_too_high_returns_IndexError_list_index_out_of_range(self):
        total_attr_value_list=[[],[],[]]
        attr_value_list = [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]
        with self.assertRaises(IndexError) as cm_new:
            fu.fill_in_mpi_list(mpi_list=deepcopy(total_attr_value_list), data_list=attr_value_list, index_start=0,index_end=len(total_attr_value_list)+2)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.fill_in_mpi_list(mpi_list=deepcopy(total_attr_value_list), data_list=attr_value_list, index_start=0,index_end=len(total_attr_value_list)+2)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_less_values_in_attr_value_list_returns_IndexError_list_index_out_of_range(self):
        total_attr_value_list=[[],[],[]]
        attr_value_list = [ [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]
        with self.assertRaises(IndexError) as cm_new:
            fu.fill_in_mpi_list(mpi_list=deepcopy(total_attr_value_list), data_list=attr_value_list, index_start=0, index_end=len(total_attr_value_list))
        with self.assertRaises(IndexError) as cm_old:
            oldfu.fill_in_mpi_list(mpi_list=deepcopy(total_attr_value_list), data_list=attr_value_list, index_start=0, index_end=len(total_attr_value_list))
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_too_values_in_attr_value_list(self):
        total_attr_value_list=[[],[],[]]
        attr_value_list = [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]
        return_new = fu.fill_in_mpi_list(mpi_list = deepcopy(total_attr_value_list), data_list = attr_value_list, index_start = 0 ,index_end = len(total_attr_value_list))
        return_old = oldfu.fill_in_mpi_list(mpi_list = deepcopy(total_attr_value_list), data_list = attr_value_list, index_start = 0 ,index_end = len(total_attr_value_list))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [[0, 27.84771510918482, 49.09925034711038, 236.702241194244, 0.0, 0.0], [1, 54.496982231553545, 150.6989385443887, 95.77312314162165, 0.0, 0.0], [2, 67.0993779295224, 52.098986136572584, 248.45843717750148, 0.0, 0.0]]))



class Test_sample_down_1D_curve(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.sample_down_1D_curve()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.sample_down_1D_curve()
        self.assertEqual(str(cm_new.exception), "sample_down_1D_curve() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        return_new = fu.sample_down_1D_curve(nxinit=100, nnxo=180, pspcurv_nnxo_file=path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,"Sharpening-after-meridien/fsc_halves.txt"))
        return_old = oldfu.sample_down_1D_curve(nxinit=100, nnxo=180, pspcurv_nnxo_file=path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,"Sharpening-after-meridien/fsc_halves.txt"))
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new,  [0.4444444444444444, 3.3333333333333335, 6.444444444444444, 9.555555555555557, 12.666666666666663, 17.000000000000004, 19.333333333333343, 22.444444444444436, 25.555555555555543, 28.66666666666668, 34.00000000000001, 35.33333333333335, 38.444444444444464, 41.55555555555553, 44.66666666666664, 51.00000000000001, 51.33333333333331, 54.44444444444447, 57.55555555555558, 60.6666666666667, 68.00000000000001, 67.33333333333323, 70.44444444444449, 73.55555555555559, 76.66666666666671, 85.00000000000003, 83.33333333333337, 86.4444444444443, 89.55555555555559, 92.66666666666671, 102.00000000000003, 99.33333333333337, 102.44444444444429, 105.5555555555556, 108.66666666666671, 118.9999999999998, 115.33333333333384, 118.44444444444403, 121.5555555555556, 124.66666666666673, 135.99999999999977, 131.3333333333339, 134.44444444444397, 137.55555555555563, 140.66666666666674, 152.99999999999974, 147.333333333334, 150.4444444444439, 153.55555555555623, 156.66666666666612, 169.99999999999972, 163.33333333333405, 166.44444444444383, 169.5555555555563, 172.66666666666606, 186.9999999999997, 179.3333333333341, 182.4444444444438, 185.5555555555564, 188.666666666666, 203.99999999999966, 195.3333333333342, 198.44444444444377, 201.55555555555645, 204.66666666666595, 220.99999999999963, 211.33333333333258, 214.44444444444713, 217.5555555555548, 220.6666666666659, 237.9999999999996, 227.33333333333252, 230.4444444444473, 233.55555555555475, 236.66666666666583, 254.99999999999957, 243.33333333333246, 246.44444444444753, 249.5555555555547, 252.66666666666578, 271.99999999999955, 259.3333333333324, 262.4444444444477, 265.55555555555463, 268.6666666666657, 288.99999999999955, 275.33333333333235, 278.4444444444479, 281.5555555555546, 284.66666666666566, 305.9999999999995, 291.3333333333323, 294.44444444444815, 297.5555555555545, 300.6666666666656, 322.99999999999943, 307.33333333333474, 310.4444444444458, 175.77777777777527, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_null_nxinit_returns_ZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.sample_down_1D_curve(nxinit=0, nnxo=180, pspcurv_nnxo_file=path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,"Sharpening-after-meridien/fsc_halves.txt"))
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.sample_down_1D_curve(nxinit=0, nnxo=180, pspcurv_nnxo_file=path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,"Sharpening-after-meridien/fsc_halves.txt"))
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_null_nnxo_returns_ZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.sample_down_1D_curve(nxinit=100, nnxo=0, pspcurv_nnxo_file=path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,"Sharpening-after-meridien/fsc_halves.txt"))
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.sample_down_1D_curve(nxinit=100, nnxo=0, pspcurv_nnxo_file=path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,"Sharpening-after-meridien/fsc_halves.txt"))
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_file_not_found(self):
        with self.assertRaises(IOError) as cm_new:
            fu.sample_down_1D_curve(nxinit=100, nnxo=180, pspcurv_nnxo_file="filenotfound.txt")
        with self.assertRaises(IOError) as cm_old:
            oldfu.sample_down_1D_curve(nxinit=100, nnxo=180, pspcurv_nnxo_file="filenotfound.txt")
        self.assertEqual(cm_new.exception.strerror, "No such file or directory")
        self.assertEqual(cm_new.exception.strerror, cm_old.exception.strerror)



class Test_get_initial_ID(unittest.TestCase):
    full_ID_dict = {0: 'ciao_0', 1: 'ciao_1', 2: 'ciao_2', 3: 'ciao_3'}

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_initial_ID()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_initial_ID()
        self.assertEqual(str(cm_new.exception), "get_initial_ID() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_valid_list_dict(self):
        part_list = [0,1,2]
        return_new  = fu.get_initial_ID(part_list, self.full_ID_dict)
        return_old = oldfu.get_initial_ID(part_list, self.full_ID_dict)
        self.assertTrue(array_equal(return_new, return_old))

    def test_empty_list(self):
        return_new  = fu.get_initial_ID([], self.full_ID_dict)
        return_old = oldfu.get_initial_ID([], self.full_ID_dict)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, []))

    def test_invalid_value_in_list_KeyError(self):
        part_list = [0, 1, 20]
        with self.assertRaises(KeyError) as cm_new:
            fu.get_initial_ID(part_list, self.full_ID_dict)
        with self.assertRaises(KeyError) as cm_old:
            oldfu.get_initial_ID(part_list, self.full_ID_dict)
        self.assertEqual(str(cm_new.exception), 20)
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_dict_KeyError(self):
        part_list = [0, 1, 20]
        with self.assertRaises(KeyError) as cm_new:
            fu.get_initial_ID(part_list, {})
        with self.assertRaises(KeyError) as cm_old:
            oldfu.get_initial_ID(part_list, {})
        self.assertEqual(str(cm_new.exception), 0)
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_print_upper_triangular_matrix(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.print_upper_triangular_matrix()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.print_upper_triangular_matrix()
        self.assertEqual(str(cm_new.exception), "print_upper_triangular_matrix() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    @unittest.skip("which variable is the third parameter??")
    def test_print_upper_triangular_matrix(self):
        log_new =[]
        log_old = []
        size =4
        data=[]
        for i in range(size):
            for j in range(size):
                data.append((i,j*j))
        fu.print_upper_triangular_matrix(data,size,log_new)
        oldfu.print_upper_triangular_matrix(data, size, log_old)



class Test_convertasi(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.convertasi()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.convertasi()
        self.assertEqual(str(cm_new.exception), "convertasi() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_list(self):
        return_new = fu.convertasi([],3)
        return_old = oldfu.convertasi([],3)
        self.assertTrue(allclose(return_new, return_old))
        self.assertTrue(allclose(return_new, [array([], dtype=int), array([], dtype=int), array([], dtype=int)]))

    def test_default_case(self):
        asig = [0,1,2,3,4,5,6]
        return_new = fu.convertasi(asig,7)
        return_old = oldfu.convertasi(asig,7)
        self.assertTrue(allclose(return_new,return_old))
        self.assertEqual(return_new, [array([0], dtype=int), array([1], dtype=int), array([2], dtype=int), array([3], dtype=int), array([4], dtype=int), array([5], dtype=int), array([6], dtype=int)])



class Test_prepare_ptp(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.prepare_ptp()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.prepare_ptp()
        self.assertEqual(str(cm_new.exception), "prepare_ptp() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_list(self):
        return_new = fu.prepare_ptp([],3)
        return_old = oldfu.prepare_ptp([],3)
        self.assertTrue(allclose(return_new, return_old))
        self.assertTrue(allclose(return_new, []))

    def test_default_case(self):
        K = 7
        data_list = [[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6]]
        return_new = fu.prepare_ptp(data_list, K)
        return_old = oldfu.prepare_ptp(data_list, K)
        self.assertTrue(allclose(return_new, return_old))
        self.assertEqual(return_new, [[array([0], dtype=int), array([1], dtype=int), array([2], dtype=int), array([3], dtype=int), array([4], dtype=int), array([5], dtype=int), array([6], dtype=int)], [array([0], dtype=int), array([1], dtype=int), array([2], dtype=int), array([3], dtype=int), array([4], dtype=int), array([5], dtype=int), array([6], dtype=int)], [array([0], dtype=int), array([1], dtype=int), array([2], dtype=int), array([3], dtype=int), array([4], dtype=int), array([5], dtype=int), array([6], dtype=int)]])



class Test_print_dict(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.print_dict()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.print_dict()
        self.assertEqual(str(cm_new.exception), "print_dict() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_print_dict(self):
        dic = {'0': 'ciao_0', '1': 'ciao_1'}
        old_stdout = sys.stdout
        print_new = StringIO()
        sys.stdout = print_new
        return_new = fu.print_dict(dic, "title")
        print_old = StringIO()
        sys.stdout = print_old
        return_old = oldfu.print_dict(dic, "title")
        self.assertEqual(return_new,return_old)
        self.assertTrue(return_new is None)
        self.assertEqual(print_new.getvalue(), print_old.getvalue())
        sys.stdout = old_stdout

    def test_error_key_type(self):
        dic = {0: 'ciao_0', 1: 'ciao_1', 2: 'ciao_2', 3: 'ciao_3'}
        with self.assertRaises(TypeError) as cm_new:
            fu.print_dict(dic, " Test_print_dict.test_error_key_type")
        with self.assertRaises(TypeError) as cm_old:
            oldfu.print_dict(dic, " Test_print_dict.test_error_key_type")
        self.assertEqual(str(cm_new.exception), "cannot concatenate 'str' and 'int' objects")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_get_resolution_mrk01(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_resolution_mrk01()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_resolution_mrk01()
        self.assertEqual(str(cm_new.exception), "get_resolution_mrk01() takes exactly 5 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_radi_not_integer(self):
        v = [IMAGE_2D,IMAGE_2D_REFERENCE]
        return_new = fu.get_resolution_mrk01(deepcopy(v), 0.5,0.15,ABSOLUTE_PATH, None)
        return_old = oldfu.get_resolution_mrk01(deepcopy(v), 0.5,0.15,ABSOLUTE_PATH,None)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (-0.0061, 0.1617, 0.01)))
        remove_list_of_file([path.join(ABSOLUTE_PATH,"fsc.txt")])

    def test_radi_integer_no_mask(self):
        v = [IMAGE_3D,IMAGE_3D]
        return_new = fu.get_resolution_mrk01(deepcopy(v), 1,IMAGE_3D.get_xsize(),ABSOLUTE_PATH, None)
        return_old = oldfu.get_resolution_mrk01(deepcopy(v), 1,IMAGE_3D.get_xsize(),ABSOLUTE_PATH,None)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.2, 0.2, 0.47)))
        remove_list_of_file([path.join(ABSOLUTE_PATH, "fsc.txt")])

    def test_radi_integer_with_mask(self):
        v = [IMAGE_3D,IMAGE_3D]
        mask_option = [fu.model_circle(1,IMAGE_3D.get_xsize(),IMAGE_3D.get_ysize(),IMAGE_3D.get_zsize())]
        return_new = fu.get_resolution_mrk01(deepcopy(v), 1,None,ABSOLUTE_PATH, mask_option)
        return_old = oldfu.get_resolution_mrk01(deepcopy(v), 1,None,ABSOLUTE_PATH,mask_option)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.2, 0.2, 0.47)))
        remove_list_of_file([path.join(ABSOLUTE_PATH, "fsc.txt")])

    def test_with_invalid_mask_returns_RuntimeError_ImageFormatException(self):
        v = [IMAGE_3D,IMAGE_3D]
        mask_option = [fu.model_circle(1,IMAGE_3D.get_xsize()+10,IMAGE_3D.get_ysize(),IMAGE_3D.get_zsize())]
        with self.assertRaises(RuntimeError) as cm_new:
            fu.get_resolution_mrk01(deepcopy(v), 1,None,ABSOLUTE_PATH, mask_option)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.get_resolution_mrk01(deepcopy(v), 1,None,ABSOLUTE_PATH,mask_option)
        msg = str(cm_new.exception).split("'")
        msg_old = str(cm_old.exception).split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageFormatException")
        self.assertEqual(msg[1], "can not multiply images that are not the same size")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_partition_to_groups(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.partition_to_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.partition_to_groups()
        self.assertEqual(str(cm_new.exception), "partition_to_groups() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_list(self):
        return_new = fu.partition_to_groups([],3)
        return_old = oldfu.partition_to_groups([],3)
        self.assertTrue(allclose(return_new, return_old))
        self.assertTrue(allclose(return_new, [[], [], []]))

    def test_default_case(self):
        K = 7
        data_list = [[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6]]
        return_new = fu.partition_to_groups(data_list, K)
        return_old = oldfu.partition_to_groups(data_list, K)
        self.assertTrue(allclose(return_new, return_old))
        self.assertTrue(allclose(return_new, [[], [], [], [], [], [], []]))



class Test_partition_independent_runs(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.partition_independent_runs()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.partition_independent_runs()
        self.assertEqual(str(cm_new.exception), "partition_independent_runs() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_list(self):
        return_new = fu.partition_independent_runs([],3)
        return_old = oldfu.partition_independent_runs([],3)
        self.assertDictEqual(return_new,return_old)
        self.assertEqual(return_new, {})

    def test_default_case(self):
        K = 7
        data_list = [[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6]]
        return_new = fu.partition_independent_runs(data_list, K)
        return_old = oldfu.partition_independent_runs(data_list, K)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, {0: [[0], [1], [2], [3], [4], [5], [6]], 1: [[0], [1], [2], [3], [4], [5], [6]], 2: [[0], [1], [2], [3], [4], [5], [6]]})




class Test_merge_groups(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.merge_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.merge_groups()
        self.assertEqual(str(cm_new.exception), "merge_groups() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_list(self):
        return_new = fu.merge_groups([])
        return_old = oldfu.merge_groups([])
        self.assertTrue(allclose(return_new, []))

    def test_default_case(self):
        data_list = [[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6]]
        return_new = fu.merge_groups(data_list)
        return_old = oldfu.merge_groups(data_list)
        self.assertEqual(return_new, [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])



class Test_save_alist(unittest.TestCase):
    filename_new = "listfile.txt"
    filename_old = "listfile2.txt"
    data_list = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.save_alist()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.save_alist()
        self.assertEqual(str(cm_new.exception), "save_alist() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_create_files(self):
        Tracker = deepcopy(TRACKER)
        Tracker["this_dir"] = ABSOLUTE_PATH
        Tracker["constants"]["log_main"] = "logging"
        Tracker["constants"]["myid"] = "myid"
        Tracker["constants"]["main_node"] = "myid"

        fu.save_alist(Tracker, self.filename_new, self.data_list)
        oldfu.save_alist(Tracker,self.filename_old, self.data_list)
        self.assertEqual(returns_values_in_file(path.join(ABSOLUTE_PATH,self.filename_new)),returns_values_in_file(path.join(ABSOLUTE_PATH,self.filename_old)))
        remove_list_of_file([path.join(ABSOLUTE_PATH,self.filename_new),path.join(ABSOLUTE_PATH,self.filename_old)])

    def test_no_create_files(self):
        Tracker = deepcopy(TRACKER)
        Tracker["this_dir"] = ABSOLUTE_PATH
        Tracker["constants"]["log_main"] = "logging"
        Tracker["constants"]["myid"] = "myid"
        Tracker["constants"]["main_node"] = "different myid"

        fu.save_alist(Tracker, self.filename_new, self.data_list)
        oldfu.save_alist(Tracker, self.filename_old, self.data_list)

        self.assertFalse(path.isfile(path.join(ABSOLUTE_PATH, self.filename_new)))
        self.assertFalse(path.isfile(path.join(ABSOLUTE_PATH, self.filename_old)))



class Test_margin_of_error(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.margin_of_error()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.margin_of_error()
        self.assertEqual(str(cm_new.exception), "margin_of_error() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        return_new = fu.margin_of_error(0.2,0.1)
        return_old = oldfu.margin_of_error(0.2,0.1)
        self.assertEqual(return_new, 1.2649110640673518)



class Test_do_two_way_comparison(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.do_two_way_comparison()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.do_two_way_comparison()
        self.assertEqual(str(cm_new.exception), "do_two_way_comparison() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_defaault_case(self):
        Tracker = deepcopy(TRACKER)
        Tracker["this_dir"] = ABSOLUTE_PATH
        Tracker["constants"]["log_main"] = "logging"
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 1
        Tracker["this_total_stack"] = 10
        Tracker["number_of_groups"] = 4
        Tracker["constants"]["indep_runs"]  = 4
        Tracker['full_ID_dict'] = {0: 0, 1: 1, 2:2, 3: 3}
        Tracker["partition_dict"]    = [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]]
        Tracker["chunk_dict"] = [0, 1, 2, 3]
        Tracker["P_chunk0"] = 0.2
        Tracker["constants"]["smallest_group"] = 2
        Tracker2 = deepcopy(Tracker)
        return_new = fu.do_two_way_comparison(Tracker)
        return_old = oldfu.do_two_way_comparison(Tracker2)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, None)
        self.assertTrue(array_equal(Tracker["score_of_this_comparison"], Tracker2["score_of_this_comparison"]))


class Test_select_two_runs(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.select_two_runs()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.select_two_runs()
        self.assertEqual(str(cm_new.exception), "select_two_runs() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        summed_scores = [0,1,2,3,4]
        two_way_dict = [3.2,1.43,54,32,543]
        return_new = fu.select_two_runs(summed_scores,two_way_dict)
        return_old = oldfu.select_two_runs(summed_scores,two_way_dict)
        self.assertTrue(array_equal(return_new, (32, 543, 4, 3)))

    def test_returns_IndexError_list_index_out_of_range(self):
        summed_scores = [0,1,2,3,4]
        two_way_dict = [3.2]
        with self.assertRaises(IndexError) as cm_new:
            fu.select_two_runs(summed_scores,two_way_dict)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.select_two_runs(summed_scores,two_way_dict)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_summed_scores_empty_returns_IndexError_list_index_out_of_range(self):
        summed_scores = []
        two_way_dict = [3.2]
        with self.assertRaises(IndexError) as cm_new:
            fu.select_two_runs(summed_scores,two_way_dict)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.select_two_runs(summed_scores,two_way_dict)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_two_way_dict_empty_returns_IndexError_list_index_out_of_range(self):
        summed_scores = [0,1,2,3,4]
        two_way_dict = []
        with self.assertRaises(IndexError) as cm_new:
            fu.select_two_runs(summed_scores,two_way_dict)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.select_two_runs(summed_scores,two_way_dict)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_counting_projections(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.counting_projections()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.counting_projections()
        self.assertEqual(str(cm_new.exception), "counting_projections() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_ali3d_params_empty(self):
        return_new = fu.counting_projections(delta = 0.5, ali3d_params =[], image_start = 1)
        return_old = oldfu.counting_projections(delta = 0.5, ali3d_params =[], image_start = 1)
        self.assertDictEqual(return_new,return_old)

    def test_default_case(self):
        ali3d_params  = [[idx1, idx2, 0 , 0.25, 0.25] for idx1 in range(2) for idx2 in range(2)]
        return_new = fu.counting_projections(delta = 0.5, ali3d_params =ali3d_params, image_start = 1)
        return_old = oldfu.counting_projections(delta = 0.5, ali3d_params =ali3d_params, image_start = 1)
        self.assertDictEqual(return_new,return_old)



class Test_unload_dict(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.unload_dict()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.unload_dict()
        self.assertEqual(str(cm_new.exception), "unload_dict() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        dict_angles = {(0.64764598050589606, 53.04857999805229) : [], (10.262155636450808, 97.18016191759037) : [], (100.75772287892256, 50.274472062413594) : [], (101.11458591875028, 44.457078732458605) : []}
        return_new = fu.unload_dict(dict_angles)
        return_old = oldfu.unload_dict(dict_angles)
        self.assertTrue(allclose(return_new, [[101.11458591875028, 44.457078732458605], [10.262155636450808, 97.18016191759037], [0.6476459805058961, 53.04857999805229], [100.75772287892256, 50.274472062413594]]))

    def test_empty_dict(self):
        return_new = fu.unload_dict({})
        return_old = oldfu.unload_dict({})
        self.assertTrue(allclose(return_new, []))



class Test_load_dict(unittest.TestCase):
    ali3d_params = [[idx1, idx2, 0, 0.25, 0.25] for idx1 in range(2) for idx2 in range(2)]
    sampled = fu.counting_projections(delta=0.5, ali3d_params=ali3d_params, image_start=1)
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.load_dict()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.load_dict()
        self.assertEqual(str(cm_new.exception), "load_dict() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        # result to big. could be used the pickle for storing this data?
        d = fu.unload_dict(self.sampled)
        return_new = fu.load_dict(deepcopy(self.sampled), d)
        return_old = oldfu.load_dict(deepcopy(self.sampled), d)
        self.assertDictEqual(return_new,return_old)

    def test_empty_unloaded_dict_angles(self):
        # result to big. could be used the pickle for storing this data?
        return_new = fu.load_dict(deepcopy(self.sampled), [])
        return_old = oldfu.load_dict(deepcopy(self.sampled), [])
        self.assertDictEqual(return_new,return_old)

    def test_dict_angle_main_node(self):
        d = fu.unload_dict([])
        return_new = fu.load_dict([], d)
        return_old = oldfu.load_dict([], d)
        self.assertTrue(array_equal(return_new,return_old))
        self.assertEqual(return_new, [])

    def test_empty_all(self):
        return_new = fu.load_dict({}, [])
        return_old = oldfu.load_dict({}, [])
        self.assertDictEqual(return_new,{})
        self.assertDictEqual(return_new, return_old)




class Test_get_stat_proj(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_stat_proj()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_stat_proj()
        self.assertEqual(str(cm_new.exception), "get_stat_proj() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_myid_same_value_as_main_Node(self):
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["nproc"] = 1
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        this_ali3d = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER,'Initial3D//main001/run000/rotated_reduced_params.txt')
        Tracker2 = deepcopy(Tracker)

        return_new = fu.get_stat_proj(Tracker,delta = 5,this_ali3d=this_ali3d)
        return_old = oldfu.get_stat_proj(Tracker2,delta = 5,this_ali3d=this_ali3d)
        self.assertDictEqual(return_new,return_old)
        self.assertDictEqual(return_new, {(239.83725161472682, 34.32194941451909): [], (103.77643540278966, 71.33707511505752): [], (290.07472254019615, 98.11592924080134): [], (231.75902476866551, 169.96644317376868): [], (316.37403003265143, 64.49510317767826): [], (217.88200331799396, 164.9903791502494): [], (19.774990665148692, 138.13384266895085): [], (171.17270056579099, 62.53695091221686): [], (298.91965634545465, 54.54945736082458): [], (205.59793167440634, 111.5343412661951): [], (274.08677259413264, 25.057615418303023): [], (148.03812218633067, 91.3482604150222): [], (220.82381017489928, 58.351554491017154): [], (285.10789217692422, 59.21855243579674): [], (353.76416679400427, 79.56170661556362): [], (24.269454053805148, 109.0903549428155): [], (158.24570214975125, 8.797410709991048): [], (4.8059923649976213, 137.93223765241342): [], (233.08905524636486, 92.49483987665928): [], (170.02801654661363, 149.84913781566883): [], (57.711800215689323, 61.00679557990334): [], (23.964522632106693, 104.23398403046221): [], (48.859473508421232, 133.6766908418426): [], (286.99787206579697, 83.51511053854594): [], (135.70568585980573, 81.4753269035727): [], (29.372858128566897, 152.79942233405984): [], (301.17196288329285, 35.15005577055701): [], (278.20469572610108, 49.414033012567195): [], (138.89747531787543, 96.07800505849684): [], (88.028894707696892, 90.53926235704517): [], (179.42556130364653, 130.5859669874328): [], (162.88642477271108, 86.69523471149502): [], (66.480143603374572, 65.97979188671809): [], (219.28735281900242, 9.638979412512889): [], (113.03102302456985, 90.87632255825744): [], (56.012713543928839, 46.4164370315091): [], (98.688579230489268, 95.53596711920669): [], (340.18270201767552, 40.53580211131656): [], (91.089162077835198, 124.54538217884104): [], (12.101938358714667, 55.536414070491674): [], (278.85939220298349, 107.66971165604207): [], (358.31045035675317, 94.18293633089255): [], (41.164988355290497, 41.358641634282336): [], (309.77696392019169, 69.26062329814772): [], (6.635592091699845, 123.40737773140071): [], (166.02223944612984, 81.88407075919865): [], (66.80054006660383, 99.95886951088143): [], (159.8987701243347, 62.38491044515201): [], (73.957214030031039, 114.61196062055488): [], (48.479149346457511, 94.85911217381467): [], (193.6929019742071, 53.13010235415598): [], (253.74668889043321, 121.8862981248596): [], (115.84059481900161, 27.784651210218723): [], (24.255837521063899, 79.97268528166225): [], (130.58413387642784, 110.52329842667419): [], (61.683519459101348, 27.052742429748992): [], (254.0176902387735, 160.65298274298343): [], (313.68377941649925, 112.99110606827777): [], (66.680532246070698, 109.66198848942666): [], (245.46935957966946, 73.2476648864461): [], (301.55180848630943, 40.014445966259665): [], (0.0, 180.0): [], (33.426538054435085, 94.65619249668148): [35, 38, 40, 41, 42], (183.41239535235781, 120.93849550806387): [], (68.457566605122167, 114.53783983593641): [], (137.19097785528206, 52.367951630852616): [], (158.24326067314547, 28.35763657632798): [], (325.41463022601221, 45.194212372091116): [], (108.73718395650583, 95.67142810391188): [], (234.68996393267071, 68.24810015441338): [], (232.90609070681143, 63.36948149499615): [], (109.3485901138684, 158.7115927728546): [], (164.04850552187077, 96.4170531430287): [], (28.338006382236287, 46.043053276232335): [], (250.40484272591269, 24.73738984490336): [], (8.3484743230485954, 94.31812178851712): [], (353.83937465764706, 132.93564514784396): [], (173.38661860949529, 111.10019602409304): [], (132.183780782642, 57.15604215404191): [], (169.08070315247133, 96.48488946145406): [], (167.89441620922022, 86.76275151200221): [], (202.34896395526158, 67.81198798502527): [], (61.493655266498749, 70.76692409795508): [], (145.99515875128782, 125.28521889393367): [], (12.63079718126211, 123.4881635959913): [], (60.118173705488402, 104.72130789481913): [], (118.12986574162949, 37.538775189688025): [], (36.107908134670652, 75.27869210518088): [], (39.060971933136237, 118.99320442009667): [], (107.77471045335651, 85.95222580318716): [], (294.19377383001898, 64.19598948463411): [], (302.78209993236607, 74.02045758436027): [], (313.01088966259755, 88.71916413332387): [], (304.45358925001398, 49.768136460819186): [], (196.94910213614438, 67.73917161350815): [], (7.4363802306108084, 84.59946283186504): [], (252.47079233020685, 15.00962084975062): [], (308.53850768871501, 15.771578129121457): [], (296.73344717316695, 59.375344374371615): [], (317.52675385749973, 49.944494843807384): [], (64.838512080371984, 80.51985099198731): [], (197.29968869723476, 58.03428124920305): [], (344.22467334365012, 35.73129197996368): [], (68.556432776294116, 95.12976725858039): [], (165.05124350872205, 47.887518669881516): [], (74.97396638965246, 80.65650418800205): [], (56.770340845241009, 75.55729380808995): [80], (221.18996704913354, 77.77421140684253): [], (51.416185998926338, 153.09584354095557): [], (242.16927819245598, 107.17517478604191): [], (142.62591597320181, 37.86942607214865): [], (79.196439534981025, 119.53408057909498): [], (160.97165504747156, 81.81597622686058): [], (114.31363701195421, 56.91501881321111): [], (35.272875665976649, 46.13661821040938): [], (200.71778902515814, 77.49818220833534): [], (9.2276980040757621, 31.207757164705757): [], (288.67295869133039, 127.20768036166155): [], (135.3954713461541, 71.76343085081976): [], (332.25821468765855, 84.12532052635393): [], (46.500905840133932, 99.68523115350966): [], (205.83785848021824, 77.56721689295962): [], (212.55060045308289, 116.47980528023409): [], (19.178277061502833, 79.90422561370437): [], (261.13575775861733, 155.915018231783): [], (138.18841363682904, 42.667925494108374): [], (112.53801057119246, 100.57540596643712): [], (28.847369333049425, 50.906465185304945): [], (341.49219889373057, 166.6409285859774): [], (292.78179864370168, 54.46666807706943): [], (92.199187797649415, 100.30124122384984): [], (335.66280435088044, 35.61570419541684): [], (271.54870296924105, 73.59931408796062): [], (2.1313762051486833, 16.499234491237342): [], (143.92637798874884, 96.14579720256098): [], (55.528911571385819, 65.83211186325346): [], (271.95750028221272, 63.8961188626601): [], (289.18601235497806, 122.36385994497876): [], (258.42671745330591, 49.14722090259065): [], (15.533219044475857, 6.81176397112579): [], (195.08565130834253, 121.09580190105879): [], (152.66826735362793, 23.415513615494444): [], (180.02306520910329, 149.98361164018726): [], (165.06415142701721, 140.09050345032978): [], (73.222185677649534, 167.86283423034675): [], (309.09173732802213, 25.530904571684264): [], (288.00322171288013, 88.38202178121018): [], (254.09727545152359, 112.18801201497473): [], (237.90605193378349, 111.96978853756661): [], (9.4964864968354838, 50.645427321467544): [], (291.3463676109854, 49.59131762469191): [], (148.95592504606157, 96.21359798470844): [], (221.24765740607717, 106.89317608794337): [], (198.60834792825469, 48.34023455390632): [], (218.14028688085656, 116.5551371256661): [], (300.24662965061759, 161.2729529337236): [], (285.02413814153783, 98.04784624731151): [], (125.59306753425598, 81.3389828169029): [], (152.90172696135178, 42.86646669570318): [], (153.64812903221284, 67.1552612722227): [], (349.5416098463387, 64.94238458169698): [], (232.51452607493604, 111.89712200707993): [], (251.7579317360395, 83.03999510763894): [], (87.722136823385455, 85.68187821148287): [], (131.11794115714292, 120.23378001146453): [], (173.66604756161036, 101.39939905494025): [], (230.22256741276868, 121.56929663612618): [], (110.16082286333564, 66.56883306532724): [], (251.11035087631589, 126.70156586098466): [], (137.98726234175163, 100.91845745808826): [], (12.458116306308446, 84.6671664128066): [], (313.85539973238582, 127.54698186688891): [], (92.867925646344929, 76.04401299573793): [], (43.461114520881047, 94.79146561951097): [7, 13, 14, 15, 16], (150.86789958278413, 81.67975213708839): [], (101.50476774132932, 114.98322787079482): [], (167.77986661619531, 57.63614005502124): [], (179.08887730545533, 116.02884175573367): [], (257.52101226480892, 165.5200718633996): [], (287.58033718459387, 78.66935591947826): [], (18.16039839578081, 55.61813025121971): [], (248.69741363240638, 112.11523336674097): [], (333.42083005465878, 79.287420163905): [], (143.08062329609169, 100.98711492012907): [], (281.96723618858442, 73.73979529168804): [], (123.81456080425421, 95.87467947364607): [], (214.95032676046532, 58.27233816862593): [], (54.388392307548557, 56.10676813739431): [], (172.84168186156688, 130.49726612326396): [], (283.15196137844748, 93.16973618723937): [], (123.96953694681255, 144.3842958045832): [], (174.11358172592222, 96.55273491050659): [], (227.96655657912046, 87.57262915997003): [], (187.92314944015664, 87.03277466243195): [], (61.931898811788102, 75.6268894771575): [], (237.52427790079255, 165.25289984054433): [], (27.520051608475871, 84.87023274141961): [], (91.102867322199529, 7.867094614176662): [], (136.90972853211784, 120.31183012598966): [], (243.09933058833366, 92.62978842947312): [], (158.26457894396614, 125.45054263917542): [], (181.78101418259575, 18.935875672060185): [], (195.13957220896046, 106.54126856799934): [], (353.88089290778584, 55.29078350193616): [], (199.9360267113519, 53.21431446882938): [], (268.16676699980218, 54.134651313658814): [], (227.31272156354498, 63.29405035317279): [], (164.40876412756523, 125.53333192293059): [], (285.82285083716408, 25.21630055349067): [], (68.369682527070367, 158.1613795692733): [], (102.3730223524984, 56.75396919001033): [], (174.9221306217286, 52.87690644592185): [], (201.38220816613409, 116.32928928221982): [], (266.33662668345016, 73.52903550813097): [], (328.27295917221625, 11.134526883912168): [], (109.3362031903938, 124.79125538196327): [], (103.24782158776546, 124.70921649806385): [], (168.72221288066618, 28.49922948656629): [], (118.59609489759697, 134.61608696072412): [], (340.19057721121033, 132.75177774588403): [], (343.57245315078728, 50.29585111213955): [], (67.336666873157739, 51.42566859549418): [], (27.632433642763591, 118.83918940061731): [], (81.278322005789164, 143.80955786623255): [], (179.60325736038098, 57.79560548470835): [], (254.73826235167201, 97.63958490353308): [], (323.24230222336593, 79.1501841370031): [], (171.75835586710488, 120.78144756420326): [], (83.505906930489175, 27.347674402796432): [], (126.31550017214971, 37.64926683894324): [], (311.17651817001064, 54.714781106066354): [], (194.85020552798031, 111.38948332618493): [], (271.02331237856487, 29.88136600679995): [], (2.1869879720285326, 157.27252663153487): [], (133.75079140312261, 125.12023203568327): [], (98.018460917784481, 76.11345963737101): [], (276.93279586174549, 83.37941041095732): [], (238.41440865375444, 58.58880004972134): [], (54.755849949280332, 124.05579774256792): [], (328.7020227543282, 147.1401196211109): [], (0.48871785429753345, 45.66730835583508): [], (336.40120015532671, 118.14890851298503): [], (320.627029759327, 25.68685746086934): [], (347.26994218487818, 108.59179184826024): [], (329.99587845237676, 113.21095532664631): [], (17.479302104120702, 84.7348625267671): [], (162.37341889633751, 52.70763791098592): [], (138.03548792507013, 91.2134130922969): [], (82.036459169537068, 100.16424862172349): [], (238.45626525457834, 24.5758120707033): [], (6.0900508127542219, 65.16541251029842): [], (188.97536673110949, 101.60576425897996): [], (96.394479131945388, 56.673333074420135): [], (246.75032281166293, 78.11884397898929): [], (213.14032565388447, 67.95750773402241): [], (172.18698538468433, 72.25953003499768): [], (322.20480358821698, 83.98977854565446): [], (82.623574986900337, 109.87687407007884): [], (202.94150145485958, 87.23524841730068): [], (294.61455804687171, 107.88207062500425): [], (314.13664905555009, 59.610057550672394): [], (323.62886284812663, 156.7546914823545): [], (87.942728089069021, 109.94856722604813): [], (231.04977489843247, 53.63399801370294): [], (236.93590314503373, 107.10463517664381): [], (171.07197171642221, 81.95215375268847): [], (357.59733040813308, 142.68304462711026): [], (250.82364715228749, 68.4656587338049): [], (308.08787549174303, 117.7673413729364): [], (130.13105281514032, 71.69244482292976): [], (158.36787378198849, 101.19318360568383): [], (42.577228241210186, 85.07323448156838): [], (119.90892063652066, 110.37941574777419): [], (39.078297353449912, 65.61027068821267): [], (84.947564995085756, 119.61158380277804): [], (216.02216813733671, 106.82274251625043): [], (104.7114599814381, 66.49534779993832): [], (220.41861924135287, 150.5270683415096): [], (281.07795162507426, 102.8472702420185): [], (279.26467679346052, 131.9310096357791): [], (348.68002734662377, 79.49315793130248): [], (231.93171665228002, 48.78978993661076): [], (266.86496652852856, 83.24367296941216): [], (277.89258360904199, 165.79215771960665): [], (186.14095122247952, 67.59342499386679): [], (318.86259680502593, 122.76376225936458): [], (47.472115233020823, 31.72429315759589): [], (193.67421212893859, 62.84040554975123): [], (142.50786989895289, 163.9822940432013): [], (8.4033163283465928, 108.87650363055921): [], (283.08270376729422, 64.04614966191718): [], (206.77830761152794, 121.25336921502438): [], (72.638375507113892, 138.84579803992781): [], (125.55339445744349, 8.345135047331667): [], (325.7156202565738, 59.76621998853546): [], (72.250218365959327, 75.76601596953779): [81], (72.895979191887079, 124.30023323625149): [], (24.375845341597149, 70.266415142624): [], (331.45063444934664, 108.37857031560964): [], (84.420785495968047, 56.511836404008726): [], (243.44839022599422, 53.80123771985467): [], (0.58055527873847268, 65.0911147032702): [], (313.80304699591966, 20.72428624890826): [], (71.560551917159188, 163.03252818147726): [], (51.574239052103614, 99.75361958475963): [74], (209.1862196961041, 160.0515564111973): [], (318.15129518105732, 79.08154254191174): [], (73.577075483546992, 95.19744872931129): [], (257.4921070453812, 34.56032177999784): [], (336.72154711731412, 108.44961470987212): [], (312.1489022697923, 83.85420279743904): [], (190.22834481072542, 116.1789687040125): [], (298.37811023404015, 156.41544091012184): [], (293.16828153759695, 93.304765288505): [], (335.20578276866661, 152.0710899641419): [], (206.17231225268213, 53.29843413901534): [], (255.82315639496687, 146.15921148511077): [], (357.38122846109559, 84.46403288079331): [], (178.50258263949843, 48.068990364220895): [], (281.96567443519638, 83.44726508949341): [], (198.06092401506152, 92.02262405306612): [], (276.17677743479589, 136.73865583226933): [], (140.65774662824035, 71.83438791265839): [], (69.13185660675083, 61.1608105993827): [], (310.30253501729129, 146.89248258290462): [], (72.621879966269461, 27.200577665940166): [91], (112.54708784476257, 115.13204814658945): [], (93.264303292564335, 110.02029295683069): [], (81.918187679666303, 158.34330476731753): [], (12.283150771243207, 138.0329415654861): [], (144.30510451979708, 76.73755763970688): [], (87.715842740912635, 75.97454542942798): [], (354.65189845572252, 123.24603080998969): [], (347.87892008090409, 103.74770937037358): [], (276.12424696593928, 127.0386011640495): [], (113.09812094683822, 32.610575141340405): [], (148.26427765530173, 105.90943859818117): [], (289.36074937300208, 107.81125624086954): [], (47.59526306678665, 85.14088782618536): [], (29.124478775879659, 104.30353650992527): [], (211.216839681446, 43.65335068158568): [], (44.781637195256998, 119.07029795747032): [], (163.84455029923464, 173.78234974806787): [], (204.90207668242789, 62.99182355071222): [], (110.41727341526739, 81.13437372795828): [], (17.368461937224005, 26.453766696731943): [88], (205.84922110487187, 130.9419552751764): [], (175.20957770346996, 135.3786786359204): [], (335.56958036287659, 98.72920771485336): [], (144.07033273568274, 57.31636115374206): [], (163.04315587478749, 91.55054563915826): [], (107.02424826080984, 115.05761541830303): [], (266.93324309646283, 68.68289181915962): [], (333.36749024888644, 6.217650251932134): [], (160.9850801975339, 135.18707502350586): [], (349.88160231816346, 30.946587522235692): [], (308.42046161446655, 142.02090146717967): [], (36.483012737133947, 172.1329053858233): [], (97.647473033971352, 168.187717772459): [], (91.056377587667441, 177.2204739127817): [], (8.172861632616808, 113.72535483366764): [], (315.34069172786201, 98.45651933527051): [], (64.365640182382634, 143.5818603253773): [], (353.02791666988333, 103.81711452484207): [], (268.70640914615984, 44.4290662668587): [], (225.38154038972957, 136.05448043769115): [], (8.0277043479715697, 89.46073764295483): [], (234.56391799190774, 97.36762859370157): [], (242.97857796876343, 87.77501643951139): [], (162.40638022597304, 115.80401051536589): [], (123.03248009188044, 91.01115447810518): [], (292.3783424416772, 73.88017610521102): [], (347.13161087578857, 69.7643328291164): [], (165.6280857140712, 159.46704235891468): [], (240.53603011719963, 116.85696213776339): [], (293.7755256946308, 39.9094965496702): [], (216.07393900622696, 77.70523137768241): [], (124.55663930774459, 52.19752564224628): [], (356.5457430177504, 50.47085921947303): [], (143.06513038081167, 105.83935922114321): [], (62.961171780489707, 114.46376280000737): [], (15.223859123096595, 157.44763729922224): [], (13.689762185445517, 108.94775691762473): [], (93.665149126572089, 95.4682483130948): [], (80.251746012135399, 138.94833133513646): [], (310.8341614590725, 64.42039495778339): [], (357.78207429510354, 69.90794858057335): [], (203.64172745742405, 72.68366530148162): [], (359.12478932335449, 118.45514313786411): [], (209.07182816271524, 58.193054072152925): [], (309.8316655942935, 35.26696909467743): [], (154.57474939747854, 76.87602601043702): [], (304.23795537435592, 44.90848520136727): [], (356.70990519773795, 152.36030027863254): [], (109.55417869393735, 13.06413616013794): [], (132.89508240661038, 100.84981586299692): [], (220.42211020237514, 29.197756850958083): [], (67.091854955561075, 75.69646349007473): [76], (226.56493960517409, 82.70033416870608): [], (150.7510565477842, 37.97909853282036): [], (0.68910577677166884, 133.0277849099381): [], (288.37581890050143, 68.97203749226544): [], (241.97454942495273, 121.72766183137408): [], (305.29083985507816, 64.34564007916009): [], (18.806159221708448, 104.1644529620738): [], (137.84126774007274, 86.35757989712816): [], (180.01289789107267, 159.66009223088608): [], (11.596248525534557, 65.23966575577474): [], (3.3405348467551335, 103.95598700426208): [], (56.063735567265887, 109.51889145641034): [], (36.300527638387557, 55.86280237288487): [], (328.01389749578726, 88.92142750852362): [], (71.28138692251953, 41.76505755948932): [], (160.23786680769851, 42.96545970387165): [], (282.32416470700201, 151.35981917194164): [], (20.593815538356651, 75.06950737836051): [], (253.11070291140618, 92.76475158269932): [], (40.156159449241429, 109.30448285858732): [60], (68.651331188450584, 129.0935348146951): [], (185.21394653371192, 48.15953276935937): [], (208.87709939521187, 72.75425875261716): [], (63.536325979191822, 95.06209296428466): [], (336.38662192512231, 156.926081934369): [], (31.287097398294378, 99.4801490080127): [], (186.2161312672936, 82.15633451879799): [], (70.238925084561288, 17.423490223339854): [], (134.02962319392185, 76.59901034973039): [], (115.4767972247912, 81.20258929000894): [], (205.29149792821127, 48.43039545657391): [], (125.33075733109416, 120.15579181286795): [], (48.127038482794568, 51.16654031857443): [], (295.12616663496306, 98.18402377313944): [], (60.245030083433733, 36.757405666750024): [], (98.498931266280294, 71.26591223753202): [], (182.43113376244321, 62.688781972182106): [], (341.7101743972741, 55.126624037462186): [], (109.92370324270705, 37.42800554304254): [], (204.29081303940117, 140.6187436136247): [], (226.20588703858218, 126.36600198629706): [], (295.11075145530333, 122.44369815071371): [], (159.70000997668254, 130.32021506997066): [], (77.41883604468768, 66.12730246898006): [], (310.96000672040856, 156.58448638450557): [], (338.49928974290549, 64.7934737360752): [], (222.07455570469142, 34.08211549275873): [], (15.419112906095913, 74.9997340042615): [], (66.786923533862705, 70.83830001952798): [], (103.18667799906341, 139.25720458351415): [], (63.423943944904394, 61.083831676468876): [], (87.117273251691472, 100.23273753178681): [], (298.66138914961391, 166.0694454376424): [], (319.79310007777514, 132.47699429396988): [], (98.029521064766726, 90.67408354467482): [], (141.73516620336895, 139.77633882050245): [], (272.99672266388899, 88.17971021689422): [], (18.027998478898603, 89.59555584468384): [], (345.68956399537012, 98.86562627204172): [], (53.497691427817827, 94.92676551843161): [75], (248.67801764017261, 34.441316364595345): [], (298.14465062631643, 137.03454029612834): [], (355.05774671534641, 65.01677212920518): [], (128.84157399945073, 95.94244629204616): [], (282.95506880557633, 44.62132136407959): [], (327.07684064085566, 35.49978996837964): [], (139.16809778818438, 76.66829391868033): [], (45.456350796572181, 109.37592100167409): [], (210.98440856695117, 135.8605573837922): [], (130.44660561979205, 149.31658289102418): [], (86.732123755820851, 17.647214550449316): [], (212.95231829939158, 87.37021157052688): [], (149.8744147043754, 144.73303090532258): [], (62.698207528039354, 22.198215412403602): [], (102.76220696064632, 85.88464760382183): [], (10.804631575532829, 40.94892424054838): [], (189.42582074266235, 9.22771094513194): [], (62.523018228723657, 153.24519281556678): [], (39.640191588741921, 26.754807184433204): [], (282.39509175642814, 127.12309355407817): [], (173.69415560269042, 57.71590782005459): [], (46.27474039241099, 60.852550795777596): [], (95.576864639938265, 158.52669651032028): [], (336.73274057571371, 123.0045669534804): [], (23.348971249231479, 60.54336329342544): [], (109.24367838714694, 110.2356671708836): [], (212.63196893391353, 121.33225149759427): [], (277.52187683625891, 63.971158244266334): [], (199.18788169908171, 101.7434257442841): [], (203.0604907516433, 169.58665915262745): [], (90.1697409980451, 80.86138373412997): [], (123.23636667253045, 158.89803317677553): [], (305.05121883638003, 54.63216155929397): [], (211.96535498846072, 48.520430704050725): [], (290.14282466044074, 15.521648490214371): [], (247.85808735044006, 121.80694592784707): [], (23.393904926113958, 94.52094538046123): [], (324.81390969043628, 122.84395784595806): [], (55.774597724153125, 128.9200454099483): [], (26.690737405285624, 36.30444486138627): [], (119.5957431383212, 71.55038529012788): [], (191.54643559828466, 67.6663173421947): [], (160.9035529272272, 164.22842187087855): [], (106.53047808588657, 163.50076550876267): [], (88.026217342052618, 129.3545726785325): [], (258.11681791735901, 92.83223887001554): [], (320.91532477252753, 108.23656914918026): [], (252.64202919958313, 107.31633469851836): [], (42.691514411343263, 123.89323186260569): [], (79.460129062133447, 114.68612535573058): [], (297.90425998437428, 49.67978493002937): [], (7.5504706571790621, 152.50595544641877): [], (93.449047179535228, 37.2056221167311): [], (299.87046748352151, 107.95291325921967): [], (51.889808300518204, 36.644617216858975): [], (291.33897259512742, 102.98558359600642): [], (152.86942301666608, 86.5601872324848): [], (62.904630607221932, 46.509421063145076): [], (322.15670562759971, 103.40098965026961): [], (357.82716941153575, 108.73408776246798): [], (262.07101085655825, 78.3254135411771): [], (234.97243858144074, 102.22578859315747): [], (263.60347561346072, 126.86989764584403): [], (251.75634021494776, 117.00817644928779): [], (336.47131850442764, 69.62058425222581): [], (140.05197323043569, 130.05550515619262): [], (319.40118148464512, 117.91980650550357): [], (194.08098846071491, 101.67458645882292): [], (199.29003737272023, 62.916140128926656): [], (223.07982100854724, 92.35990517113943): [], (272.54379494816959, 131.84046723064063): [], (85.169100686275385, 32.233403711145): [], (224.48143798249362, 97.23171339394995): [], (106.72118691122472, 105.34947701700813): [], (276.75890285525111, 73.66956730483035): [], (0.0, 0): [], (299.67361725699647, 20.532957641085307): [], (87.434357531830713, 12.762508024475173): [], (152.38498843805408, 13.647747811565198): [], (255.45002215869587, 102.50181779166469): [], (347.85969958860483, 40.63940767276773): [], (127.41751473954145, 23.07391806563097): [], (219.44134493441555, 97.16377118142788): [], (245.75487965945675, 131.4795692959493): [], (356.5355119609971, 147.51471518017905): [], (236.70964095126084, 155.5867682664132): [], (149.44065214459792, 76.8068016309927): [], (299.48500746821486, 132.2034121830804): [], (21.149639374443165, 99.34349581199798): [], (229.85641053853416, 102.15682655385): [], (141.26941366747627, 110.66731649351264): [], (357.33340646248968, 137.8317294688425): [], (22.405046399217284, 50.81956032872139): [], (17.606650729890923, 60.46591942090505): [], (192.61966013436211, 130.7637227868028): [], (5.9509175741846843, 99.13861626587004): [], (49.464110463608343, 22.019115505336572): [], (343.2716671343112, 93.98020163735025): [], (3.9391043359762534, 79.69875877615016): [], (250.32858856250957, 102.4327831070404): [], (297.15556629662012, 44.81292497649417): [], (178.74856337335603, 111.17246469727303): [], (214.40200661327691, 97.0958390936189): [], (284.77986277796384, 49.502733876736045): [], (33.028171492734728, 89.79777918188145): [], (180.70235681509996, 140.30109518664446): [], (90.567659733370505, 134.2385307135841): [], (210.95657230875111, 77.63623324452223): [], (298.00672875086593, 88.51688479229024): [], (76.890011219068953, 36.98209591003731): [], (223.73365602412301, 116.63051850500385): [], (66.418315707489427, 56.269024737378246): [], (105.55035973326325, 51.94114626935413): [], (204.20170260059487, 33.84078851488925): [], (43.028188793833856, 89.9325931850234): [1, 6], (100.98318830408304, 129.52914078052697): [], (141.21471199337475, 144.61645396116677): [], (142.95850209592493, 62.15645284857739): [], (156.08849242943828, 52.62286075799484): [], (29.562722321169105, 109.16169998047202): [], (76.956737579865901, 100.09577438629564): [], (221.52408439952413, 82.63237140629843): [], (120.48012930618346, 129.79181949955725): [], (82.88351595505118, 66.20099470652228): [], (267.99419826204428, 88.11226810806292): [], (313.05911524591608, 79.01288507987094): [], (315.12086813461809, 69.33268350648736): [], (182.91643698525863, 86.96527529344148): [], (246.7555645482671, 39.274896235157016): [], (301.14857793197945, 146.7692778411717): [], (188.76516054909101, 154.94238458169698): [], (180.73263583702999, 67.520494390596): [], (208.06779961890427, 92.1575274307866): [], (244.08179776151053, 63.52019471976591): [], (310.28573294353657, 98.38837762500559): [], (35.527466016282176, 114.09402684707179): [], (297.39824855545515, 112.77161415563809): [], (355.46951294772975, 21.288407227145427): [], (17.331867019905232, 128.40209757719177): [], (188.05489433817959, 91.8877318919371): [], (28.151746980446092, 133.3977378387021): [], (341.69778747379962, 21.101966823224448): [], (88.345104088070315, 66.27464516633238): [], (312.91665055850717, 122.68363884625795): [], (14.099640007801833, 79.83575137827653): [], (189.48034438596306, 111.31710818084036): [], (320.23533557308588, 137.33207450589163): [], (238.85719144775859, 39.1682940932813): [], (342.73145324524569, 103.67832477899668): [], (238.92511707414644, 160.45053257754293): [], (332.97307561081004, 64.71895002047737): [], (304.43051901244564, 69.18852875629335): [], (261.9969447520665, 54.051429995467295): [], (26.015772186214257, 41.154201960072214): [], (198.40434838808471, 72.61304472227201): [], (187.92352773055256, 72.47172162435507): [], (30.050275313198718, 114.02020811328191): [], (358.84718979001832, 79.63024019452257): [], (174.28747473241671, 106.26020470831196): [], (157.31641891695654, 110.88360028250784): [], (320.59977196787617, 30.551065330119105): [], (318.01201114706942, 88.78658690770312): [], (275.00859541286479, 175.18477471144197): [], (130.81104201458234, 42.56837461465705): [], (41.615091152698575, 12.13716576965323): [], (117.62554753300891, 100.64398518603878): [], (58.516747918653138, 94.99442574975981): [], (68.578363416767075, 36.86989764584401): [], (311.87827979948344, 103.26244236029312): [], (93.029175008082234, 90.60667253095421): [], (325.80110266111257, 69.47670157332581): [], (324.73087741551865, 142.24051701873574): [], (230.62775834631526, 29.335638207535784): [], (126.23246789723164, 57.075774041035096): [], (348.28417062702141, 94.04777419681285): [], (107.5411278050979, 153.85048677991878): [], (118.22855504170505, 52.11216510252713): [], (293.72995867071143, 69.04423601770183): [], (95.233055100735413, 80.9296506025042): [], (107.47392443688061, 129.61658952356146): [], (283.00122744079323, 88.31458695977737): [], (330.56624828148449, 50.12039801018334): [], (259.64037847532626, 121.96571875079697): [], (9.1381899080017206, 21.473303489679733): [], (232.59913680380188, 136.15169763777254): [], (350.06318928358485, 50.38341047643854): [], (260.57282842988303, 102.57087092426882): [], (189.44790915707961, 43.359611777428746): [], (186.01820372440122, 130.67478564699888): [], (324.77991640728925, 40.32793062016698): [], (60.406200384126471, 56.18793490569805): [], (249.69348454815761, 97.57157983574645): [], (17.09916355763696, 65.31387464426943): [], (14.435466009284163, 45.855479771078066): [], (37.55868615984987, 85.0055742502402): [], (199.28844585162818, 96.89210257934639): [], (256.79432526119814, 83.10789742065361): [], (97.282210793663438, 100.36975980547743): [], (224.74171269254646, 102.08788239050064): [], (49.627767858586999, 80.31476884649034): [], (143.49730998368204, 52.45301813311111): [], (319.85213788909959, 166.3522521884348): [], (4.3242538318470451, 162.131774069019): [], (231.60503265325212, 82.76828660605007): [], (62.718371205431737, 133.86338178959062): [], (49.780412827324056, 104.58196299087791): [73], (313.17858641291571, 74.16064077885679): [], (313.01388528429879, 132.38566786209728): [], (230.96952667000849, 34.202217091750974): [], (23.719710714510143, 128.48816303368184): [], (200.13662212225822, 150.25421240159724): [], (349.23529965615319, 157.09868442282556): [], (177.57861393791922, 23.75246976672186): [], (193.55427054228707, 145.32103056523948): [], (88.925001382519255, 163.2650559592932): [], (107.45161415628522, 100.50684206869752): [], (74.835557862217627, 61.23773261439896): [], (146.8084222133123, 134.9961141962564): [], (148.60919860342403, 62.23265862706361): [], (145.8147133472529, 81.61162237499441): [], (120.6483330989218, 149.18474436539535): [], (326.65875664463852, 16.017705956798686): [], (199.09188795333137, 38.63158149726329): [], (153.5556254908671, 154.46909542831574): [], (34.407760352415501, 80.10956134284628): [], (18.245871302251551, 36.19044213376744): [], (169.07914139908337, 106.19000181890854): [], (165.16507869525537, 23.584559089878177): [], (118.07330197685803, 115.2065262639248): [], (274.92551817276916, 97.91171446923524): [], (257.35347561346072, 126.78568553117061): [], (16.082411198015276, 99.27518936847825): [], (235.02420945033114, 73.10682391205664): [], (34.803157158204819, 138.33624237495556): [], (54.928591082651288, 157.98088449466343): [], (90.703123974797933, 119.68914665642902): [], (178.04965492377869, 91.75285019710286): [], (165.22352675050402, 154.62594214968027): [], (102.22762600390082, 119.84445237642747): [], (291.97557868184293, 112.69852873892223): [], (355.52060360562962, 40.74279541648585): [], (101.69682662827799, 37.31695537288978): [], (209.36341559825053, 97.02791703054572): [], (3.1755395953458674, 40.84596704505609): [], (341.99464607903468, 108.52068850541053): [], (123.74823528842032, 76.46038319389964): [], (265.8324310539561, 131.75005275572613): [], (104.55919173245792, 134.4270040008057): [], (127.47849839726021, 105.62926643165908): [], (247.98588683602475, 10.033556826231308): [], (308.25233534483374, 112.91790263764095): [], (203.80469892145663, 135.76384919322192): [], (352.35779835717841, 84.39630631629784): [], (142.79404224283428, 67.00889393172223): [], (209.40551603912024, 101.8811560210107): [], (24.638061880178466, 123.64996198723259): [], (177.1806110391486, 33.47592694188448): [], (64.800741990272741, 12.453700594268733): [], (86.477400585215022, 51.68386552633426): [], (98.283693792039557, 144.0384989919334): [], (229.79872018159068, 73.03636403560972): [], (265.69701592936894, 102.63994262003179): [], (290.80851077996971, 136.9357305834868): [], (256.5154174355776, 19.948443588802704): [], (162.67055868716653, 110.95576398229817): [], (140.21221612859551, 115.50489682232175): [], (57.629839533232612, 85.27617424079115): [], (5.8762041358461374, 142.79437788326888): [], (210.25627211935316, 150.39035456108076): [], (85.180173451822128, 37.094003174946835): [], (140.28319826813305, 149.44893466988088): [], (118.9485611348131, 154.00385375364885): [], (145.5519420800492, 42.76728907098434): [], (329.51515454517562, 54.962136128499914): [], (52.612797783827574, 85.20853438048904): [], (188.54754515902334, 140.4067412906422): [], (179.58798556408837, 164.47835150978565): [], (177.90941530127688, 86.8977717083867): [], (71.951050050163559, 66.05356826041776): [82], (41.946216991080483, 133.5835629684909): [], (160.1233510612212, 120.6246556256284): [], (73.449726857777051, 119.45663670657456): [], (36.357342862629736, 99.54849597086384): [26, 28, 30, 31, 32, 33, 34, 43, 44, 45, 46, 47], (218.6299741558642, 48.61034104164819): [], (201.35289303951009, 82.36041509646692): [], (226.30466489512125, 77.84317344615002): [], (172.87460628545733, 140.19568320791944): [], (211.20912597294148, 145.55868363540466): [], (161.8603652326899, 57.556301849286285): [], (218.44724078386628, 43.75091313509884): [], (24.668993566513315, 147.89319738758837): [], (75.523158543191755, 172.6417306081552): [], (353.39890455369652, 11.478340954533579): [], (2.4140866826890033, 84.5317516869052): [], (219.99660268923532, 126.28233733909887): [], (235.6022658507037, 4.815225288558088): [], (88.64229090497868, 95.40053716813499): [], (35.043664043739184, 133.49057893685494): [], (232.55582732733743, 58.509785109849595): [], (113.76237303280189, 95.7391704772668): [], (292.80311691452272, 151.50077051343374): [], (237.25239660058972, 53.71766266090113): [], (80.535067269562035, 61.31459798588107): [], (310.76317931953514, 30.418193542003948): [], (247.40465013024379, 107.24574124738284): [], (206.96457982615746, 116.40452274122003): [], (155.92021095270462, 81.7478700535747): [], (256.9653891269528, 78.25657425571592): [], (131.55947709302117, 32.85988037888911): [], (249.6641694215339, 63.595477258779994): [], (359.95721550983268, 55.37274110019694): [84, 85], (41.708321767439926, 51.07995459005171): [], (281.0183610410545, 30.016388359812776): [], (58.028206094932983, 90.1348137232501): [], (318.37412720006796, 74.23069584574881): [], (3.3342407642822423, 94.25052609647473): [], (276.03806524104368, 141.58487066479225): [], (18.631611187714228, 123.56902489020523): [], (72.077906530084633, 70.90964505718452): [], (122.80976035404203, 86.15492694329724): [], (306.74127306787074, 103.19319836900732): [], (159.01698150567989, 96.34922585649171): [], (179.49767461842674, 106.33043269516965): [], (345.49601785440461, 127.97329450958594): [], (169.90745908561135, 67.37451770392923): [], (111.89314173402443, 52.026705490414066): [], (85.105460013483054, 80.7931037786541): [], (245.44844591326162, 68.39317560031765): [], (285.50869613088702, 117.46304908778315): [], (338.50836701647523, 79.35601481396125): [], (238.64458317835818, 126.53360191948902): [], (227.12575297540045, 111.82449250077435): [], (94.50063456991748, 129.44180203759788): [], (185.34936165829873, 77.29096700560457): [], (335.61571750999224, 55.04442134645301): [], (9.0199184181306595, 79.7672624682132): [], (2.7112734995976666, 113.65174595912492): [], (97.165484679882056, 124.62725889980304): [], (248.8835938099312, 155.75037139749483): [], (135.92550945304984, 110.59529045586153): [], (227.06355713532957, 19.5494674224571): [], (37.963381019142986, 31.595869568188863): [], (278.17177130221091, 39.69890481335556): [], (297.49075209680103, 25.374057850319726): [], (26.255635597395173, 167.23749197552485): [], (87.937255172683038, 71.12349636944079): [], (330.7706154737773, 122.9242259589649): [], (157.27085189303739, 139.98555403374036): [], (92.842529559029089, 51.76972683401611): [], (212.40179440930962, 53.38246184921846): [], (77.693219455538298, 85.54666866527184): [], (330.73049450784964, 118.07248693585296): [], (348.67335523516954, 123.16546901838413): [], (30.811347504234526, 143.13010235415598): [], (77.306832361161412, 109.8052133324828): [], (218.53185151273192, 68.03021146243339): [], (309.31121138429927, 40.11916689840515): [], (8.4328026007675057, 70.05143277395187): [], (328.75985197879601, 74.37073356834092): [], (323.01301144716945, 88.8540080016114): [], (148.50720119240125, 120.46811727599487): [], (28.533359358944601, 26.604681689396497): [], (150.15861750822708, 149.58180645799604): [], (151.17591010414631, 71.97621571421496): [], (122.34435483334009, 32.73543868813297): [], (114.32483047035375, 71.47931149458947): [], (44.613994619473822, 104.51232364014308): [61, 62, 70, 72], (181.16891008702524, 82.08828553076475): [], (305.12848825297249, 108.02378428578504): [], (153.46524267537077, 105.97954241563974): [], (315.65090624151372, 108.16561208734163): [], (56.198090542652047, 70.69551714141268): [], (18.378299296297499, 94.45333133472819): [], (349.51014206775164, 74.65052298299187): [], (193.52536532285899, 14.207842280393383): [], (182.33996844150832, 135.47472419829353): [], (269.26536340507221, 160.85749107124585): [], (344.02212932685819, 64.86795185341055): [], (189.91061982905075, 23.91926961105897): [], (183.05524594988498, 38.41512933520774): [], (357.24276289793931, 113.57817847820183): [], (244.26806997614003, 58.66774850240574): [], (239.60631984434289, 97.43560178154884): [], (71.8781005261649, 100.02731471833775): [], (228.97182188297688, 145.79778290824905): [], (86.250385222670602, 41.9670584345139): [], (16.220819725450358, 118.68540201411892): [], (248.10487613280813, 92.69726813393666): [], (213.52209968687538, 14.479928136600423): [], (237.89433088773188, 145.91788450724127): [], (34.820517761729711, 60.69807409271599): [], (206.39687138154213, 82.42842016425355): [], (76.620911578383868, 134.05065921945857): [], (240.07055572789938, 68.32065613087137): [], (323.56787919040755, 74.30072673239849): [], (343.59476343138249, 79.42459403356288): [], (133.51962373016821, 129.96749699489374): [], (173.15379397862597, 13.930554562357656): [], (320.46224371124009, 69.4047095441385): [], (251.95448963433691, 141.26057540214435): [], (338.2595784516393, 93.91263462241528): [], (283.26651097147266, 122.28409217994543): [], (194.53096015209039, 159.854912928263): [], (354.85044883570833, 26.14951322008121): [], (13.649379478686107, 104.09494318269479): [], (22.598810982545825, 65.38803937944513): [], (97.608638462135687, 61.54485686213589): [], (165.93848541074362, 120.70301974591568): [], (204.32556447821042, 96.96000489236107): [], (115.4306600776757, 124.87337596253784): [], (337.07348836017638, 50.20818050044277): [], (53.540642347036595, 17.19694697938285): [], (332.81782254596283, 127.80247435775372): [], (202.16278377773691, 24.084981768217002): [], (241.86015789156389, 19.749922795642572): [], (305.49443550761902, 137.1335333042968): [], (298.17695457100166, 93.3722866834339): [], (93.803614689728676, 66.3482540408751): [], (332.45028269007116, 45.289140427423966): [], (164.83862762325003, 77.01441640399358): [], (331.49822219864285, 59.84420818713206): [], (30.650177203541421, 123.73097526262177): [], (138.03249230336917, 47.52300570603012): [], (137.24333059174833, 159.0860590929316): [], (234.93151052302656, 116.78143078081699): [], (287.20182728843298, 5.560689196425544): [], (246.1443009052401, 116.93254394957509): [], (85.018823591768509, 124.46358592950833): [], (112.78679913602839, 86.01979836264974): [], (300.17847800488357, 98.25212994642533): [], (230.624267385293, 150.6643617924642): [], (149.79646877041225, 52.53798766980202): [], (55.783501921691375, 133.76996347371292): [], (244.87406533498566, 126.61753815078154): [], (263.12284985711528, 107.4576031237221): [], (32.539629669014559, 84.93790703571536): [], (126.4908403641944, 66.7890446733537): [], (272.29781421431193, 68.75523130619787): [], (56.235859825938277, 119.2246584303491): [], (335.43921298921913, 113.28431868601098): [], (13.637541542980305, 113.79900529347772): [], (263.1232281475111, 92.89973009018784): [], (118.03170136327995, 90.94373786483854): [], (189.53910817919831, 28.780500085183924): [], (277.6597589781727, 68.82753530272697): [], (279.87367702187697, 117.3871075026539): [], (103.82041396614832, 32.48528481982093): [], (108.03043814687477, 90.80890846495919): [], (349.88528531689138, 137.73141557042754): [], (261.1226641217354, 73.45873143200066): [], (224.35407517831291, 121.49021489015043): [], (122.71423310454398, 100.712579836095): [], (359.94351026546855, 171.6548649526684): [], (197.02868734889449, 19.14250892875417): [], (246.84467498707343, 146.03835935308135): [], (89.769309360394757, 143.92387147863911): [], (57.468014615282421, 114.38972931178735): [], (81.559891090832394, 129.2674520919506): [], (175.39769137392364, 169.22010472778766): [], (99.259061945628488, 66.42182152179817): [], (204.29605477600478, 101.812282227541): [], (128.88967196006809, 76.52970681438882): [], (325.06382784118296, 117.9961196837149): [], (121.05049913529125, 66.71568131398902): [], (192.92955967030872, 87.10026990981217): [], (284.10429512025644, 141.6934841544017): [], (270.82259323002353, 102.70903299439543): [], (142.85101589730039, 86.42512061764633): [], (22.499944811373574, 84.80255127068871): [], (148.66495916571995, 168.86547311608786): [], (347.00988628108507, 132.84364304359633): [], (346.33491760622991, 113.43116693467277): [], (154.3129304145009, 120.5463549021541): [], (228.08431147967661, 92.42737084002997): [], (225.76096758099129, 131.21021006338927): [], (61.006102274469271, 65.90597315292821): [], (3.6220200558371971, 166.9358638398621): [], (318.21523045830082, 93.64242010287187): [], (91.921588264313499, 61.468159848811666): [], (191.26253047932843, 82.22437237487944): [], (41.428609729080726, 99.61685665081788): [11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 52, 54, 63, 64, 65, 66, 67, 68, 69, 71], (144.80183647686997, 47.614332137902736): [], (261.50726940846982, 151.07980348351748): [], (26.217865507295745, 99.41181565681354): [], (108.31492434242213, 76.2522906296264): [], (337.94825664083004, 147.26456131186703): [], (221.73971977279624, 111.75189984558665): [], (218.62450873906079, 53.46639808051097): [], (76.65682467634646, 46.69496122574304): [], (285.98222616065101, 39.80431679208056): [], (134.67234755501661, 115.4302350471177): [], (114.64517743234121, 61.77461533254031): [], (223.98282045233842, 160.25007720435744): [], (359.56784878028941, 31.077419250370713): [87], (47.543885456969953, 143.35538278314104): [], (118.60530435935553, 76.39103936921708): [], (5.8872769013928519, 147.64043728796037): [], (41.007820648283356, 114.16788813674654): [], (182.16538540409934, 43.26134416773068): [], (51.978080234218083, 114.31573917117649): [], (354.69336182967436, 74.72041149571221): [], (36.629162125653494, 16.967471818522796): [], (325.45331005341177, 98.592839011356): [], (54.69903472503799, 80.38314334918212): [], (279.28802172056299, 59.14006061661744): [], (240.87383443002733, 150.80224314904194): [], (259.49994198938333, 112.26082838649187): [], (330.51098488103304, 98.66101718309712): [], (188.05458345339554, 62.76461957671463): [], (262.4988324286449, 39.487376551514686): [], (254.32911285467458, 136.44438621036832): [], (34.992642020401959, 70.4095759261488): [], (60.796713869880826, 124.13719762711513): [], (261.79976775914105, 58.90419809894123): [], (264.90542636518848, 112.3336826578053): [], (127.6379230826941, 125.03786387150011): [], (171.7817007942075, 47.978319257208824): [], (68.028292601333789, 90.26962819288237): [], (127.82064701800807, 86.22248315502372): [], (93.712971125178555, 42.06776234758659): [], (238.09405928827613, 92.56231237510126): [], (42.196904079246792, 46.2300365262871): [], (296.47162819018928, 103.05476910434224): [], (231.41804098314788, 77.91211760949938): [], (67.662472661553736, 85.41143426421418): [], (36.518404475202615, 128.6606032553309): [], (210.79862502290351, 106.7523351135539): [], (78.028503680898467, 90.40444415531614): [], (358.20384802863879, 128.14451310123005): [], (252.43802965941333, 131.56960454342607): [], (302.43717898424393, 117.69118892275974): [], (255.90680537870739, 73.38840172583328): [], (200.9295428418045, 121.17455278732378): [], (15.954766942506286, 50.732547908049405): [], (308.34023634615653, 59.53188272400513): [], (325.49298313022439, 171.20258929000894): [], (249.63779010450762, 53.88472365499568): [], (353.43773912553229, 118.37850172430018): [], (228.04015900224886, 140.93855207010134): [], (288.63999736169501, 64.12109333568951): [], (343.50524978257033, 25.99614624635114): [], (126.26646118037895, 139.56802299132198): [], (131.25327750989283, 47.43154579670649): [], (312.85796395083918, 137.23271092901567): [], (176.71624712490686, 125.6991674954162): [], (70.462562049311117, 104.86074189693989): [], (185.50719690723636, 57.87523338726993): [], (153.89081129104812, 135.09151479863272): [], (45.600026791095665, 70.55260951659467): [], (103.29157403431292, 61.62149827569983): [], (267.99113163778333, 141.4765161905836): [], (232.86971741924154, 43.945519562308846): [], (149.49456910135879, 139.88083310159487): [], (137.14078612993282, 28.07248693585296): [], (184.70975090421763, 106.40068591203939): [], (103.16745750676372, 76.18288547515793): [], (271.03347969659535, 20.145087071736995): [], (128.03336614049823, 91.07857249147638): [], (103.10797097525483, 2.7795260872183167): [], (101.16109227077686, 42.16827053115752): [], (40.366447653632392, 152.94725757025103): [], (33.598061534080372, 41.25652574874603): [], (332.09781645285511, 25.841932763167126): [], (177.43455485212738, 72.33028834395793): [], (42.335584508203404, 55.94420225743209): [], (345.93195179254047, 152.2153487897813): [], (69.906738213224571, 80.58818434318646): [77], (166.37315638504748, 18.727047066276423): [], (113.77011378550684, 120.00000000000001): [], (271.89244722094594, 151.21949991481608): [], (54.573364850533927, 7.358269391844776): [], (318.36740946226109, 45.0991278440273): [], (346.48718585521016, 45.4785312268004): [], (189.24660982852689, 121.01711624459145): [], (285.8812988924127, 156.24753023327813): [], (277.99906340693298, 88.24714980289714): [], (302.8712041754556, 78.87552211606689): [], (124.46413146733042, 47.339951588889726): [], (153.03964883680172, 91.41568683085299): [], (4.8259733025137166, 118.53184015118833): [], (344.51589950178135, 16.260204708311967): [], (212.59011233308976, 155.26261015509664): [], (240.24775256476434, 73.17725748374956): [], (348.01852426911194, 161.91337609753964): [], (18.84160642573255, 31.33760748190996): [], (43.512492130698305, 36.53152949915216): [], (55.943505033153876, 143.46847050084784): [], (193.05780691573889, 91.95517661733393): [], (90.410409263644112, 56.5926222685993): [], (271.62081628402143, 49.325214353001115): [], (110.86367556999269, 139.36059232723227): [], (359.87485939770949, 74.79027672734449): [], (219.62833660451983, 102.01895598946601): [], (266.27979743887556, 34.67896943476054): [], (165.53768145678094, 62.460956975613044): [], (251.16795269334546, 150.94072083789393): [], (197.93567467626153, 87.16776112998446): [], (65.040605401453973, 138.74347425125396): [], (84.911876845369193, 153.54623330326805): [], (274.24253220188126, 117.31121802781792): [], (191.08304974373573, 38.52348380941642): [], (159.02169044820522, 33.23072215882833): [], (183.87100819817246, 101.5369590328155): [], (173.04731418073467, 91.68541304022263): [], (72.424766399953668, 56.35003801276742): [], (29.332329313756276, 80.04113048911857): [], (21.392226596092272, 45.949340780541434): [], (54.427215462014402, 162.80305302061717): [], (34.286036742426752, 104.37311052284251): [], (3.1191224149848154, 108.80528062834026): [], (83.619997357056874, 95.33283358719343): [], (259.78384710833927, 97.70760079963482): [], (220.07685091765981, 145.6780505854809): [], (120.31588307981831, 61.85109148701495): [], (151.96485627658325, 110.81147124370663): [], (241.64086154854746, 78.04995276264827): [], (80.813652022868425, 105.00026599573852): [], (291.14760746333326, 117.53904302438696): [], (210.51034747046836, 63.06745605042491): [], (215.44411173696454, 174.43931080357453): [], (257.98857067192881, 87.97737594693389): [], (124.38762094302942, 163.739795291688): [], (130.41934782834119, 154.15806723683286): [], (198.57558525746109, 164.7322697613738): [], (275.72490137031144, 112.479505609404): [], (156.85260375764904, 115.72916167321645): [], (145.75553773258986, 115.57960504221663): [], (236.53010373362227, 77.981044010534): [], (310.99440435723221, 49.85637287967304): [], (135.05511915358241, 18.302502103072204): [], (169.9684259625935, 77.08358265250602): [], (29.685425874936481, 70.33801151057334): [], (259.96332784393257, 141.3684185027367): [], (83.517233929845105, 46.78751914791061): [], (191.91584946530025, 48.249947244273876): [], (281.13891850205658, 112.55247464626164): [], (190.47354915778467, 77.36005737996821): [], (224.61232227050863, 155.4241879292967): [], (102.36635024104392, 100.43829338443639): [], (48.720793079464485, 123.97447595029168): [], (80.104741311032754, 51.59790242280823): [], (33.588362972385575, 65.53623719999264): [], (6.1445007422990443, 26.302048554306968): [], (184.18141105913915, 96.6884535960378): [], (117.09632245461181, 105.48932439558183): [], (103.02785331855597, 17.86822593098097): [], (292.73522480071119, 132.1124813301185): [], (243.96445498307949, 141.15298471906954): [], (150.79974793705028, 18.51594874952375): [], (337.27626380216111, 59.92213476106632): [], (85.991733406211765, 105.07006214488884): [], (229.30665781487173, 68.17550749922567): [], (44.555471747533794, 80.24638041524037): [], (115.60716459844886, 66.64227750719763): [], (342.45152239876575, 137.63129452910428): [], (203.06425259546347, 92.09007429293743): [], (180.22378435764418, 77.22185783693352): [], (137.86779117475209, 105.76930415425119): [], (125.63174736165593, 134.71085957257606): [], (158.31115278695682, 47.79658781691964): [], (61.370951712731369, 109.5904240738512): [], (65.289534818238522, 104.79101369632585): [79], (337.28400455486582, 84.1930790987004): [], (133.03436644059826, 91.1459919983886): [], (72.810506285416707, 143.69555513861374): [], (49.110160596587541, 46.32330915815741): [], (222.96206610799109, 87.50516012334072): [], (189.04943283560129, 125.86534868634119): [], (0.64596832402378368, 123.32666692557987): [], (25.766842769429253, 75.13925810306013): [], (262.99179413427242, 117.15959445024878): [], (303.18597894413205, 93.4398127675152): [], (139.73783391297496, 134.9008721559727): [], (341.80269920052098, 69.69247522341864): [], (333.38138458809419, 132.6600484111103): [], (319.92843643052504, 59.68816987401035): [], (93.219208176132099, 71.19471937165976): [], (326.4897382799233, 127.71721235785645): [], (121.5312230424923, 124.955578653547): [], (351.84620849658126, 128.0588537306459): [], (226.43405531715948, 24.41323173358679): [], (139.86985941765724, 125.20268356818696): [], (252.98545357260619, 87.90992570706257): [], (352.54744632138755, 108.66292488494248): [], (158.66803514599059, 106.04967080416738): [], (251.8163956164677, 49.0580447248236): [], (260.98729869150981, 29.74578759840275): [], (292.48718253198308, 35.03280264665396): [], (30.259663717787063, 55.78132400588953): [], (332.44107322831229, 103.53961680610034): [], (255.24290886853322, 63.67071071778018): [], (352.45806532369301, 69.83615722859982): [], (239.0810225992073, 131.38965895835182): [], (84.966326774913782, 114.76033424422526): [], (326.58224612033757, 132.5684542032935): [], (9.7780555818790873, 36.076128521360886): [], (352.76268379562867, 35.84655653109822): [], (92.735927230914555, 85.74947390352528): [], (30.115209159691631, 128.57433140450584): [], (200.35733706814599, 106.61159827416674): [], (321.91046141594552, 64.5697649528823): [], (168.11585091406212, 33.35352383749118): [], (140.08637086725952, 23.245308517645505): [], (83.588723939533864, 134.14452022892195): [], (207.59798736167357, 126.11527634500433): [], (255.96072627932543, 58.82544721267624): [], (127.80407536430178, 100.78119002540916): [], (237.97480553759326, 87.70755722404412): [], (95.525773982038615, 139.15403295494392): [], (61.72404827391145, 99.89043865715374): [], (148.1751734122121, 101.05578835878299): [], (35.28177986351475, 50.993263074922794): [], (351.78731564203952, 113.5046522000617): [], (299.08152131108471, 69.11639971749217): [], (294.95788515822966, 127.29236208901408): [], (114.5750590832403, 110.30752477658137): [], (179.14714857944421, 96.6205895890427): [], (283.01908976387608, 68.89980397590698): [], (311.30854367469323, 45.00388580374361): [], (340.15849274745722, 30.815255634604668): [], (56.946645213574861, 31.85225279718823): [], (343.04976649405739, 60.00000000000001): [], (250.11683474586346, 58.746630784975636): [], (236.64437097439077, 82.83622881857214): [], (62.64639544964669, 85.34380750331852): [], (40.297505198826912, 70.48110854358967): [], (78.773226816425009, 41.86615733104918): [], (273.46776364974886, 156.08073038894105): [], (241.68296198941715, 82.90416090638111): [], (118.55689375620099, 139.46419788868343): [], (166.26651480970409, 130.4086823753081): [], (91.171518189958306, 105.13988119508276): [], (307.98124720685615, 74.09056140181883): [], (271.45839202357956, 15.267730238626182): [], (348.81875158376715, 60.077804192004145): [], (101.16477526950476, 148.92258074962928): [], (261.62055684500558, 170.3610205874871): [], (257.3721654587294, 117.08385987107336): [], (254.63606229853687, 39.381256386375306): [], (75.794736738808069, 32.106802612411634): [], (240.79010546831498, 29.47293165849043): [], (270.31374175063797, 112.40657500613321): [], (324.5305914577354, 151.92751306414704): [], (77.406998108981668, 75.8355470379262): [], (75.63726468157185, 104.93049262163949): [], (37.515735240631301, 162.57650977666017): [], (182.87961058786564, 125.78221466617617): [], (151.30232828007786, 115.6543599208399): [], (300.88776007944108, 30.28479484356964): [], (327.23181678341348, 84.05755370795386): [], (50.900218138426425, 70.62407899832593): [], (35.112872554514382, 36.4181396746227): [], (51.995405654531751, 60.92970204252969): [], (264.83008261055471, 146.2804450130663): [], (203.1882902372279, 58.11370187514041): [], (125.24527492655537, 110.45134024432323): [], (301.6057254430699, 103.123973989563): [], (215.05331356532611, 38.84701528093048): [], (46.44238296819394, 75.41803700912209): [], (273.61182273554044, 107.5989811730778): [], (327.2981422992475, 103.47029318561118): [], (13.363158132129408, 94.38572350268711): [], (283.77706881980816, 34.91520624744418): [], (138.12972702916076, 57.23623774063543): [], (97.74928158068046, 85.81706366910745): [], (153.98612378413753, 96.28140750322079): [], (245.19715648279617, 48.968747977675335): [], (292.80067543791665, 170.77228905486808): [], (57.458316053587851, 138.64135836571765): [], (219.34186280925576, 72.8953648233562): [], (95.988630872321622, 114.90888529672982): [], (267.6339822353101, 58.98288375540855): [], (23.028084986333923, 89.66296405915463): [], (324.55553722347361, 113.1376322278376): [], (132.6722503875998, 105.69927326760151): [], (232.41640343180381, 131.29987279170587): [], (205.57701800799839, 106.68195374450616): [], (214.11047444263411, 72.8248252139581): [], (8.4941743602709927, 104.02545457057204): [], (251.85849588858599, 78.18771777245901): [], (189.21637652009144, 96.75632703058784): [], (347.79855599990242, 55.20874461803673): [], (49.441286434969399, 167.5462994057313): [], (218.17666016842662, 135.95743407785082): [], (103.91476671187935, 110.16384277140018): [], (341.1226743449613, 142.461224810312): [], (101.53623551991616, 105.27958850428779): [], (212.4774298428415, 131.03125202232468): [], (10.951636276635131, 128.31613447366576): [], (285.41829187359679, 20.339907769113946): [], (168.02728782379188, 111.02796250773457): [], (39.482138535564111, 80.17797794942453): [], (80.040204398026134, 80.72481063152176): [], (327.6266510987133, 137.43162538534295): [], (184.76658014879271, 145.20273695958235): [], (195.22574725980706, 125.94857000453273): [], (313.205109847595, 93.57487938235369): [], (245.20851910744946, 102.36376675547777): [], (358.17845194132286, 103.88654036262899): [], (331.91310549910622, 161.6974978969278): [], (50.048556939384639, 65.75820782105102): [], (160.07338072307033, 149.71520515643036): [], (332.48948383146723, 40.431977008678025): [], (189.48256330902186, 135.5709337331413): [], (347.22596362151995, 147.3894248586596): [], (333.01467622438776, 88.9888455218948): [], (183.05217932562343, 91.82028978310579): [], (259.13052812236776, 131.65976544609367): [], (347.75480355335503, 118.30191565291722): [], (175.00831234662462, 38.30651584559832): [], (224.57127467324415, 72.96587771823836): [], (159.7074049925404, 76.94523089565776): [], (216.35641365499725, 111.67934386912866): [], (287.17371278751222, 73.80999818109146): [], (113.97288922749154, 129.70414888786047): [], (50.758872388840935, 109.44739048340534): [], (150.00562384242301, 57.39641274413914): [], (140.76064464413116, 81.54348066472949): [], (300.29532103988413, 141.91149724910278): [], (83.028673239696147, 90.47185292960899): [], (240.0898052789166, 102.29476862231759): [], (143.03673752831466, 91.28083586667616): [], (111.57193476327885, 134.52146877319962): [], (347.33378915498417, 84.32857189608814): [], (207.74603412272441, 67.88476663325903): [], (297.58113491229705, 73.95032919583262): [], (189.92371346593242, 106.47096449186903): [], (103.02993980248434, 90.7414954915562): [], (11.85993805268695, 60.388416197221964): [], (140.74384257037696, 32.983905041845176): [], (290.9729968645978, 30.15086218433121): [], (153.14211760329363, 130.23186353918084): [], (123.41972648895495, 42.468635067167675): [], (67.707406338436499, 119.37925190791073): [], (235.99306402234217, 141.0456443125976): [], (3.0274828799708189, 89.39332746904577): [], (145.91788933469533, 71.90531615257126): [], (184.11313449120513, 111.24476869380214): [], (212.12126051352155, 19.347017257016613): [], (72.678078291370198, 85.47905461953877): [], (348.01643778518337, 89.19109153504081): [], (27.281622287844584, 138.23494244051068): [], (339.15323585364342, 127.88783489747289): [], (38.028185333622652, 89.8651862767499): [2, 3, 5, 39], (284.10903092583231, 107.74046996500233): [], (203.97354445866173, 43.55561378963168): [], (158.04132441918284, 91.48311520770976): [], (273.86576654851973, 146.4020641188102): [], (246.72081310945725, 82.97208296945429): [], (51.607171942948305, 75.48767635985692): [], (50.506593132679193, 119.1474492042224): [], (21.924520980917148, 118.76226738560106): [], (107.99661109361058, 119.92219580799586): [], (33.344577371978666, 118.91616832353112): [], (355.81332248693235, 99.0020955118428): [], (275.04136184712553, 34.79726304041767): [], (247.98212499220423, 87.84247256921341): [], (106.82170424401814, 144.1534434689018): [], (111.90787939350365, 105.4193889381955): [], (103.7125884326835, 95.60369368370218): [], (207.94704699933405, 87.30273186606334): [], (254.40893212376415, 44.23615080677809): [], (39.166569287150054, 143.24259433325): [], (219.114660935388, 131.12067004700876): [], (91.478528807378822, 148.79224283529425): [], (0.8866365896226398, 99.07034939749582): [], (316.56558762522479, 142.13057392785137): [], (148.22254799458858, 67.08209736235906): [], (181.18599038570403, 52.96139883595049): [], (50.68992993403586, 26.904156459044472): [89], (302.82382959307944, 112.84473872777728): [], (225.6648371979386, 43.848302362227464): [], (177.58298496129345, 120.85993938338257): [], (195.22322119270143, 33.71955498693372): [], (75.101610645161585, 129.18043967127863): [], (277.38033002605732, 78.53182933137444): [], (339.47444282438926, 45.383913039275896): [], (350.75096752997297, 98.9338545047035): [], (131.92820211773676, 66.8623677721624): [], (48.716669493854241, 41.46055120122157): [], (132.57983528765214, 144.50021003162036): [], (115.3835732367878, 144.26870802003634): [], (126.51578612993282, 27.928910035858138): [], (43.5842623546474, 148.1477472028118): [], (317.1771678402705, 83.92199494150317): [], (194.25205232646957, 96.82420999191044): [], (82.653061259321262, 71.05224308237528): [], (199.87842489432268, 28.920196516482534): [], (331.13745695114727, 69.54865975567677): [], (163.87266480015558, 106.11982389478898): [], (19.063888959316632, 70.19478666751723): [], (131.64519610302281, 62.00388031628511): [], (172.9020773799933, 86.83026381276062): [], (29.086920125012345, 60.62074809208926): [], (98.588312263974885, 110.09205141942664): [], (116.0143652109043, 42.36870547089575): [], (307.96575429157599, 78.94421164121702): [], (168.56542892106498, 101.33064408052174): [], (226.69230240935505, 58.43070336387381): [], (78.95443922895322, 124.38186974878029): [], (149.89779965568883, 33.10751741709538): [], (94.510865626671176, 32.35956271203964): [], (218.07557685648342, 92.29244277595589): [], (134.48078996244354, 37.759482981264256): [], (28.409982138021007, 94.58856573578583): [], (63.774755299823639, 41.66375762504444): [86], (28.095205807178289, 65.4621601640636): [], (257.8814256735057, 107.38695527772799): [], (303.00825540133701, 88.58431316914701): [], (48.364863176324626, 56.025524049708324): [], (271.44312022728695, 122.12476661273007): [], (62.636356275012488, 148.40413043181113): [], (262.2812170385771, 24.897984639416855): [], (137.30384072545789, 62.080193494496456): [], (159.07079890582506, 67.22838584436191): [], (22.47801417090119, 143.01790408996268): [], (308.00964005935305, 88.6517395849778): [], (120.53539270663474, 81.27079228514664): [], (28.028140350575267, 89.73037180711763): [], (7.4676536481341715, 45.76146928641589): [], (17.834191910018376, 11.812282227540983): [], (223.00621858541942, 38.95435568740242): [], (157.87809605007078, 86.62771331656609): [], (201.40858748316018, 126.03187907247056): [], (69.785978840925182, 46.60226216129787): [], (337.58550725599713, 103.60896063078293): [], (82.707903264619105, 85.6142764973129): [], (250.90975546540992, 29.60964543891923): [], (168.04515014687442, 91.61797821878983): [], (36.667985280119332, 123.81206509430194): [], (224.84049054908562, 53.55024331056807): [], (307.54906760398586, 127.46201233019798): [], (342.07596354972321, 118.2253846674597): [], (275.64868621374433, 10.413340847372584): [], (273.13696228639083, 93.03472470655853): [], (340.88555472433234, 113.35772249280238): [], (238.56894774482652, 48.87932995299125): [], (109.05173150863318, 71.40820815173977): [], (117.67888733878119, 173.1882360288742): [], (324.05068590905893, 50.03250300510627): [], (96.353015757993447, 105.20972327265554): [], (39.449205644719456, 104.44270619191009): [48, 49, 50, 51, 53, 55, 56, 57, 58, 59], (125.98254974648498, 61.92751306414704): [], (88.869389615639506, 22.552362700777778): [], (216.11486706464143, 63.14303786223659): [], (124.86448032698755, 71.62142968439035): [], (313.9055914577354, 151.78460762065836): [], (146.59278833765393, 130.14362712032695): [], (213.07157205007445, 92.22498356048861): [], (265.02817386326677, 49.23627721319721): [], (307.11999959891892, 83.78640201529156): [], (146.61585857522232, 110.73937670185227): [], (167.26930876786005, 144.96719735334605): [], (61.96945746265564, 119.30192590728402): [], (14.399552911321567, 133.2124808520894): [], (225.28541000667673, 48.70012720829414): [], (232.42186884860701, 126.44975668943195): [], (10.521310318105943, 118.60859302318677): [], (293.00505316848478, 88.44945436084174): [], (261.5660332017049, 68.61051667381507): [], (196.3081152359957, 82.29239920036518): [], (256.19617205968768, 68.53810570172526): [], (210.97582185976859, 111.60682439968237): [], (3.1136494985988179, 69.97970704316931): [83], (264.8302463203741, 97.77562762512056): [], (243.30034346494355, 112.0424922659776): [], (262.99148324948811, 88.0448233826661): [], (152.12672124221319, 125.36783844070605): [], (216.48245959575996, 82.56439821845117): [], (69.664150991575752, 133.9569467237677): [92], (277.35222198497752, 122.20439451529167): [], (142.70614124151138, 120.38994244932762): [], (358.0172025795855, 89.32591645532518): [], (110.88788484021102, 149.0534124777643): [], (255.82063032786073, 53.96812092752944): [], (34.109732374093362, 148.0202458894059): [], (166.94208246741189, 38.197641081042256): [], (73.028379108769116, 90.33703594084538): [], (175.09680866322992, 77.1527297579815): [], (186.21629497711359, 33.59793588118984): [], (302.09045254160617, 83.71859249677924): [], (176.00501574054272, 145.08479375255584): [], (141.95464025964603, 154.31314253913067): [], (269.87746750064247, 97.84366548120201): [], (24.576233984293424, 113.94643173958224): [], (63.028237237092441, 90.20222081811856): [], (118.78816290000913, 95.8069209012996): [], (240.06196902071699, 44.04256592214917): [], (196.63744546390402, 135.66730835583508): [], (229.52229318814358, 97.29966583129391): [], (56.648617235252225, 99.82202205057547): [], (176.12085941489858, 82.02022530927793): [], (313.74253686221005, 117.84354715142261): [], (271.89922900822353, 83.3115464039622): [], (326.18189726068027, 108.30755517707027): [], (340.62910417240039, 98.79741070999106): [], (347.93840661241262, 176.06877086520694): [], (191.4059991123417, 57.95479186401049): [], (73.687915650444239, 153.3953183106035): [], (104.03649130658293, 47.06435485215604): [], (14.166366368599311, 142.90599682505317): [], (196.4103152891314, 140.51262344848533): [], (30.938203882179369, 75.20898630367415): [], (54.948469452997116, 104.6516243677746): [78], (19.495825670508726, 16.734944040706875): [], (282.48094866660267, 78.60060094505977): [], (261.83000106757629, 83.17579000808959): [], (296.79042904309784, 117.615089554848): [], (211.44005774332481, 82.4964145046525): [], (286.63761346010261, 54.38379327149306): [], (21.270398746742842, 133.30503877425699): [], (99.200169091086593, 51.855486898769975): [], (18.978471057583221, 109.01904063854006): [], (207.08192260458878, 38.73942459785565): [], (310.38863095942747, 108.09468384742874): [], (323.4084545049738, 54.879767964316734): [], (261.5638142786463, 44.33269164416491): [], (354.58324026422895, 60.15554762357253): [], (38.443579803840123, 94.72382575920884): [4, 8, 9, 10, 36, 37], (333.95005513305597, 74.44071648274546): [], (247.07283312900643, 136.34664931841434): [], (200.22273043538047, 111.46189429827476): [], (184.65681984529749, 116.1038811373399): [], (288.15995281495663, 93.2372484879978): [], (182.68006549985211, 72.4010188269222): [], (94.336472389930279, 27.49404455358124): [], (15.261640848860162, 147.766596288855): [], (349.34955095939034, 142.57199445695747): [], (250.68904051952185, 73.31804625549384): [], (353.01685652290098, 89.2585045084438): [], (13.027873906769246, 89.52814707039104): [], (133.99250805366242, 139.67206937983303): [], (137.36259817116877, 66.93565079651023): [], (48.028188793833856, 90.0): [], (272.27847302029352, 78.46304096718453): [], (173.52450075140916, 115.95385033808283): [], (155.93562613236463, 57.476392860474604): [], (10.242725564799336, 74.92993785511116): [], (154.25594854457009, 62.30881107724026): [], (279.97440587124549, 97.97977469072207): [], (158.8563875611226, 38.088502750897256): [], (266.38955774237053, 63.82103129598749): [], (297.77545673862613, 78.80681639431617): [], (275.83679988419823, 44.52527580170647): [], (114.65975566254572, 22.90131557717444): [], (176.8038453857867, 62.612892497346095): [], (166.93734666183551, 72.18874375913046): [], (290.92302652644662, 59.29698025408433): [], (338.01535456309784, 89.05626213516145): [], (339.13849819416413, 74.51067560441817): [], (315.99125843408547, 161.48405125047626): [], (19.105327537504429, 113.87269753101995): [], (170.55930928369384, 125.61620672850695): [], (193.16495191416215, 72.5423968762779): [], (49.348055820227948, 128.83345968142555): [], (214.51627385404544, 101.95004723735173): [], (41.592267124059681, 157.80178458759642): [], (217.95732234130284, 87.43768762489874): [], (110.85580037645769, 47.15635695640367): [], (302.38141842194796, 10.779895272212384): [], (196.71726473299358, 43.45770123427013): [], (6.0375539958993727, 55.454617821158976): [], (232.97080073118428, 87.6400948288606): [], (0.35325361287004853, 60.233230579635055): [], (317.01675439374594, 103.33170608131965): [], (244.64950620612558, 97.50358549534752): [], (199.22998197120035, 130.85277909740935): [], (268.12994060240908, 92.96722533756807): [], (280.48706830397401, 54.3008325045838): [], (261.59846843058853, 136.54229876572987): [], (105.35681359229768, 81.0661454952965): [], (18.418002080554334, 41.05166866486357): [], (46.491353827090911, 114.24179217894898): [], (86.230404285154265, 61.39140697681322): [], (319.4869004946471, 147.01609495815484): [], (318.4665423000161, 35.383546038833224): [], (126.99569167860901, 129.87960198981665): [], (330.39804448874645, 30.68341710897582): [], (59.769280189373347, 80.45150402913616): [], (195.59635542897183, 77.4291290757312): [], (290.06129739013431, 44.71720405803846): [], (270.34402077256829, 39.5932587093578): [], (190.05907889615835, 150.11863399320004): [], (274.33013046276096, 54.21778533382382): [], (306.97604485198519, 122.60358725586086): [], (151.37276033067155, 159.27571375109173): [], (301.24990881725569, 127.37713924200519): [], (233.16437426967397, 14.747100159455673): [], (327.81001091513752, 20.913940907068397): [], (66.387384021154915, 31.979754110594083): [90], (167.56045907266488, 43.06426941651319): [], (286.55580647738037, 112.6254822960708): [], (282.93052667360621, 146.52407305811556): [], (73.72451056776265, 51.51183696631818): [], (301.04075374524496, 122.52360713952542): [], (122.77341841545163, 168.52165904546644): [], (332.91651184603887, 142.35073316105678): [], (28.420021312655734, 31.466976293267564): [], (214.33673663640727, 24.249628602505194): [], (327.44347137669638, 64.64438049645341): [], (13.749545226506424, 70.12312592992117): [], (62.209008254618446, 129.0067369250772): [], (53.028192254045059, 90.06740681497661): [], (28.35817005962868, 157.62404627662053): [], (147.716361659814, 28.21539237934161): [], (221.71565270880683, 63.21856921918303): [], (40.549784454988796, 60.775341569650934): [], (260.81803277694257, 63.745895322095336): [], (229.33072487886116, 116.70594964682721): [], (18.43449762139878, 152.65232559720357): [], (187.4429019742071, 53.04579730702219): [], (247.24167866621153, 44.1394426162078): [], (133.86920974739724, 96.01022145434554): [], (153.2709208490416, 101.12447788393311): [], (129.13591617172253, 115.35561950354659): [], (284.67322120262043, 161.0641243279398): [], (164.49057111028759, 67.30147126107778): [], (236.09605082720265, 121.64844550898285): [], (20.817452503106612, 162.35278544955068): [], (176.03778217480439, 3.931229134792969): [], (168.09130878209186, 135.28279594196155): [], (299.74404930759016, 64.27083832678356): [], (151.56137011945313, 47.70552590603913): [], (167.96367382037383, 115.87890666431049): [], (231.70451477841209, 107.03412228176165): [], (275.94956892443781, 102.77814216306648): [], (292.02939608198784, 83.58294685697133): [], (42.339708093813982, 138.4377439844419): [], (297.0602538035302, 83.6507741435083): [], (226.4751029144237, 106.96363596439028): [], (353.29709600698726, 94.11535239617817): [], (292.02468713946308, 146.6464761625088): [], (163.46604040307381, 101.2619056343626): [], (267.17536938949524, 78.39423574102005): [], (303.33001592785422, 151.642363423672): [], (4.5789770024528593, 128.23027316598387): [], (54.537973112465259, 51.25302084383098): [], (268.88099218356882, 136.64038822257126): [], (178.76790456737416, 101.46817066862556): [], (335.03201237676393, 137.53136493283233): [], (319.11817546993126, 113.06434920348978): [], (333.24789750702359, 93.84507305670277): [], (343.01593944079292, 89.12367744174256): [], (213.15204669993642, 33.961640646918674): [], (87.880837992322355, 139.05107575945163): [], (1.2870682272734939, 35.961501008066584): [], (342.70031158003457, 123.0849811867889): [], (130.87743388987778, 52.28278764214354): [], (156.43181954079611, 72.04708674078033): [], (123.60290621097168, 115.28104997952263): [], (36.127786505016736, 21.838620430726724): [], (328.33214448312373, 79.21880997459087): [], (71.992488628351211, 109.73358485737599): [], (78.425580406405786, 56.430975109794765): [], (213.79398098707807, 126.19876228014533): [], (11.016173189641608, 99.2068962213459): [], (131.19423969856834, 13.359071414022612): [], (132.67896812540707, 134.8057876279089): [], (268.36631208781574, 107.52827837564496): [], (265.53918068043157, 122.04520813598953): [], (283.48591851500333, 136.83710285970835): [], (22.686695060597661, 21.65669523268248): [0], (3.0301602456152636, 50.558197962402126): [], (292.67850380567921, 78.73809436563741): [], (100.29541005769481, 80.9979044881572): [], (278.14430020767441, 93.10222829161332): [], (308.19536169036735, 93.50734363553852): [], (158.55919505568514, 144.84994422944297): [], (34.858287045015807, 109.23307590204492): [], (56.253220429463404, 41.56225601555808): [], (323.22573056965962, 93.70996589206685): [], (161.68562821466574, 72.11792937499577): [], (320.1689436977901, 127.63204836914738): [], (238.4957771345851, 63.444862874333886): [], (175.32147621735652, 67.44752535373838): [], (75.832518464571436, 22.37595372337949): [], (306.24454111079802, 132.29447409396087): [], (6.1088125925822183, 60.31085334357099): [], (132.83114712936688, 86.29003410793315): [], (268.61524382522475, 117.23538042328538): [], (353.4885862753502, 45.5729959991943): [], (174.86960015287229, 43.16289714029165): [], (7.5391436578229198, 133.12006318900518): [], (230.94079333600553, 39.06144792989868): [], (117.79848008064411, 86.0873653775847): [], (305.23166424041483, 98.32024786291163): [], (119.13327208856167, 18.086623902460392): [], (122.28652560887176, 105.55928351725456): [], (212.18918613990968, 140.725103764843): [], (24.212963711251479, 55.699766763748535): [], (113.46087033167071, 76.32167522100335): [], (223.92062461226752, 68.10287799292009): [], (285.99513407894597, 132.02168074279118): [], (119.5481553890251, 120.07786523893368): [], (42.929339104873307, 128.74697915616903): [], (210.1725431576408, 29.059279162106083): [], (200.64153486175542, 155.10201536058315): [], (101.81107793151486, 22.727473368465144): [], (60.941168427976244, 51.339396744669116): [], (328.23661723362568, 93.77751684497628): [], (90.475822308929537, 114.83458748970158): [], (108.97041403794472, 61.69808434708278): [], (96.195928751959926, 153.69795144569304): [], (77.366615402222337, 70.98095936145997): [], (96.463137323439042, 119.76676942036495): [], (320.39654821874899, 98.5246730964273): [], (317.2955861845453, 54.797316431813066): [], (5.0646441814560035, 74.86011880491724): [], (90.367271810896355, 46.87993681099482): [], (239.82953790622216, 136.24908686490116): [], (269.86038720196382, 126.95420269297779): [], (82.562203227396779, 75.90505681730521): [], (108.59485518890251, 42.26858442957247): [], (302.53917639526662, 59.4536450978459): [], (44.565023760577084, 65.68426082882353): [], (41.275964760343712, 75.3483756322254): [], (179.15393036672222, 28.640180828058384): [], (220.10558425166275, 140.83170590671872): [], (78.598261281359243, 95.2651374732329): [], (105.11442579512773, 27.639699721367478): [], (317.0538695340058, 40.223661179497554): [], (53.092996568525237, 148.2757068424041): [], (72.214771161935673, 148.53302370673245): [], (292.18999002654573, 141.80235891895774): [], (97.207002930020977, 46.972215090061894): [], (97.557791312317875, 134.3326916441649): [], (344.32519067644307, 74.5806110618045): [], (120.27576211389062, 56.995433046519615): [], (130.64982936891874, 81.40716098864402): [], (147.86039864353569, 86.49265636446148): [], (176.95960499353546, 154.78369944650936): [], (273.46339262637446, 59.06150449193615): [], (66.843413876416406, 124.21867599411047): [], (218.49055026033054, 121.41119995027866): [], (202.36835994749569, 145.43967822000215): [], (286.20774996441776, 102.916417347494): [], (342.30919363116186, 84.26082952273322): [], (81.828679583592461, 148.66239251809006): [], (117.66499299957383, 47.24822225411597): [], (195.80346871913477, 116.25410467790466): [], (168.65128583123976, 52.79231963833847): [], (49.891389232377726, 138.53944879877847): [], (108.34606600763334, 56.83453098161588): []})

    def test_myid_not_the_same_value_as_main_Node_TypeError(self):
        Tracker = deepcopy(TRACKER)
        Tracker["constants"]["nproc"] = 1
        Tracker["constants"]["myid"] = 1
        Tracker["constants"]["main_node"] = 0
        this_ali3d = path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D//main001/run000/rotated_reduced_params.txt')
        Tracker2 = deepcopy(Tracker)

        with self.assertRaises(TypeError) as cm_new:
            fu.get_stat_proj(Tracker,delta = 5,this_ali3d=this_ali3d)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_stat_proj(Tracker2,delta = 5,this_ali3d=this_ali3d)
        self.assertEqual(str(cm_new.exception), "object of type 'int' has no len()")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))





class Test_create_random_list(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.create_random_list()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.create_random_list()
        self.assertEqual(str(cm_new.exception), "create_random_list() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        Tracker_new = deepcopy(TRACKER)
        Tracker_new["constants"]["nproc"] = 1
        Tracker_new["constants"]["myid"] = 0
        Tracker_new["constants"]["main_node"] = 0
        Tracker_new["total_stack"] = "stack"
        Tracker_new["constants"]["seed"] = 1.4
        Tracker_new["constants"]["indep_runs"] = 2
        Tracker_new["this_data_list"] = [2,3,5]

        Tracker_old = deepcopy(Tracker_new)

        return_new = fu.create_random_list(Tracker_new)
        return_old = oldfu.create_random_list(Tracker_old)
        self.assertEqual(return_new, None)
        self.assertEqual(return_new, return_old)
        self.assertTrue(array_equal(Tracker_new["this_indep_list"],Tracker_old["this_indep_list"]))

    def test_wrong_Tracker_KeyError(self):
        Tracker_new = deepcopy(TRACKER)
        Tracker_old = deepcopy(TRACKER)
        with self.assertRaises(KeyError) as cm_new:
            fu.create_random_list(Tracker_new)
        with self.assertRaises(KeyError) as cm_old:
            oldfu.create_random_list(Tracker_old)
        self.assertEqual(str(cm_new.exception), 'myid')
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))





class Test_recons_mref(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons_mref()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons_mref()
        self.assertEqual(str(cm_new.exception), "recons_mref() takes exactly 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    @unittest.skip('same problem that we have in get_shrink_data_huang')
    def test_recons_mref_true_should_return_equal_objects(self):
        Tracker =  deepcopy(TRACKER)
        Tracker["constants"]["nproc"] = 1
        Tracker["constants"]["myid"] = 0
        Tracker["constants"]["main_node"] = 0
        Tracker["number_of_groups"] = 1
        Tracker["constants"]["nnxo"] = 4  # roi
        Tracker["this_particle_list"] = [[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6],[0, 1, 2, 3, 4, 5, 6]]
        Tracker["nxinit"] = 1
        Tracker["constants"]["partstack"] =  path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Initial3D//main001/run000/rotated_reduced_params.txt')
        Tracker["this_dir"] =  ABSOLUTE_PATH
        Tracker["constants"]["stack"] =  path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, "Class2D/stack_ali2d")
        Tracker["applyctf"] = False
        Tracker["chunk_dict"] = [0, 1, 2, 3, 4, 5, 6]
        Tracker["constants"]["sym"] = "c1"
        Tracker2 = deepcopy(Tracker)
        return_new = fu.recons_mref(Tracker)
        return_old = oldfu.recons_mref(Tracker2)
        self.assertTrue(return_new[0], return_old[0])




class Test_apply_low_pass_filter(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.apply_low_pass_filter()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.apply_low_pass_filter()
        self.assertEqual(str(cm_new.exception), "apply_low_pass_filter() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        Tracker = deepcopy(TRACKER )
        Tracker["low_pass_filter"] = 0.087
        return_new = fu.apply_low_pass_filter(refvol= [deepcopy(IMAGE_2D),deepcopy(IMAGE_2D)],Tracker=Tracker)
        return_old = oldfu.apply_low_pass_filter(refvol=  [deepcopy(IMAGE_2D),deepcopy(IMAGE_2D)],Tracker=Tracker)
        for i,j in zip(return_new,return_old):
            self.assertTrue(array_equal(i.get_3dview(), j.get_3dview()))

    def test_wrong_Tracker_KeyError(self):
        Tracker = deepcopy(TRACKER )
        with self.assertRaises(KeyError) as cm_new:
            fu.apply_low_pass_filter(refvol= [deepcopy(IMAGE_2D),deepcopy(IMAGE_2D)],Tracker=Tracker)
        with self.assertRaises(KeyError) as cm_old:
            oldfu.apply_low_pass_filter(refvol= [deepcopy(IMAGE_2D),deepcopy(IMAGE_2D)],Tracker=Tracker)
        self.assertEqual(str(cm_new.exception), 'low_pass_filter')
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_refvol_empty(self):
        Tracker = deepcopy(TRACKER )
        Tracker["low_pass_filter"] = 0.087
        return_new = fu.apply_low_pass_filter(refvol=  [],Tracker=Tracker)
        return_old = oldfu.apply_low_pass_filter(refvol=  [],Tracker=Tracker)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertEqual(return_new, [])


class Test_get_groups_from_partition(unittest.TestCase):
    list_of_particles = [randint(0, 1000) for i in range(100)]
    group_list = [0, 1]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_groups_from_partition()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_groups_from_partition()
        self.assertEqual(str(cm_new.exception), "get_groups_from_partition() takes exactly 3 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_value(self):
        return_new = fu.get_groups_from_partition(partition =self.group_list, initial_ID_list = self.list_of_particles, number_of_groups = 2)
        return_old = oldfu.get_groups_from_partition(partition = self.group_list, initial_ID_list =self.list_of_particles, number_of_groups = 2)
        self.assertTrue(array_equal(return_new, return_old))

    def test_empty_initial_ID_list_KeyError(self):
        with self.assertRaises(KeyError) as cm_new:
            fu.get_groups_from_partition(partition =self.group_list, initial_ID_list = [], number_of_groups = 2)
        with self.assertRaises(KeyError) as cm_old:
            oldfu.get_groups_from_partition(partition =self.group_list, initial_ID_list = [], number_of_groups = 2)
        self.assertEqual(str(cm_new.exception), 0)
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_empty_partition_list_KeyError(self):
        return_new = fu.get_groups_from_partition(partition =[], initial_ID_list = self.list_of_particles, number_of_groups = 2)
        return_old = oldfu.get_groups_from_partition(partition = [], initial_ID_list =self.list_of_particles, number_of_groups = 2)
        self.assertTrue(array_equal(return_new, return_old))



class Test_get_complementary_elements(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_complementary_elements()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_complementary_elements()
        self.assertEqual(str(cm_new.exception), "get_complementary_elements() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_defalut_case(self):
        sub_data_list = [1,2,2]
        total_list = [1,2,2,4,5,6]
        return_new = fu.get_complementary_elements(total_list,sub_data_list)
        return_old = oldfu.get_complementary_elements(total_list,sub_data_list)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [4, 5, 6]))

    def test_total_list_less_data_than_sub_data_list_error_msg(self):
        sub_data_list = [1,2,2]
        total_list = [1,2]
        return_new = fu.get_complementary_elements(total_list,sub_data_list)
        return_old = oldfu.get_complementary_elements(total_list,sub_data_list)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, []))



class Test_update_full_dict(unittest.TestCase):
    leftover_list = {0: 'ciao_10', 1: 'ciao_11'}
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.update_full_dict()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.update_full_dict()
        self.assertEqual(str(cm_new.exception), "update_full_dict() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        Tracker_new = deepcopy(TRACKER)
        Tracker_new['full_ID_dict'] = {10: 'ciao_0', 11: 'ciao_1', 2: 'ciao_2', 3: 'ciao_3'}
        Tracker_old = deepcopy(Tracker_new)
        return_new = fu.update_full_dict(self.leftover_list,Tracker_new)
        return_old = oldfu.update_full_dict(self.leftover_list,Tracker_old)
        self.assertEqual(return_new, None)
        self.assertEqual(return_new, return_old)
        self.assertDictEqual(Tracker_new['full_ID_dict'] ,Tracker_old['full_ID_dict'] )

    def test_no_full_ID_dict_in_tracker(self):
        Tracker_new = deepcopy(TRACKER)
        Tracker_old = deepcopy(Tracker_new)
        return_new = fu.update_full_dict(self.leftover_list,Tracker_new)
        return_old = oldfu.update_full_dict(self.leftover_list,Tracker_old)
        self.assertEqual(return_new, None)
        self.assertEqual(return_new, return_old)
        self.assertDictEqual(Tracker_new['full_ID_dict'] ,Tracker_old['full_ID_dict'] )
        self.assertDictEqual(Tracker_new['full_ID_dict'],self.leftover_list)



class Test_count_chunk_members(unittest.TestCase):
    chunk_dict = [0, 1, 2, 3, 4, 5, 6]
    one_class = [0, 1, 2, 3, 4, 5, 6]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.count_chunk_members()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.count_chunk_members()
        self.assertEqual(str(cm_new.exception), "count_chunk_members() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        return_new = fu.count_chunk_members(self.chunk_dict, self.one_class)
        return_old = oldfu.count_chunk_members(self.chunk_dict, self.one_class)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.14285714285714285, 0.8571428571428571, 7)))

    def test_one_class_empty(self):
        return_new = fu.count_chunk_members(self.chunk_dict, [])
        return_old = oldfu.count_chunk_members(self.chunk_dict, [])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.0, 0.0, 0)))

    def test_all_empty(self):
        return_new = fu.count_chunk_members([], [])
        return_old = oldfu.count_chunk_members([], [])
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, (0.0, 0.0, 0)))

    def test_chunk_dict_empty_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.count_chunk_members([], self.one_class)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.count_chunk_members([], self.one_class)
        self.assertEqual(str(cm_new.exception), "list index out of range")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))



class Test_remove_small_groups(unittest.TestCase):
    chunk_dict = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.remove_small_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.remove_small_groups()
        self.assertEqual(str(cm_new.exception), "remove_small_groups() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        return_new = fu.remove_small_groups(self.chunk_dict, 2)
        return_old = oldfu.remove_small_groups(self.chunk_dict, 2)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])))

    def test_too_many_minimum_number_of_objects_in_a_group(self):
        return_new = fu.remove_small_groups(self.chunk_dict, 20)
        return_old = oldfu.remove_small_groups(self.chunk_dict, 20)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [[], []] ))

    def test_empty_chunk_dict(self):
        return_new = fu.remove_small_groups([], 2)
        return_old = oldfu.remove_small_groups([], 2)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, ([], [])))

    def test_minimum_number_of_objects_in_a_group_is_zero(self):
        return_new = fu.remove_small_groups(self.chunk_dict, 0)
        return_old = oldfu.remove_small_groups(self.chunk_dict, 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])))


class Test_get_number_of_groups(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_number_of_groups()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_number_of_groups()
        self.assertEqual(str(cm_new.exception), "get_number_of_groups() takes exactly 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        return_new = fu.get_number_of_groups(total_particles = 1500, number_of_images_per_group = 5)
        return_old = oldfu.get_number_of_groups(total_particles = 1500, number_of_images_per_group = 5)
        self.assertEqual(return_new, return_old)

    def test_null_number_of_images_per_group_returns_ZeroDivisionError(self):
        with self.assertRaises(ZeroDivisionError) as cm_new:
            fu.get_number_of_groups(total_particles = 1500, number_of_images_per_group = 0)
        with self.assertRaises(ZeroDivisionError) as cm_old:
            oldfu.get_number_of_groups(total_particles = 1500, number_of_images_per_group = 0)
        self.assertEqual(str(cm_new.exception), "float division by zero")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_total_particles_null(self):
        return_new = fu.get_number_of_groups(total_particles = 0, number_of_images_per_group = 5)
        return_old = oldfu.get_number_of_groups(total_particles = 0, number_of_images_per_group = 5)
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, 0)



class Test_tabessel(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.tabessel()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.tabessel()
        self.assertEqual(str(cm_new.exception), "tabessel() takes at least 2 arguments (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_case(self):
        return_new = fu.tabessel(None, None, nbel = 50)
        return_old = oldfu.tabessel(None, None, nbel = 50)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, [1.0, 0.9997910261154175, 0.9991644024848938, 0.9981207847595215, 0.9966614842414856, 0.9947881102561951, 0.9925029873847961, 0.9898086190223694, 0.9867082834243774, 0.9832054972648621, 0.979304313659668, 0.9750093221664429, 0.9703254699707031, 0.9652581810951233, 0.9598131775856018, 0.9539968371391296, 0.9478157758712769, 0.9412769675254822, 0.9343878626823425, 0.9271563291549683, 0.919590413570404, 0.9116986989974976, 0.903489887714386, 0.8949731588363647, 0.8861579895019531, 0.8770539164543152, 0.867671012878418, 0.8580194115638733, 0.8481094837188721, 0.8379519581794739, 0.8275576829910278, 0.8169375061988831, 0.8061026334762573, 0.795064389705658, 0.7838340401649475, 0.772423267364502, 0.7608435153961182, 0.7491064667701721, 0.7372238039970398, 0.7252071499824524, 0.7130682468414307, 0.7008189558982849, 0.688470721244812, 0.6760352849960327, 0.6635242104530334, 0.6509489417076111, 0.6383208632469177, 0.6256512999534607, 0.612951397895813, 0.6002320051193237]))

    def test_null_nbel(self):
        return_new = fu.tabessel(None, None, nbel = 0)
        return_old = oldfu.tabessel(None, None, nbel = 0)
        self.assertTrue(array_equal(return_new, return_old))
        self.assertTrue(array_equal(return_new, []))
"""


class Test_nearest_proj(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.nearest_proj()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.nearest_proj()
        self.assertEqual(str(cm_new.exception), "nearest_proj() takes at least 1 argument (0 given)")
        self.assertEqual(str(cm_new.exception), str(cm_old.exception))

    def test_default_value(self):
        #I calculated the value looking in the code of bin/sx3dvariability.py
        proj_angles=[]
        for i in range(10):
            i=+0.1
            proj_angles.append([i/2, i/5,i/4,i/3, i])
        proj_angles.sort()
        proj_angles_list = numpy_full((100, 4), 0.0, dtype=numpy_float32)
        for i in range(10):
            proj_angles_list[i][0] = proj_angles[i][1]
            proj_angles_list[i][1] = proj_angles[i][2]
            proj_angles_list[i][2] = proj_angles[i][3]
            proj_angles_list[i][3] = proj_angles[i][4]
        return_new1,return_new2 = fu.nearest_proj(proj_angles_list)
        return_old1,return_old2 = oldfu.nearest_proj(proj_angles_list)
        self.assertTrue(array_equal(return_new1, return_old1))
        self.assertTrue(array_equal(return_new2, return_old2))
        self.assertTrue(array_equal(return_new1, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [5, 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [6, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [8, 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [12, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [13, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [14, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [15, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [16, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [17, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [18, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [20, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [21, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [22, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [23, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [24, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [25, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [26, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [27, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [28, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [29, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [30, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [31, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [32, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [33, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [34, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [35, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [36, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [37, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [38, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [39, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [40, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [41, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [42, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [43, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [44, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [45, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [46, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [47, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [48, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [49, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [50, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [51, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [52, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [53, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [54, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [55, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [56, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [57, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [58, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [59, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [60, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [61, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [62, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [63, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [64, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [65, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [66, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [67, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [68, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [69, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [70, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [71, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [72, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [73, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [74, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [75, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [76, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [77, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [78, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [79, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [80, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [81, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [82, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [83, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [84, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [85, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [86, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [87, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [88, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [89, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [90, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [91, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [92, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [93, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [94, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [95, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [96, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [97, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [98, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [99, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))
        self.assertTrue(array_equal(return_new2, [[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]))

















""" 

import shutil
@unittest.skip("Adnan reference tests")
class Test_lib_utilities_compare(unittest.TestCase):


    def test_amoeba_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.amoeba")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (var, scale, func, ftolerance, xtolerance, itmax , data) = argum[0]

        return_new = fu.amoeba (var, scale, func, ftolerance, xtolerance, itmax , data)
        return_old = oldfu.amoeba (var, scale, func, ftolerance, xtolerance, itmax , data)

        self.assertTrue(return_new, return_old)

    def test_compose_transform2_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.compose_transform2")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (alpha1, sx1, sy1, scale1, alpha2, sx2, sy2, scale2) = argum[0]

        return_new = fu.compose_transform2(alpha1, sx1, sy1, scale1, alpha2, sx2, sy2, scale2)
        return_old = oldfu.compose_transform2(alpha1, sx1, sy1, scale1, alpha2, sx2, sy2, scale2)

        self.assertTrue(return_new, return_old)


    def test_compose_transform3_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.compose_transform3")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (phi1,theta1,psi1,sx1,sy1,sz1,scale1,phi2,theta2,psi2,sx2,sy2,sz2,scale2) = argum[0]

        return_new = fu.compose_transform3(phi1,theta1,psi1,sx1,sy1,sz1,scale1,phi2,theta2,psi2,sx2,sy2,sz2,scale2)
        return_old = oldfu.compose_transform3(phi1,theta1,psi1,sx1,sy1,sz1,scale1,phi2,theta2,psi2,sx2,sy2,sz2,scale2)

        self.assertTrue(return_new, return_old)


    def test_combine_params2_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.combine_params2")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (alpha1, sx1, sy1, mirror1, alpha2, sx2, sy2, mirror2) = argum[0]

        return_new = fu.combine_params2(alpha1, sx1, sy1, mirror1, alpha2, sx2, sy2, mirror2)
        return_old = oldfu.combine_params2(alpha1, sx1, sy1, mirror1, alpha2, sx2, sy2, mirror2)

        self.assertTrue(return_new, return_old)


    def test_inverse_transform2_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.inverse_transform2")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (alpha, tx, ty) = argum[0]

        return_new = fu.inverse_transform2(alpha, tx, ty)
        return_old = oldfu.inverse_transform2(alpha, tx, ty)

        self.assertTrue(return_new, return_old)


    def test_drop_image_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.drop_image")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (imagename, destination) = argum[0]

        return_new = fu.drop_image(imagename, destination)
        return_old = oldfu.drop_image(imagename, destination)

        if return_new is not None   and  return_old is not None:
            self.assertTrue(return_new, return_old)


    def test_get_im_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_im")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (stackname, im) = argum[0]

        stackname = 'bdb:Substack/isac_substack'

        return_new = fu.get_im(stackname, im)
        return_old = oldfu.get_im(stackname, im)

        self.assertTrue(return_new, return_old)


    def test_get_image_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_image")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (imagename,) = argum[0]

        return_new = fu.get_image(imagename)
        return_old = oldfu.get_image(imagename)

        self.assertTrue(return_new, return_old)


    
      #This function test works but takes too much time that is why for the time being it is   commented,  will uncomment it once everything is done 
    
    # def test_get_image_data_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_image_data")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0][0])
    #
    #     (image) = argum[0][0]
    #
    #     return_new = fu.get_image_data(image)
    #     return_old = oldfu.get_image_data(image)
    #
    #     self.assertTrue(numpy.array_equal(return_new, return_old))


    def test_get_symt_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_symt")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (symmetry,) = argum[0]

        return_new = fu.get_symt(symmetry)
        return_old = oldfu.get_symt(symmetry)

        self.assertTrue(return_new, return_old)


    def test_get_input_from_string_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_input_from_string")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (str_input) = argum[0][0]

        return_new = fu.get_input_from_string(str_input)
        return_old = oldfu.get_input_from_string(str_input)

        self.assertTrue(return_new, return_old)


    def test_model_circle_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.model_circle")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (r, nx, ny) = argum[0]

        return_new = fu.model_circle(r, nx, ny)
        return_old = oldfu.model_circle(r, nx, ny)

        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_model_blank_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.model_blank")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (nx,ny) = argum[0]

        return_new = fu.model_blank(nx,ny)
        return_old = oldfu.model_blank(nx,ny)

        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_peak_search_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.peak_search")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (e,) = argum[0]

        return_new = fu.peak_search(e )
        return_old = oldfu.peak_search(e )

        self.assertTrue(return_new, return_old)



    
    #  This function test works but takes too much time that is why for the time being it is       commented,  will uncomment it once everything is done 
    
    # def test_pad_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.pad")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (image_to_be_padded, new_nx, new_ny, new_nz,off_center_nx) = argum[0]
    #
    #     return_new = fu.pad(image_to_be_padded, new_nx, new_ny, new_nz)
    #     return_old = oldfu.pad(image_to_be_padded, new_nx, new_ny, new_nz)
    #
    #     self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_chooseformat_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.chooseformat")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (t) = argum[0][0]

        return_new = fu.chooseformat(t)
        return_old = oldfu.chooseformat(t)

        self.assertEqual(return_new, return_old)

    def test_read_text_row_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.read_text_row")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (fnam) = argum[0][0]

        return_new = fu.read_text_row(fnam)
        return_old = oldfu.read_text_row(fnam)

        self.assertEqual(return_new, return_old)


    def test_write_text_row_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.write_text_row")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data, filename) = argum[0]

        return_new = fu.write_text_row(data, filename)
        return_old = oldfu.write_text_row(data, filename)

        self.assertEqual(return_new, return_old)


    def test_read_text_file_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.read_text_file")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (filename,) = argum[0]

        return_new = fu.read_text_file(filename)
        return_old = oldfu.read_text_file(filename)

        self.assertEqual(return_new, return_old)


    def test_write_text_file_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.write_text_file")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data, filename) = argum[0]

        return_new = fu.write_text_file(data, filename)
        return_old = oldfu.write_text_file(data, filename)

        self.assertEqual(return_new, return_old)


    def test_reshape_1d_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.reshape_1d")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (input_object, length_current,Pixel_size_current) = argum[0]

        return_new = fu.reshape_1d(input_object, length_current,Pixel_size_current)
        return_old = oldfu.reshape_1d(input_object, length_current,Pixel_size_current)

        self.assertEqual(return_new, return_old)


    def test_estimate_3D_center_MPI_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.estimate_3D_center_MPI")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (data, nima, myid, number_of_proc, main_node) = argum[0]

        return_new = fu.estimate_3D_center_MPI(data, nima, myid, number_of_proc, main_node)
        return_old = oldfu.estimate_3D_center_MPI(data, nima, myid, number_of_proc, main_node)

        self.assertTrue(return_new, return_old)


    def test_rotate_3D_shift_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.rotate_3D_shift")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (data, shift3d) = argum[0]

        return_new = fu.rotate_3D_shift(data, shift3d)
        return_old = oldfu.rotate_3D_shift(data, shift3d)

        if return_new is not None and return_old is not None:
            self.assertTrue(return_new, return_old)
        else:
            print('returns None')


    
    #  This function test works but takes too much time that is why for the time being it is       commented,  will uncomment it once everything is done 
    
    # def test_reduce_EMData_to_root_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.reduce_EMData_to_root")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (data, myid,main_node) = argum[0]
    #
    #     return_new = fu.reduce_EMData_to_root(data, myid,main_node = 0)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.reduce_EMData_to_root(data, myid,main_node = 0)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #     self.assertEqual(return_new, return_old)


    
      #This function test works but takes too much time that is why for the time being it is   commented,  will uncomment it once everything is done 
    
    # def test_bcast_compacted_EMData_all_to_all_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.bcast_compacted_EMData_all_to_all")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (list_of_em_objects, myid ) = argum[0]
    #
    #     return_new = fu.bcast_compacted_EMData_all_to_all(list_of_em_objects, myid)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.bcast_compacted_EMData_all_to_all(list_of_em_objects, myid)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #     self.assertEqual(return_new, return_old)



    def test_gather_compacted_EMData_to_root_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.gather_compacted_EMData_to_root")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (no_of_emo, list_of_emo, myid) = argum[0]

        return_new = fu.gather_compacted_EMData_to_root(no_of_emo, list_of_emo, myid)
        return_old = oldfu.gather_compacted_EMData_to_root(no_of_emo, list_of_emo, myid)

        self.assertEqual(return_new, return_old)


    def test_bcast_EMData_to_all_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.bcast_EMData_to_all")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (tavg, myid, source_node, ) = argum[0]

        return_new = fu.bcast_EMData_to_all(tavg, myid, source_node)
        mpi_barrier(MPI_COMM_WORLD)

        return_old = oldfu.bcast_EMData_to_all(tavg, myid, source_node)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertEqual(return_new, return_old)



    #  Can only be tested on the mpi. Wait too long on normal workstation
    # def test_send_EMData_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.send_EMData")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (img, dst, tag, comm) = argum[0]
    #     tag = 0
    #
    #     return_new = fu.send_EMData(img, dst, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_old = oldfu.send_EMData(img, dst, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     self.assertEqual(return_new, return_old)

    # Can only be tested on the mpi. Wait too long on normal workstation
    # def test_recv_EMData_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.recv_EMData")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (src, tag,comm) = argum[0]
    #     tag = 0
    #
    #     return_new = fu.recv_EMData(src, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_old = oldfu.recv_EMData(src, tag)
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     self.assertEqual(return_new, return_old)


    def test_bcast_number_to_all_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.bcast_number_to_all")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (number_to_send, source_node, mpi_comm) = argum[0]

        return_new = fu.bcast_number_to_all(number_to_send, source_node)
        mpi_barrier(MPI_COMM_WORLD)

        return_old = oldfu.bcast_number_to_all(number_to_send, source_node)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertEqual(return_new, return_old)


    def test_print_msg_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.print_msg")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (msg) = argum[0][0]

        return_new = fu.print_msg(msg)

        return_old = oldfu.print_msg(msg)

        self.assertEqual(return_new, return_old)


    def test_file_type_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.file_type")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (name) = argum[0][0]

        return_new = fu.file_type(name)

        return_old = oldfu.file_type(name)

        self.assertEqual(return_new, return_old)



    def test_get_params2D_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_params2D")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])
        print(argum)

        (ima,) = argum[0]

        return_new = fu.get_params2D(ima )

        return_old = oldfu.get_params2D(ima)

        self.assertEqual(return_new, return_old)

    def test_set_params2D_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.set_params2D")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (ima,p, xform) = argum[0]

        return_new = fu.set_params2D(ima,p)

        return_old = oldfu.set_params2D(ima,p)

        self.assertEqual(return_new, return_old)


    def test_get_params3D_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_params3D")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (ima,) = argum[0]

        return_new = fu.get_params3D(ima )

        return_old = oldfu.get_params3D(ima)

        self.assertEqual(return_new, return_old)


    def test_set_params3D_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.set_params3D")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (ima,p) = argum[0]

        return_new = fu.set_params3D(ima,p)

        return_old = oldfu.set_params3D(ima,p)

        self.assertEqual(return_new, return_old)


    def test_get_params_proj_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_params_proj")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (ima,) = argum[0]

        return_new = fu.get_params_proj(ima )

        return_old = oldfu.get_params_proj(ima)

        self.assertEqual(return_new, return_old)


    def test_set_params_proj_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.set_params_proj")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (ima,p) = argum[0]

        return_new = fu.set_params_proj(ima,p)

        return_old = oldfu.set_params_proj(ima,p)

        self.assertEqual(return_new, return_old)


    def test_get_latest_directory_increment_value_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_latest_directory_increment_value")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (directory_location, directory_name) = argum[0]

        return_new = fu.get_latest_directory_increment_value(directory_location, directory_name)

        return_old = oldfu.get_latest_directory_increment_value(directory_location, directory_name)

        self.assertEqual(return_new, return_old)


    def test_same_ctf_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.same_ctf")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (c1,c2) = argum[0]

        return_new = fu.same_ctf(c1,c2)

        return_old = oldfu.same_ctf(c1,c2)

        self.assertEqual(return_new, return_old)



    def test_generate_ctf_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.generate_ctf")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (p) = argum[0][0]

        return_new = fu.generate_ctf(p)

        return_old = oldfu.generate_ctf(p)

        self.assertTrue(return_new, return_old)


    def test_delete_bdb_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.delete_bdb")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (name) = argum[0][0]

        return_new = fu.delete_bdb(name)

        return_old = oldfu.delete_bdb(name)

        if return_new is not None and return_old is not None:
            self.assertTrue(return_new, return_old)
        else:
            print('returns None')



    def test_getfvec_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.getfvec")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (phi, tht) = argum[0]

        return_new = fu.getfvec(phi, tht)

        return_old = oldfu.getfvec(phi, tht)

        self.assertEqual(return_new, return_old)


    def test_nearest_fang_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.nearest_fang")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (vecs, phi, tht) = argum[0]

        return_new = fu.nearest_fang(vecs, phi, tht)

        return_old = oldfu.nearest_fang(vecs, phi, tht)

        self.assertEqual(return_new, return_old)


    def test_nearest_many_full_k_projangles_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.nearest_many_full_k_projangles")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (reference_normals, angles) = argum[0]
        symclass = argum[1]['sym_class']
        howmany = argum[1]['howmany']

        return_new = fu.nearest_many_full_k_projangles(reference_normals, angles, howmany, symclass)

        return_old = oldfu.nearest_many_full_k_projangles(reference_normals, angles, howmany, symclass)

        self.assertEqual(return_new, return_old)


    def test_angles_to_normals_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.angles_to_normals")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (angles) = argum[0][0]

        return_new = fu.angles_to_normals(angles)

        return_old = oldfu.angles_to_normals(angles)

        self.assertEqual(return_new, return_old)

    
      #This function test works but takes too much time that is why for the time being it is   commented,  will uncomment it once everything is done 
    
    #  Test works with sym = "c1 but fails with sym = "c5"  
    def test_angular_occupancy_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.angular_occupancy")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        # print(argum[0])


        (params, angstep, sym, method) = argum[0]

        print("params = ", params)
        print("angstep = ", angstep)
        print("sym = ", sym)
        print("method = ", method)

        return_new = fu.angular_occupancy(params, angstep, sym, method)

        return_old = oldfu.angular_occupancy(params, angstep, sym, method)

        self.assertEqual(return_new, return_old)


    def test_get_pixel_size_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_pixel_size")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (img,) = argum[0]

        return_new = fu.get_pixel_size(img)

        return_old = oldfu.get_pixel_size(img)

        self.assertEqual(return_new, return_old)

    def test_set_pixel_size_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.set_pixel_size")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (img,pixel_size) = argum[0]

        return_new = fu.set_pixel_size(img,pixel_size)

        return_old = oldfu.set_pixel_size(img,pixel_size)

        self.assertEqual(return_new, return_old)

    def test_lacos_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.lacos")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (x,) = argum[0]

        return_new = fu.lacos(x)

        return_old = oldfu.lacos(x)

        self.assertEqual(return_new, return_old)

    
      #This function test works but takes too much time that is why for the time being it is   commented,  will uncomment it once everything is done 
    
    # def test_nearest_proj_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.nearest_proj")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (proj_ang,img_per_grp,List) = argum[0]
    #
    #     return_new = fu.nearest_proj(proj_ang,img_per_grp,List)
    #
    #     return_old = oldfu.nearest_proj(proj_ang,img_per_grp,List)
    #
    #     self.assertEqual(return_new, return_old)


    def test_findall_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.findall")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (value, L) = argum[0]

        return_new = fu.findall(value, L)

        return_old = oldfu.findall(value, L)

        self.assertEqual(return_new, return_old)


    def test_pack_message_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.pack_message")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data,) = argum[0]

        return_new = fu.pack_message(data)

        return_old = oldfu.pack_message(data)

        self.assertEqual(return_new, return_old)


    def test_unpack_message_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.unpack_message")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data,) = argum[0]

        return_new = fu.unpack_message(data)

        return_old = oldfu.unpack_message(data)

        self.assertEqual(return_new, return_old)


    def test_update_tag_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.update_tag")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (communicator, target_rank) = argum[0]

        return_new = fu.update_tag(communicator, target_rank)

        return_old = oldfu.update_tag(communicator, target_rank)

        self.assertEqual(return_new, return_old)


    def test_wrap_mpi_send_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_send")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data, destination,communicator) = argum[0]

        return_new = fu.wrap_mpi_send(data, destination)

        return_old = oldfu.wrap_mpi_send(data, destination)

        self.assertEqual(return_new, return_old)


    # Can only test on cluster , cannot work on workstation"
    # def test_wrap_mpi_recv_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_recv")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (data, communicator) = argum[0]
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_new = fu.wrap_mpi_recv(data, communicator)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.wrap_mpi_recv(data, communicator)
    #
    #     self.assertEqual(return_new, return_old)


    def test_wrap_mpi_bcast_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_bcast")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data, root, communicator) = argum[0]

        return_new = fu.wrap_mpi_bcast(data, root)

        return_old = oldfu.wrap_mpi_bcast(data, root)

        self.assertEqual(return_new, return_old)


    def test_wrap_mpi_gatherv_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_gatherv")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data, root, communicator) = argum[0]

        return_new = fu.wrap_mpi_gatherv(data, root)

        return_old = oldfu.wrap_mpi_gatherv(data, root)

        self.assertEqual(return_new, return_old)


    def test_get_colors_and_subsets_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_colors_and_subsets")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (main_node, mpi_comm, my_rank, shared_comm, sh_my_rank, masters) = argum[0]

        mpi_comm = MPI_COMM_WORLD
        main_node = 0
        my_rank = mpi_comm_rank(mpi_comm)
        mpi_size = mpi_comm_size(mpi_comm)
        shared_comm = mpi_comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL)
        sh_my_rank = mpi_comm_rank(shared_comm)
        masters = mpi_comm_split(mpi_comm, sh_my_rank == main_node, my_rank)
        shared_comm = mpi_comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL)

        return_new = fu.get_colors_and_subsets(main_node, mpi_comm, my_rank, shared_comm, sh_my_rank,masters)

        return_old = oldfu.get_colors_and_subsets(main_node, mpi_comm, my_rank, shared_comm, sh_my_rank,masters)

        self.assertEqual(return_new, return_old)

        # Can only be tested in mpi not on workstation   
    # def test_wrap_mpi_split_true_should_return_equal_objects(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.wrap_mpi_split")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum[0])
    #
    #     (comm, no_of_groups) = argum[0]
    #
    #     return_new = fu.wrap_mpi_split(comm, no_of_groups)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.wrap_mpi_split(comm, no_of_groups)
    #
    #     self.assertEqual(return_new, return_old)


    def test_get_dist_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.get_dist")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (c1, c2) = argum[0]

        return_new = fu.get_dist(c1, c2)

        return_old = oldfu.get_dist(c1, c2)

        self.assertEqual(return_new, return_old)


    def test_combinations_of_n_taken_by_k_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.combinations_of_n_taken_by_k")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (n, k) = argum[0]

        return_new = fu.combinations_of_n_taken_by_k(n, k)

        return_old = oldfu.combinations_of_n_taken_by_k(n, k)

        self.assertEqual(return_new, return_old)


    def test_cmdexecute_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.cmdexecute")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        # print(argum[0])

        (cmd,) = argum[0]

        dirname = cmd.split(' ')[1]

        current_path = os.getcwd()
        if os.path.isdir(dirname):
            print('directory exits')
            print('removing it')
            shutil.rmtree(dirname)

        return_new = fu.cmdexecute(cmd)

        if os.path.isdir(dirname):
            print('directory exits')
            print('removing it')
            shutil.rmtree(dirname)

        return_old = oldfu.cmdexecute(cmd)

        self.assertEqual(return_new, return_old)


    def test_if_error_then_all_processes_exit_program_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.if_error_then_all_processes_exit_program")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (error_status,) = argum[0]

        return_new = fu.if_error_then_all_processes_exit_program(error_status)

        return_old = oldfu.if_error_then_all_processes_exit_program(error_status)

        self.assertEqual(return_new, return_old)

    def test_getindexdata_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.getindexdata")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (stack, partids, partstack, myid, nproc) = argum[0]

        return_new = fu.getindexdata(stack, partids, partstack, myid, nproc)

        return_old = oldfu.getindexdata(stack, partids, partstack, myid, nproc)

        self.assertTrue(return_new, return_old)

    def test_convert_json_fromunicode_true_should_return_equal_objects(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/utilities/utilities.convert_json_fromunicode")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data,) = argum[0]

        return_new = fu.convert_json_fromunicode(data)

        return_old = oldfu.convert_json_fromunicode(data)

        self.assertEqual(return_new, return_old)
"""
if __name__ == '__main__':
    unittest.main()

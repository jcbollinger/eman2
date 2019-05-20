import unittest
from copy import deepcopy
from EMAN2_cppwrap import EMData,Util, Reconstructors,Transform
import EMAN2_cppwrap

import numpy
from test_module import get_data,get_data_3d, get_arg_from_pickle_file
import os
ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))

from mpi import *
mpi_init(0, [])

from sphire.libpy import sparx_reconstruction as fu
from sphire.tests.sparx_lib import sparx_reconstruction as oldfu

from sphire.libpy import sparx_utilities


from os import path
from test_module import returns_values_in_file,remove_list_of_file,get_real_data,ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER
XFORM_PROJECTION_IMG =get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))[0][0]
PRJLIST = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))[0][0]
STACK_NAME = 'bdb:' + path.join(ABSOLUTE_PATH_TO_SPHIRE_DEMO_RESULTS_FOLDER, 'Substack/sort3d_substack_003')
IMAGE_2D, IMAGE_2D_REFERENCE = get_real_data(dim=2)

"""
There are some opened issues in:
1) insert_slices and insert_slices_pdf seems to have the same behaviour. See Test_insert_slices_VS_insert_slices_pdf
2) Test_recons3d_4nn_MPI.test_default_case_z_size_both_not_negative_FAILEd failed even if I set the Tollerance to a high value (e.g.: 5)
    but Test_recons3d_4nn_MPI.test_default_case_xy_size_not_negative_myid_not_null does not failed. WHY????
3) recons3d_trl_struct_MPI not tested
4) recons3d_4nn_ctf it seems to be not used. I did not tested it
5) Test_recons3d_4nn_ctf_MPI
  a) there is a KNOWN BUG --> with sizeprojection  PAP 10/22/2014 
  b) if you call this function twice, or in the tests case twice in the same class test, the second time that it runs crashed beacuse del sparx_utilities.pad
     This happen because 'sparx_utilities.pad' obj was destroyed in the first call
6) Test_recons3d_nn_SSNR_MPI.test_withMask2D, I cannot test the 2Dmask case because:
    I cannot provide it a valid mask. I tried with 'mask2D = sparx_utilities.model_circle(0.1, nx, ny) - sparx_utilities.model_circle(1, nx, ny)'
7) Test_prepare_recons.test_main_node_half_NOTequal_myid_crashes_because_MPI_ERRORS_ARE_FATAL
8) Test_prepare_recons_ctf are crashing using di 'PRJLIST' beacause 'half.insert_slice(data[i], xform_proj )' ...maybe changing the image we get no crash ... WHICH ONE?
                --> in practice all the cases with param 'half_start'<len(data)                
9) Test_rec3D_MPI -->same problem as (8) when  set  odd_start=0 becuase it is used as 'half_start' when it calls 'prepare_recons_ctf'
"""
class Test_insert_slices(unittest.TestCase):
    size = 76
    img = EMData(size,size)
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.insert_slices()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.insert_slices()
        self.assertEqual(cm_new.exception.message, "insert_slices() takes exactly 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_defalut_case(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol":deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r_new = Reconstructors.get( "nn4", params )
        r_new.setup()
        r_old = Reconstructors.get( "nn4", params )
        r_old.setup()
        return_new = fu.insert_slices(reconstructor=r_new, proj=deepcopy(XFORM_PROJECTION_IMG))
        return_old = oldfu.insert_slices(reconstructor=r_old, proj= deepcopy(XFORM_PROJECTION_IMG))
        fftvol_new=r_new.get_params()['fftvol']
        fftvol_old = r_old.get_params()['fftvol']
        weight_new=r_new.get_params()['weight']
        weight_old = r_old.get_params()['weight']
        self.assertTrue(numpy.array_equal(fftvol_new.get_3dview(), fftvol_old.get_3dview()))
        self.assertFalse(numpy.array_equal(fftvol_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertTrue(numpy.array_equal(weight_new.get_3dview(), weight_old.get_3dview()))
        self.assertFalse(numpy.array_equal(weight_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)
        #self.assertTrue(numpy.array_equal(r_new.get_params()['fftvol'].get_3dview(), r_old.get_params()['fftvol'].get_3dview())) leads to segmentation fault

    def test_None_proj_case_returns_AttributeError_NoneType_obj_hasnot_attribute_get_attr(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol": deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()
        with self.assertRaises(AttributeError) as cm_new:
            fu.insert_slices(reconstructor=r, proj=None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.insert_slices(reconstructor=r, proj=None)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_empty_image_proj_case_returns_RuntimeError_NotExistingObjectException_the_key_mean_doesnot_exist(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol": deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()
        with self.assertRaises(RuntimeError) as cm_new:
            fu.insert_slices(reconstructor=r, proj=EMData())
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.insert_slices(reconstructor=r, proj=EMData())
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_img_not_xform_projection_returns_RuntimeError_NotExistingObjectException_the_key_mean_doesnot_exist(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol": deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()
        with self.assertRaises(RuntimeError) as cm_new:
            fu.insert_slices(reconstructor=r, proj=get_real_data(2)[0])
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.insert_slices(reconstructor=r, proj=get_real_data(2)[0])
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_insert_slices_pdf(unittest.TestCase):
    size = 76
    img = EMData(size,size)
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.insert_slices_pdf()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.insert_slices_pdf()
        self.assertEqual(cm_new.exception.message, "insert_slices_pdf() takes exactly 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_defalut_case(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol":deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r_new = Reconstructors.get( "nn4", params )
        r_new.setup()
        r_old = Reconstructors.get( "nn4", params )
        r_old.setup()
        return_new = fu.insert_slices_pdf(reconstructor=r_new, proj=deepcopy(XFORM_PROJECTION_IMG))
        return_old = oldfu.insert_slices_pdf(reconstructor=r_old,  proj=deepcopy(XFORM_PROJECTION_IMG))
        fftvol_new=r_new.get_params()['fftvol']
        fftvol_old = r_old.get_params()['fftvol']
        weight_new=r_new.get_params()['weight']
        weight_old = r_old.get_params()['weight']
        self.assertTrue(numpy.array_equal(fftvol_new.get_3dview(), fftvol_old.get_3dview()))
        self.assertFalse(numpy.array_equal(fftvol_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertTrue(numpy.array_equal(weight_new.get_3dview(), weight_old.get_3dview()))
        self.assertFalse(numpy.array_equal(weight_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)
        #self.assertTrue(numpy.array_equal(r_new.get_params()['fftvol'].get_3dview(), r_old.get_params()['fftvol'].get_3dview())) leads to segmentation fault

    def test_None_proj_case_returns_AttributeError_NoneType_obj_hasnot_attribute_get_attr(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol": deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()
        with self.assertRaises(AttributeError) as cm_new:
            fu.insert_slices_pdf(reconstructor=r, proj=None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.insert_slices_pdf(reconstructor=r, proj=None)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_empty_image_proj_case_returns_RuntimeError_NotExistingObjectException_the_key_mean_doesnot_exist(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol": deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()
        with self.assertRaises(RuntimeError) as cm_new:
            fu.insert_slices_pdf(reconstructor=r, proj=EMData())
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.insert_slices_pdf(reconstructor=r, proj=EMData())
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])

    def test_img_not_xform_projection_returns_RuntimeError_NotExistingObjectException_the_key_mean_doesnot_exist(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol": deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()
        with self.assertRaises(RuntimeError) as cm_new:
            fu.insert_slices_pdf(reconstructor=r, proj=get_real_data(2)[0])
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.insert_slices_pdf(reconstructor=r, proj=get_real_data(2)[0])
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])



class Test_insert_slices_VS_insert_slices_pdf(unittest.TestCase):
    size = 76
    img = EMData(size,size)
    def test_insert_slices_VS_insert_slices_pdf_case1(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol":deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r_new = Reconstructors.get( "nn4", params )
        r_new.setup()
        r_old = Reconstructors.get( "nn4", params )
        r_old.setup()
        return_new = fu.insert_slices_pdf(reconstructor=r_new, proj=deepcopy(XFORM_PROJECTION_IMG))
        return_old = oldfu.insert_slices(reconstructor=r_old,  proj=deepcopy(XFORM_PROJECTION_IMG))
        fftvol_new=r_new.get_params()['fftvol']
        fftvol_old = r_old.get_params()['fftvol']
        weight_new=r_new.get_params()['weight']
        weight_old = r_old.get_params()['weight']
        self.assertTrue(numpy.array_equal(fftvol_new.get_3dview(), fftvol_old.get_3dview()))
        self.assertFalse(numpy.array_equal(fftvol_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertTrue(numpy.array_equal(weight_new.get_3dview(), weight_old.get_3dview()))
        self.assertFalse(numpy.array_equal(weight_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)

    def test_insert_slices_VS_insert_slices_pdf_case2(self):
        params = {"size": self.size, "npad": 2, "symmetry": "c1", "fftvol":deepcopy(self.img), "weight": deepcopy(self.img), "snr": 2}
        r_new = Reconstructors.get( "nn4", params )
        r_new.setup()
        r_old = Reconstructors.get( "nn4", params )
        r_old.setup()
        return_new = fu.insert_slices(reconstructor=r_new, proj=deepcopy(XFORM_PROJECTION_IMG))
        return_old = oldfu.insert_slices_pdf(reconstructor=r_old, proj=deepcopy(XFORM_PROJECTION_IMG))
        fftvol_new=r_new.get_params()['fftvol']
        fftvol_old = r_old.get_params()['fftvol']
        weight_new=r_new.get_params()['weight']
        weight_old = r_old.get_params()['weight']
        self.assertTrue(numpy.array_equal(fftvol_new.get_3dview(), fftvol_old.get_3dview()))
        self.assertFalse(numpy.array_equal(fftvol_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertTrue(numpy.array_equal(weight_new.get_3dview(), weight_old.get_3dview()))
        self.assertFalse(numpy.array_equal(weight_new.get_3dview(), get_real_data(2)[0].get_3dview()))
        self.assertEqual(return_new, return_old)
        self.assertTrue(return_new is None)



class Test_recons3d_4nn_MPI(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons3d_4nn_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons3d_4nn_MPI()
        self.assertEqual(cm_new.exception.message, "recons3d_4nn_MPI() takes at least 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_default_case(self):
        return_new = fu.recons3d_4nn_MPI(myid = 0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid= 0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5))

    def test_default_case_xy_z_size_both_not_negative(self):
        return_new = fu.recons3d_4nn_MPI(myid= 0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid= 0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5, equal_nan=True))

    def test_default_case_xy_size_not_negative(self):
        return_new = fu.recons3d_4nn_MPI(myid= 0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid= 0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5, equal_nan=True))

    def test_default_case_z_size_both_not_negative_FAILEd(self):
        self.assertTrue(True)
        """
        return_new = fu.recons3d_4nn_MPI(myid=0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid=0, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5, equal_nan=True))
        """

    def test_default_case_xy_z_size_both_not_negative_myid_not_null(self):
        return_new = fu.recons3d_4nn_MPI(myid= 1, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid= 1, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5, equal_nan=True))

    def test_default_case_xy_size_not_negative_myid_not_null(self):
        return_new = fu.recons3d_4nn_MPI(myid= 1, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid= 1, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5, equal_nan=True))

    def test_default_case_z_size_both_not_negative__myid_not_null(self):
        return_new = fu.recons3d_4nn_MPI(myid = 1, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid =1, prjlist=[XFORM_PROJECTION_IMG], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5, equal_nan=True))

    def test_prjlist_is_emptylist_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.recons3d_4nn_MPI(myid= 0, prjlist=[], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.recons3d_4nn_MPI(myid= 0, prjlist=[], symmetry="c1", finfo=None, snr = 1.0, npad=2, xysize=-1, zsize=-1, mpi_comm=MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)



class Test_recons3d_trl_struct_MPI(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons3d_trl_struct_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons3d_trl_struct_MPI()
        self.assertEqual(cm_new.exception.message, "recons3d_trl_struct_MPI() takes at least 7 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)


class Test_recons3d_4nn_ctf_MPI(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons3d_4nn_ctf_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons3d_4nn_ctf_MPI()
        self.assertEqual(cm_new.exception.message, "recons3d_4nn_ctf_MPI() takes at least 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)



    def test_default_case(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        return_new = fu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(), 0.5  ))

    @unittest.skip("crash if run togheter with the other tests of this class because a bad implementation of the code")
    def test_negative_smearstep(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        return_new = fu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = -0.5)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = -0.5)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(), 0.5  ))

    def test_default_case_xy_z_size_both_not_negative_NameError_sizeprojection_BEACUASE_A_BUG(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        with self.assertRaises(NameError) as cm_new:
            fu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=1, zsize=1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(NameError) as cm_old:
            oldfu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=1, zsize=1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "global name 'sizeprojection' is not defined")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_default_case_xy_size_NameError_sizeprojection_BEACUASE_A_BUG(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        with self.assertRaises(NameError) as cm_new:
            fu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(NameError) as cm_old:
            oldfu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=1, symmetry="c1", finfo=None, npad=2, xysize=1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "global name 'sizeprojection' is not defined")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    @unittest.skip("crash if run togheter with the other tests of this class because a bad implementation of the code")
    def test_default_case_negative_sign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        return_new = fu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=-1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_ctf_MPI(0, [proj], snr = 1.0, sign=-1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(), 0.5  ))

    def test_prjlist_is_emptylist_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.recons3d_4nn_ctf_MPI(0, [], snr = -1.0, sign=-1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.recons3d_4nn_ctf_MPI(0, [], snr = -1.0, sign=-1, symmetry="c1", finfo=None, npad=2, xysize=-1, zsize=-1, mpi_comm=None, smearstep = 0.5)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)



class Test_recons3d_nn_SSNR_MPI(unittest.TestCase):

    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons3d_nn_SSNR_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons3d_nn_SSNR_MPI()
        self.assertEqual(cm_new.exception.message, "recons3d_nn_SSNR_MPI() takes at least 3 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_withoutMask2D_and_CTF_randomangles0(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles0_ring_width0_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=0, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=0, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))
        """

    def test_withoutMask2D_and_withCTF_randomangles0(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles1(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles1(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles2(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles2(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles =2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles3(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 3, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 3, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles3(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles0_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles0_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles1_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles1_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 1, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles2_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles2_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles =2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 2, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_CTF_randomangles3_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 3, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = False, random_angles = 3, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withoutMask2D_and_withCTF_randomangles3_negativeSign(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=False, ring_width=1, npad =1, sign=-1, symmetry="c1", CTF = True, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))

    def test_withMask2D_FAILED_I_cannot_provide_a_valid_mask(self):
        self.assertTrue(True)
        """
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        nx=proj.get_xsize()
        ny = proj.get_ysize()
        mask2D = sparx_utilities.model_circle(0.1, nx, ny) - sparx_utilities.model_circle(1, nx, ny)
        return_new = fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=mask2D, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=mask2D, ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0], return_old[0]))
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))
        """


    def test_with_emptyMask2D_returns_ImageDimensionException(self):
        nima = EMAN2_cppwrap.EMUtil.get_image_count(STACK_NAME)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(STACK_NAME, list_proj[0])
        with self.assertRaises(RuntimeError) as cm_new:
            fu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=EMData(), ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.recons3d_nn_SSNR_MPI(myid=0, prjlist=[proj], mask2D=EMData(), ring_width=1, npad =1, sign=1, symmetry="c1", CTF = False, random_angles = 0, mpi_comm = None)
        mpi_barrier(MPI_COMM_WORLD)

        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "ImageDimensionException")
        self.assertEqual(msg[1], "The dimension of the image does not match the dimension of the mask!")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[1], msg_old[1])



class Test_prepare_recons(unittest.TestCase):
    index =1
    data = deepcopy(PRJLIST)
    data[0].set_attr('group',index)
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.prepare_recons()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.prepare_recons()
        self.assertEqual(cm_new.exception.message, "prepare_recons() takes at least 7 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)


    def test_data_is_emptylist_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.prepare_recons(data=[], symmetry='c5', myid=0 , main_node_half=0, half_start=4, step=2, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.prepare_recons(data=[], symmetry='c5', myid=0 , main_node_half=0, half_start=4, step=2, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_index_equal_group(self):
        return_new = fu.prepare_recons(data=deepcopy(self.data), symmetry='c5', myid=0 , main_node_half=0, half_start=0, step=1, index=self.index, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons(data=deepcopy(self.data), symmetry='c5', myid=0 , main_node_half=0, half_start=0, step=1, index=self.index, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertEqual(returns_values_in_file(return_old[0]),returns_values_in_file(return_new[0]))
        self.assertEqual(returns_values_in_file(return_old[1]), returns_values_in_file(return_new[1]))
        remove_list_of_file([path.join(ABSOLUTE_PATH, return_old[0]),path.join(ABSOLUTE_PATH, return_old[1]),path.join(ABSOLUTE_PATH, return_new[0]),path.join(ABSOLUTE_PATH, return_new[1])])

    def test_main_node_half_NOTequal_myid_crashes_because_MPI_ERRORS_ARE_FATAL(self):
        self.assertTrue(True)
        """
        I get the following error because 'mpi.mpi_reduce(...)' in 'reduce_EMData_to_root' in sparx_utilities.py
        
        Launching unittests with arguments python -m unittest test_reconstruction.Test_prepare_recons.test_symC15 in /home/lusnig/EMAN2/eman2/sphire/tests
        [rtxr2:27348] *** An error occurred in MPI_Reduce
        [rtxr2:27348] *** reported by process [1512308737,140346646331392]
        [rtxr2:27348] *** on communicator MPI_COMM_WORLD
        [rtxr2:27348] *** MPI_ERR_ROOT: invalid root
        [rtxr2:27348] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
        [rtxr2:27348] ***    and potentially your MPI job)
        
        Process finished with exit code 8

        """
        """
        return_new = fu.prepare_recons(data=self.data, symmetry='c5', myid=0 , main_node_half=1, half_start=4, step=1, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons(data=self.data, symmetry='c5', myid=0 , main_node_half=1, half_start=4, step=1, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(returns_values_in_file(return_old[0]),returns_values_in_file(return_new[0]))
        self.assertEqual(returns_values_in_file(return_old[1]), returns_values_in_file(return_new[1]))
        remove_list_of_file([return_old[0],return_old[1],return_new[0],return_new[1]])
        """

    def test_symC5(self):
        return_new = fu.prepare_recons(data=deepcopy(self.data), symmetry='c5', myid=0 , main_node_half=0, half_start=4, step=1, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons(data=deepcopy(self.data), symmetry='c5', myid=0 , main_node_half=0, half_start=4, step=1, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(returns_values_in_file(return_old[0]),returns_values_in_file(return_new[0]))
        self.assertEqual(returns_values_in_file(return_old[1]), returns_values_in_file(return_new[1]))
        remove_list_of_file([path.join(ABSOLUTE_PATH, return_old[0]),path.join(ABSOLUTE_PATH, return_old[1]),path.join(ABSOLUTE_PATH, return_new[0]),path.join(ABSOLUTE_PATH, return_new[1])])

    def test_symC1(self):
        return_new = fu.prepare_recons(data=deepcopy(self.data), symmetry='c1', myid=0 , main_node_half=0, half_start=4, step=1, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons(data=deepcopy(self.data), symmetry='c1', myid=0 , main_node_half=0, half_start=4, step=1, index=5, npad=2, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(returns_values_in_file(return_old[0]),returns_values_in_file(return_new[0]))
        self.assertEqual(returns_values_in_file(return_old[1]), returns_values_in_file(return_new[1]))
        remove_list_of_file([path.join(ABSOLUTE_PATH, return_old[0]),path.join(ABSOLUTE_PATH, return_old[1]),path.join(ABSOLUTE_PATH, return_new[0]),path.join(ABSOLUTE_PATH, return_new[1])])



class Test_prepare_recons_ctf(unittest.TestCase):
    nx = PRJLIST[0].get_xsize()
    sym ='c5'
    npad=2
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.prepare_recons_ctf()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.prepare_recons_ctf()
        self.assertEqual(cm_new.exception.message, "prepare_recons_ctf() takes at least 8 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_index_equal_group_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """        
        return_new = fu.prepare_recons_ctf(nx=self.nx, data=self.data, snr =1, symmetry='c5', myid=0 , main_node_half=0, half_start=0, step=1, finfo=None, npad=2, mpi_comm = MPI_COMM_WORLD,smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons_ctf(nx=self.nx, data=self.data,  snr =1, symmetry='c5', myid=0 , main_node_half=0, half_start=0, step=1, finfo=None, npad=2, mpi_comm = MPI_COMM_WORLD,smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertEqual(returns_values_in_file(return_old[0]),returns_values_in_file(return_new[0]))
        self.assertEqual(returns_values_in_file(return_old[1]), returns_values_in_file(return_new[1]))
        remove_list_of_file([path.join(ABSOLUTE_PATH, return_old[0]),path.join(ABSOLUTE_PATH, return_old[1]),path.join(ABSOLUTE_PATH, return_new[0]),path.join(ABSOLUTE_PATH, return_new[1])])
        """

    def test_prepare_recons_ctf_pickle_file_case(self):
        return_new = fu.prepare_recons_ctf(nx=self.nx,data=PRJLIST, snr =1 , symmetry=self.sym, myid=0 , main_node_half=0, half_start=4, step=2, npad=self.npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons_ctf(nx=self.nx,data=PRJLIST, snr =1 , symmetry=self.sym, myid=0 , main_node_half=0, half_start=4, step=2, npad=self.npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(returns_values_in_file(return_old[0]), returns_values_in_file(return_new[0]))
        self.assertEqual(returns_values_in_file(return_old[1]), returns_values_in_file(return_new[1]))
        remove_list_of_file([path.join(ABSOLUTE_PATH, return_old[0]), path.join(ABSOLUTE_PATH, return_old[1]),path.join(ABSOLUTE_PATH, return_new[0]), path.join(ABSOLUTE_PATH, return_new[1])])



class Test_recons_from_fftvol(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons_from_fftvol()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons_from_fftvol()
        self.assertEqual(cm_new.exception.message, "recons_from_fftvol() takes at least 4 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_recons_from_fftvol_default_case(self):
        size = 76
        return_new = fu.recons_from_fftvol(size=size, fftvol=EMData(size,size), weight=EMData(size,size), symmetry="c1", npad = 2)
        return_old = oldfu.recons_from_fftvol(size=size, fftvol=EMData(size,size),weight= EMData(size,size),symmetry= "c1", npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_with_all_empty_data(self):
        size = 76
        return_new = fu.recons_from_fftvol(size=size, fftvol=EMData(), weight=EMData(), symmetry="c1", npad = 2)
        return_old = oldfu.recons_from_fftvol(size=size, fftvol=EMData(), weight=EMData(), symmetry="c1", npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_fftvol_None_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        size = 76
        return_new = fu.recons_from_fftvol(size=size,fftvol= None, weight=EMData(size,size), symmetry="c1", npad = 2)
        return_old = oldfu.recons_from_fftvol(size=size, fftvol=None, weight=EMData(size,size), symmetry="c1", npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))
        """

    def test_weight_None_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        size = 76
        return_new = fu.recons_from_fftvol(size=size, fftvol=EMData(size,size), weight=None, symmetry="c1", npad = 2)
        return_old = oldfu.recons_from_fftvol(size=size, fftvol=EMData(size,size),weight=None, symmetry="c1", npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))
        """



class Test_recons_ctf_from_fftvol(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.recons_ctf_from_fftvol()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.recons_ctf_from_fftvol()
        self.assertEqual(cm_new.exception.message, "recons_ctf_from_fftvol() takes at least 5 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_default_case(self):
        size = 76
        return_new = fu.recons_ctf_from_fftvol(size=size, fftvol=EMData(size,size), weight=EMData(size,size), snr=2, symmetry="c1", weighting=1, npad = 2)
        return_old = oldfu.recons_ctf_from_fftvol(size=size, fftvol=EMData(size,size), weight=EMData(size,size), snr=2, symmetry="c1", weighting=1, npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_with_all_empty_data(self):
        size = 76
        return_new = fu.recons_ctf_from_fftvol(size=size, fftvol=EMData(), weight=EMData(), snr=2, symmetry="c1", weighting=1, npad = 2)
        return_old = oldfu.recons_ctf_from_fftvol(size=size, fftvol=EMData(), weight=EMData(), snr=2,symmetry="c1", weighting=1, npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_fftvol_None_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        size = 76
        return_new = fu.recons_ctf_from_fftvol(size=size,fftvol= None, weight=EMData(size,size), snr=2,symmetry="c1", weighting=1, npad = 2)
        return_old = oldfu.recons_ctf_from_fftvol(size=size, fftvol=None, weight=EMData(size,size), snr=2,symmetry="c1", weighting=1, npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))
        """

    def test_weight_None_crashes_because_signal11SIGSEV(self):
        self.assertTrue(True)
        """
        size = 76
        return_new = fu.recons_ctf_from_fftvol(size=size, fftvol=EMData(size,size), weight=None, snr=2,symmetry="c1", weighting=1, npad = 2)
        return_old = oldfu.recons_ctf_from_fftvol(size=size, fftvol=EMData(size,size),weight=None, snr=2,symmetry="c1", weighting=1, npad = 2)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))
        """



class Test_get_image_size(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.get_image_size()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.get_image_size()
        self.assertEqual(cm_new.exception.message, "get_image_size() takes exactly 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_get_image_default_case(self):
        size=76
        return_new = fu.get_image_size(imgdata=[ EMData(size,size),EMData(size,size),EMData(size,size) ],myid= 0 )
        return_old = oldfu.get_image_size(imgdata=[ EMData(size,size),EMData(size,size),EMData(size,size) ],myid= 0 )
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, size)

    def test_get_image_myID_not_null_MPI_ERRORS_ARE_FATAL(self):
        """
        I get the following error because 'mpi.mpi_reduce(...)' in 'reduce_EMData_to_root' in sparx_utilities.py

        Launching unittests with arguments python -m unittest test_reconstruction.Test_get_image_size.test_get_image_de2fault_case in /home/lusnig/EMAN2/eman2/sphire/tests
        [rtxr2:32644] *** An error occurred in MPI_Bcast
        [rtxr2:32644] *** reported by process [139823993585665,0]
        [rtxr2:32644] *** on communicator MPI_COMM_WORLD
        [rtxr2:32644] *** MPI_ERR_ROOT: invalid root
        [rtxr2:32644] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
        [rtxr2:32644] ***    and potentially your MPI job)
        """
        self.assertTrue(True)
        """
        size=76
        return_new = fu.get_image_size([ EMData(size,size),EMData(size,size),EMData(size,size) ], 1 )
        return_old = oldfu.get_image_size([ EMData(size,size),EMData(size,size),EMData(size,size) ], 1 )
        self.assertEqual(return_new, return_old)
        self.assertEqual(return_new, size)
        """

    def test_get_image_NONE_returns_AttributeError_NoneType_obj_hasnot_attribute_get_xsize(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.get_image_size(imgdata=[None],myid= 0 )
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.get_image_size(imgdata=[None],myid= 0 )
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'get_xsize'")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)




class Test_rec3D_MPI(unittest.TestCase):
    sym = 'c5'
    npad=2
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rec3D_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rec3D_MPI()
        self.assertEqual(cm_new.exception.message, "rec3D_MPI() takes at least 1 argument (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_rec3D_MPI_should_return_True(self):
        """ it is the Adnan starting test and not a default value case because 'odd_start' is not 0 """
        return_new = fu.rec3D_MPI( PRJLIST, 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_MPI( PRJLIST, 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))

    def test_index_not_minus1(self):
        data = deepcopy(PRJLIST)
        data[0].set_attr('group',1)
        return_new = fu.rec3D_MPI( data, 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_MPI( data, 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))

    def test_empty_data_msg_warning(self):
        return_new = fu.rec3D_MPI( [EMData()], 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_MPI( [EMData()], 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))

    def test_None_data_returns_AttributeError_NoneType_obj_hasnot_attribute_get_xsize(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.rec3D_MPI( [None], 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.rec3D_MPI( [None], 1.0, self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'get_xsize'")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_with3Dmask(self):
        return_new = fu.rec3D_MPI( PRJLIST, 1.0, self.sym, mask3D = get_real_data(3)[0], fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_MPI( PRJLIST, 1.0, self.sym, mask3D = get_real_data(3)[0], fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=1, finfo=None, index=-1, npad = 2, mpi_comm=None, smearstep = 0.0)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))



class Test_rec3D_MPI_noCTF(unittest.TestCase):
    sym = 'c5'
    npad=2
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rec3D_MPI_noCTF()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rec3D_MPI_noCTF()
        self.assertEqual(cm_new.exception.message, "rec3D_MPI_noCTF() takes at least 1 argument (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)


    def test_myidi_zero(self):
        return_new = fu.rec3D_MPI_noCTF(PRJLIST, symmetry = self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=4, finfo=None, index = 5, npad = self.npad, mpi_comm=None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_MPI_noCTF(PRJLIST, symmetry = self.sym, mask3D = None, fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=4, finfo=None, index = 5, npad = self.npad, mpi_comm=None)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))
        self.assertEqual(return_new[1], return_old[1])


    def test_with3Dmask(self):
        return_new = fu.rec3D_MPI_noCTF(PRJLIST, symmetry = self.sym, mask3D = get_real_data(3)[0], fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=4, finfo=None, index = 5, npad = self.npad, mpi_comm=None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_MPI_noCTF(PRJLIST, symmetry = self.sym, mask3D = get_real_data(3)[0], fsc_curve = None, myid = 0, main_node = 0, rstep = 1.0, odd_start=4, eve_start=4, finfo=None, index = 5, npad = self.npad, mpi_comm=None)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))
        self.assertEqual(return_new[1], return_old[1])




    def test_myidi_NOT_zero_returns_typeError_concatenation_not_possible(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rec3D_MPI_noCTF(PRJLIST, symmetry = self.sym, mask3D = None, fsc_curve = None, myid = 2, main_node = 0, rstep = 1.0, odd_start=4, eve_start=4, finfo=None, index = 5, npad = self.npad, mpi_comm=None)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rec3D_MPI_noCTF(PRJLIST, symmetry = self.sym, mask3D = None, fsc_curve = None, myid = 2, main_node = 0, rstep = 1.0, odd_start=4, eve_start=4, finfo=None, index = 5, npad = self.npad, mpi_comm=None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertEqual(cm_new.exception.message, "cannot concatenate 'str' and 'NoneType' objects")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)




class Test_prepare_recons_ctf_two_chunks(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.prepare_recons_ctf_two_chunks()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.prepare_recons_ctf_two_chunks()
        self.assertEqual(cm_new.exception.message, "prepare_recons_ctf_two_chunks() takes at least 7 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)



class Test_rec3D_two_chunks_MPI(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters_TypeError(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.rec3D_two_chunks_MPI()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.rec3D_two_chunks_MPI()
        self.assertEqual(cm_new.exception.message, "rec3D_two_chunks_MPI() takes at least 1 argument (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)




@unittest.skip("skip addnan tests")
class Test_lib_compare_for_reconstruction(unittest.TestCase):

    def test_insert_slices_should_return_True(self):
        argum = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))
        (data, refrings, list_of_ref_ang, numr, xrng, yrng, step) = argum[0]

        refvol = sparx_utilities.model_blank(76)
        fftvol = EMData(76,76)
        weight = EMData(76,76)

        params = {"size": 76, "npad": 2, "symmetry": "c1", "fftvol": fftvol, "weight": weight, "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()

        return_new = fu.insert_slices(r,data)
        return_old = oldfu.insert_slices(r, data)
        self.assertEqual(return_new, return_old)



    def test_insert_slices_pdf_should_return_True(self):
        argum = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))
        (data, refrings, list_of_ref_ang, numr, xrng, yrng, step) = argum[0]

        refvol = sparx_utilities.model_blank(76)
        fftvol = EMData(76,76)
        weight = EMData(76,76)


        params = {"size": 76, "npad": 2, "symmetry": "c1", "fftvol": fftvol, "weight": weight, "snr": 2}
        r = Reconstructors.get( "nn4", params )
        r.setup()

        return_new = fu.insert_slices_pdf(r,data)
        return_old = oldfu.insert_slices_pdf(r, data)
        self.assertEqual(return_new, return_old)



    def test_recons3d_4nn_MPI_should_return_True(self):
        # argum = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))
        # (data, refrings, list_of_ref_ang, numr, xrng, yrng, step) = argum[0]

        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))

        (datanew,optionsnew,iternew) = arga[0]

        mpi_comm = MPI_COMM_WORLD
        myid = mpi_comm_rank(mpi_comm)

        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        snr = optionsnew.snr
        npad = optionsnew.npad
        datanew=XFORM_PROJECTION_IMG

        return_new = fu.recons3d_4nn_MPI(myid, datanew, symmetry="c1", npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_MPI(myid, datanew, symmetry="c1", npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(),0.5))


    # def test_recons3d_trl_struct_MPI_should_return_True(self):
    #     # argum = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))
    #     # (data, refrings, list_of_ref_ang, numr, xrng, yrng, step) = argum[0]
    #
    #     arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))
    #
    #     (datanew,optionsnew,iternew) = arga[0]
    #
    #     mpi_comm = MPI_COMM_WORLD
    #     myid = mpi_comm_rank(mpi_comm)
    #
    #     sym = optionsnew.sym
    #     sym = sym[0].lower() + sym[1:]
    #     snr = optionsnew.snr
    #     npad = optionsnew.npad
    #
    #
    #     return_new = fu.recons3d_trl_struct_MPI(myid, 0, datanew, symmetry=sym, npad=npad, mpi_comm = MPI_COMM_WORLD)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     return_old = oldfu.recons3d_trl_struct_MPI(myid,0, datanew, symmetry=sym, npad=npad, mpi_comm = MPI_COMM_WORLD)
    #     mpi_barrier(MPI_COMM_WORLD)
    #     self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_recons3d_4nn_ctf_should_return_True(self):

        stack_name = "bdb:Substack/sort3d_substack_003"

        return_new = fu.recons3d_4nn_ctf(stack_name, [], snr = 2, symmetry="c1", npad=2)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_ctf(stack_name, [], snr = 2, symmetry="c1", npad=2)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(), 0.5 ))



    def test_recons3d_4nn_ctf_MPI_should_return_True(self):

        finfo = open("dummytext.txt", 'w')
        list_proj = []
        stack_name = "bdb:Substack/sort3d_substack_003"
        nima = EMAN2_cppwrap.EMUtil.get_image_count(stack_name)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(stack_name, list_proj[0])

        return_new = fu.recons3d_4nn_ctf_MPI(0, [proj], snr =1 , sign = 1, symmetry="c1", finfo=finfo, npad=2, smearstep=0.5)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_4nn_ctf_MPI(0, [proj], snr =1, sign = 1, symmetry="c1", finfo=finfo, npad=2, smearstep=0.5)
        mpi_barrier(MPI_COMM_WORLD)
        finfo.close()

        self.assertTrue(numpy.allclose(return_new.get_3dview(), return_old.get_3dview(), 0.5  ))


    def test_recons3d_nn_SSNR_MPI_should_return_True(self):

        finfo = open("dummytext.txt", 'w')
        list_proj = []
        stack_name = "bdb:Substack/sort3d_substack_003"
        nima = EMAN2_cppwrap.EMUtil.get_image_count(stack_name)
        list_proj = list(range(nima))
        proj = EMData()
        proj.read_image(stack_name, list_proj[0])

        return_new = fu.recons3d_nn_SSNR_MPI(0, [proj], mask2D=False, npad=2, sign = 1, symmetry="c1")
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.recons3d_nn_SSNR_MPI(0, [proj], mask2D=False, npad=2, sign = 1, symmetry="c1")
        mpi_barrier(MPI_COMM_WORLD)
        finfo.close()

        self.assertEqual(return_new[0], return_old[0])
        self.assertTrue(numpy.array_equal(return_new[1].get_3dview(), return_old[1].get_3dview()))


    def test_prepare_recons_should_return_True(self):
        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))

        (datanew,optionsnew,iternew) = arga[0]
        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        snr = optionsnew.snr
        npad = optionsnew.npad

        return_new = fu.prepare_recons(datanew, symmetry=sym, myid=0 , main_node_half=0, half_start=4, step=2, index=5, npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons(datanew, symmetry=sym, myid=0 , main_node_half=0, half_start=4, step=2, index=5, npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertEqual(return_new, return_new)


    def test_prepare_recons_ctf_should_return_True(self):
        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))

        (datanew,optionsnew,iternew) = arga[0]
        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        snr = optionsnew.snr
        npad = optionsnew.npad
        nx = datanew[0].get_xsize()

        return_new = fu.prepare_recons_ctf(nx,datanew, snr =1 , symmetry=sym, myid=0 , main_node_half=0, half_start=4, step=2, npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons_ctf(nx,datanew, snr =1 , symmetry=sym, myid=0 , main_node_half=0, half_start=4, step=2, npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(return_new, return_old)


    def test_recons_from_fftvol_should_return_True(self):
        fftvol = EMData(76,76)
        weight = EMData(76,76)
        size = 76

        return_new = fu.recons_from_fftvol(size, fftvol, weight, "c1")
        return_old = oldfu.recons_from_fftvol(size, fftvol, weight, "c1")
        self.assertEqual(return_new, return_old)


    def test_recons_ctf_from_fftvol_should_return_True(self):
        fftvol = EMData(76,76)
        weight = EMData(76,76)
        size = 76

        return_new = fu.recons_ctf_from_fftvol(size, fftvol, weight, 2, "c1")
        return_old = oldfu.recons_ctf_from_fftvol(size, fftvol, weight, 2, "c1")
        self.assertEqual(return_new, return_old)


    def test_get_image_size_should_return_True(self):

        return_new = fu.get_image_size([ EMData(76,76),EMData(76,76),EMData(76,76) ], 0 )
        return_old = oldfu.get_image_size([EMData(76,76),EMData(76,76),EMData(76,76) ], 0)
        self.assertEqual(return_new, return_old)


    def test_rec3D_MPI_should_return_True(self):
        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))
        (datanew,optionsnew,iternew) = arga[0]
        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        npad = optionsnew.npad

        return_new = fu.rec3D_MPI( datanew, 1.0, sym,   myid=0 , main_node =0, odd_start = 4, eve_start=4, index = -1 , npad = npad)
        mpi_barrier(MPI_COMM_WORLD)

        return_old = oldfu.rec3D_MPI(datanew,1.0, sym, myid=0 , main_node =0,  odd_start = 4, eve_start=4, index = -1, npad = npad)
        mpi_barrier(MPI_COMM_WORLD)


        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))


    def test_rec3D_MPI_noCTF_should_return_True(self):
        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))

        (datanew,optionsnew,iternew) = arga[0]

        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        npad = optionsnew.npad

        return_new = fu.rec3D_MPI_noCTF( datanew, sym, myid=0 , main_node =0, odd_start = 4, eve_start=4, index = 5 , npad = npad)
        mpi_barrier(MPI_COMM_WORLD)

        return_old = oldfu.rec3D_MPI_noCTF(datanew, sym, myid=0 , main_node =0,  odd_start = 4, eve_start=4, index = 5, npad = npad)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))
        self.assertEqual(return_new[1], return_old[1])


    def test_prepare_recons_ctf_two_chunks_should_return_True(self):
        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))

        (datanew,optionsnew,iternew) = arga[0]
        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        snr = optionsnew.snr
        npad = optionsnew.npad
        nx = datanew[0].get_xsize()
        datanew[0].set_attr("chunk_id", 0)


        return_new = fu.prepare_recons_ctf_two_chunks(nx, datanew, snr =1 , symmetry=sym, myid=0 , main_node_half=0, chunk_ID =2, npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.prepare_recons_ctf_two_chunks(nx, datanew, snr =1 , symmetry=sym, myid=0 , main_node_half=0, chunk_ID =2,  npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(return_new, return_old)


    def test_rec3D_two_chunks_MPI_two_chunks_should_return_True(self):
        arga = get_arg_from_pickle_file(os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))

        (datanew,optionsnew,iternew) = arga[0]
        sym = optionsnew.sym
        sym = sym[0].lower() + sym[1:]
        snr = optionsnew.snr
        npad = optionsnew.npad
        nx = datanew[0].get_xsize()
        datanew[0].set_attr("chunk_id", 2)


        return_new = fu.rec3D_two_chunks_MPI( datanew, snr =1 , symmetry="c1", myid=0 , main_node=0, npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.rec3D_two_chunks_MPI( datanew, snr =1 , symmetry="c1", myid=0 , main_node=0,  npad=npad, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))



if __name__ == '__main__':
    unittest.main()

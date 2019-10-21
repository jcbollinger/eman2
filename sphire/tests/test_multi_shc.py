from __future__ import print_function
from __future__ import division
import unittest


from os import path
from mpi import *
import global_def


mpi_init(0, [])
global_def.BATCH = True
global_def.MPI = True

ABSOLUTE_PATH = path.dirname(path.realpath(__file__))



from test_module import  get_arg_from_pickle_file
from os import path
from numpy import array_equal as numpy_array_equal,allclose
from copy import deepcopy
from EMAN2_cppwrap import EMData
TOLERANCE = 0.00005

"""
There are some opened issues in:
-) orient_params
    -) if the symmetry_class is not cn it will lead to an 'UnboundLocalError'. beacuase it returns a value that not exists. it is a BUG
-) ali3d_multishc
    -) it crashes, using the pickle file values, because 'ZeroDivisionError' after 5 min .... why?? ask adnan 
-) ali3d_multishc_2
    -) using the pickle file values we do not have the same results 
-) multi_shc
    -) i cannot understand with which value I'd run it
-) mirror_and_reduce_dsym
    -) find a set of values that 'll produce changing in the 'params' inputvalue
"""

"""
pickle files stored under smb://billy.storage.mpi-dortmund.mpg.de/abt3/group/agraunser/transfer/Adnan/pickle files
"""


""" start: new in sphire 1.3"""
#from sphire.libpy_py3 import sp_alignment as new_sp_alignment
#from sphire.libpy import sp_alignment as old_sp_alignment
from sphire.libpy import sp_multi_shc as oldfu
from sphire.libpy_py3 import sp_multi_shc as fu

#todo: need data. I tried to use "pickle files/multi_shc/multi_shc.do_volume BUT it seems that this pickle file is not usable now!!!
# anyway the data are just a stack
# options should be a parser:
    #1) look wich value it need
    # fake it like https://stackoverflow.com/questions/18160078/how-do-you-write-tests-for-the-argparse-portion-of-a-python-module
class Test_volume_reconstruction(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.volume_reconstruction()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.volume_reconstruction()
        self.assertEqual(cm_new.exception.message, "volume_reconstruction() takes exactly 3 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_volume_reconstruction(self):
        return_old = oldfu.volume_reconstruction(data="", options="", mpi_comm=0)
        return_new = fu.volume_reconstruction(data="", options="", mpi_comm=0)
        self.assertTrue(numpy_array_equal(return_new, return_old))


class Test_volume_recsp(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.volume_recsp()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.volume_recsp()
        self.assertEqual(cm_new.exception.message, "volume_recsp() takes exactly 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_volume_recsp(self):
        return_old = oldfu.volume_recsp(data=0, options=0)
        return_new = fu.volume_recsp(data=0, options=0)
        self.assertTrue(numpy_array_equal(return_new, return_old))



class Test_proj_ali_incore_multi(unittest.TestCase):
    (data, refrings, list_of_ref_ang, numr, xrng, yrng, step) = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))[0]

    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError):
            fu.proj_ali_incore_multi()
            oldfu.proj_ali_incore_multi()

    def test_empty_input_image_returns_runtimeError_NotExistingObjectException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.proj_ali_incore_multi(data=EMData(), refrings=self.refrings,  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.proj_ali_incore_multi(data=EMData(), refrings=self.refrings,  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_process(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.proj_ali_incore_multi(data=None, refrings=self.refrings,  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.proj_ali_incore_multi(data=None, refrings=self.refrings,  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'get_xsize'")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_empty_input_image_refrings_returns_runtimeError_NotExistingObjectException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.proj_ali_incore_multi(data=self.data, refrings= [EMData(),EMData(),EMData()],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.proj_ali_incore_multi(data=self.data, refrings= [EMData(),EMData(),EMData()],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_None_input_image_refrings_crash_because_SIGSEGV(self):
        pass
        """
        with self.assertRaises(RuntimeError) as cm_new:
            fu.proj_ali_incore_multi(data=self.data, refrings= [None,None,None],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.proj_ali_incore_multi(data=self.data, refrings= [None,None,None],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        """

    def test_empty_list_numr_returns_IndexError_list_index_out_of_range(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.proj_ali_incore_multi(data=self.data, refrings=self.refrings,  numr=[], xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(IndexError) as cm_old:
            oldfu.proj_ali_incore_multi(data=self.data, refrings=self.refrings,  numr=[], xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_negative_nsoft(self):
        return_old = oldfu.proj_ali_incore_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step,an=1.0, nsoft=-1, finfo=None, sym="not_used")
        return_new = fu.proj_ali_incore_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step,an=1.0, nsoft=-1, finfo=None, sym="not_used")
        self.assertEqual(return_old,return_new)
        self.assertEqual(return_old, 4022.77539062)

    def test_positive_nsoft(self):
        return_old = oldfu.proj_ali_incore_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step,an=1.0, nsoft=3, finfo=None, sym="not_used")
        return_new = fu.proj_ali_incore_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step,an=1.0, nsoft=3, finfo=None, sym="not_used")
        self.assertEqual(return_old,return_new)
        self.assertEqual(return_old, 30659.9891357)


    def test_null_nsoft(self):
        return_old = oldfu.proj_ali_incore_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step,an=1.0, nsoft=0, finfo=None, sym="not_used")
        return_new = fu.proj_ali_incore_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step,an=1.0, nsoft=0, finfo=None, sym="not_used")
        self.assertEqual(return_old,return_new)
        self.assertEqual(return_old, 0)




class Test_shc_multi(unittest.TestCase):
    (data, refrings, list_of_ref_ang, numr, xrng, yrng, step) = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/alignment.shc"))[0]
    def test_wrong_number_params_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.shc_multi()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.shc_multi()
        self.assertEqual(cm_new.exception.message, "shc_multi() takes at least 9 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_empty_input_image_returns_runtimeError_NotExistingObjectException(self):
        with self.assertRaises(RuntimeError) as cm_new:
            fu.shc_multi(data=EMData(), refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="c1", finfo=None)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.shc_multi(data=EMData(), refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="c1", finfo=None)
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)


    def test_NoneType_as_img_returns_AttributeError_NoneType_obj_hasnot_attribute_get_Attr(self):
        with self.assertRaises(AttributeError) as cm_new:
            fu.shc_multi(data=None, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="c1", finfo=None)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.shc_multi(data=None, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="c1", finfo=None)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'get_attr'")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_empty_input_image_refrings_crash_because_SIGSEGV(self):
        pass
        """
        with self.assertRaises(RuntimeError) as cm_new:
            fu.proj_ali_incore_multi(data=self.data, refrings= [EMData(),EMData(),EMData()],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.proj_ali_incore_multi(data=self.data, refrings= [EMData(),EMData(),EMData()],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        """

    def test_Noneinput_image_refrings_crash_because_SIGSEGV(self):
        pass
        """
        with self.assertRaises(RuntimeError) as cm_new:
            fu.proj_ali_incore_multi(data=self.data, refrings= [None,None,None],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.proj_ali_incore_multi(data=self.data, refrings=[None,None,None],  numr=self.numr, xrng= 2.0, yrng=2.0, step=self.step, an = 1.0, nsoft = -1, finfo=None, sym="not_used")
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        """

    def test_shc_multi_negative_nsoft_c1(self):
        return_old = oldfu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="c1", finfo=None)
        return_new = fu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="c1", finfo=None)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal((1338.955078125, 48.97479809939048, 0, 2127), return_old))

    def test_shc_multi_negative_nsoft_d1(self):
        return_old = oldfu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="d1", finfo=None)
        return_new = fu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=-1, sym="d1", finfo=None)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal((1349.222412109375, 49.23334842295393, 0, 2127), return_old))

    def test_shc_multi_positive_nsoft_c1(self):
        return_old = oldfu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=1, sym="c1", finfo=None)
        return_new = fu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=1, sym="c1", finfo=None)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal((1065.65380859375, 55.0943717956543, 0, 1), return_old))

    def test_shc_multi_positive_nsoft_d1(self):
        return_old = oldfu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=1, sym="d1", finfo=None)
        return_new = fu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0, step=self.step, an=1.0, nsoft=1, sym="d1", finfo=None)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal((1225.196533203125, 56.53429412841797, 0, 1), return_old))

    def test_shc_multi_null_nsoft_c1(self):
        return_old = oldfu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0,step=self.step, an=1.0, nsoft=0, sym="c1", finfo=None)
        return_new = fu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0,step=self.step, an=1.0, nsoft=0, sym="c1", finfo=None)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal((0,0,0,0), return_old))

    def test_shc_multi_null_nsoft_d1(self):
        return_old = oldfu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0,step=self.step, an=1.0, nsoft=-1, sym="d1", finfo=None)
        return_new = fu.shc_multi(data=self.data, refrings=self.refrings, numr=self.numr, xrng=2.0, yrng=2.0,step=self.step, an=1.0, nsoft=-1, sym="d1", finfo=None)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal((0,0,0,0), return_old))



#todo: need data. I tried to use "pickle files/multi_shc/multi_shc.do_volume BUT it seems that this pickle file is not usable now!!!
class Test_ali3d_multishc_soft(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.ali3d_multishc_soft()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.ali3d_multishc_soft()
        self.assertEqual(cm_new.exception.message, "ali3d_multishc_soft() takes at least 3 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_ali3d_multishc_soft(self):
        return_old = oldfu.ali3d_multishc_soft(stack="", ref_vol="", ali3d_options="", mpi_comm = None, log = None, nsoft=2 )
        return_new = fu.ali3d_multishc_soft(stack="", ref_vol="", ali3d_options="", mpi_comm = None, log = None, nsoft=2 )
        self.assertTrue(numpy_array_equal(return_new, return_old))


#todo: need data. I tried to use "pickle files/multi_shc/multi_shc.do_volume BUT it seems that this pickle file is not usable now!!!
class Test_no_of_processors_restricted_by_data__do_volume(unittest.TestCase):
    def test_wrong_number_params_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.no_of_processors_restricted_by_data__do_volume()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.no_of_processors_restricted_by_data__do_volume()
        self.assertEqual(cm_new.exception.message, "angle_error() takes exactly 4 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_no_of_processors_restricted_by_data__do_volume(self):
        return_old = oldfu.no_of_processors_restricted_by_data__do_volume(projections="", ali3d_options="", iter="", mpi_comm=0)
        return_new = fu.no_of_processors_restricted_by_data__do_volume(projections="", ali3d_options="", iter="", mpi_comm=0)
        self.assertTrue(numpy_array_equal(return_new, return_old))

""" start: end in sphire 1.3"""

class Test_orient_params(unittest.TestCase):
    symmetry_class = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.orient_params"))[1]['symmetry_class']
    params = [[48.05124726579686, 69.05455778294466, 54.84375071663101, -0.0, -0.0], [56.845645784288195, 71.23236296721491, 55.88562174587844, -0.0, -0.0], [63.01890116107117, 69.45801186277717, 53.437500358889054, -0.0, -0.0], [48.78633889868564, 65.19273007904152, 53.437498529745426, -0.0, -0.0], [46.75583767738047, 51.55046889255816, 57.65625043396955, -0.0, -0.0], [46.50278311828407, 41.835647911839025, 53.43749893314299, -0.0, -0.0], [48.56280893139251, 34.12192773635157, 53.437499922907875, -0.0, -0.0], [1.6390995549009233, 28.968938083640264, 39.37499908518612, -0.0, -0.0], [63.02071009501989, 36.45465298624063, 42.18749889520927, -0.0, -0.0], [46.14631640064633, 30.170426205646173, 53.437500136526864, -0.0, -0.0], [48.73457100022452, 24.407400818343685, 57.245970159677995, -0.0, -0.0], [24.285651841215625, 23.7468002673281, 78.08666994272221, -0.0, -0.0], [29.074914300174996, 19.983170959991, 54.314147089510584, -0.0, -0.0], [47.16905631296311, 88.43670393467808, 330.46874988232844, -0.0, -0.0], [39.16575862912984, 88.22101165002283, 331.8749999522553, -0.0, -0.0], [43.31352598088864, 94.15394400739405, 331.8749990026355, -0.0, -0.0], [49.10023899407898, 86.54830776126175, 331.87499930500826, -0.0, -0.0], [47.324476288434795, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [44.86615332639434, 84.49372082999594, 333.28125059192405, -0.0, -0.0], [45.16831136248007, 88.38278300096842, 331.87500062955047, -0.0, -0.0], [55.17633498515568, 90.59291140785325, 335.0065014459757, -0.0, -0.0], [53.17099464029562, 88.59845821263497, 336.09375030281024, -0.0, -0.0], [53.1762286681647, 90.53900870573675, 334.68750144755785, -0.0, -0.0], [51.17613894898591, 90.48510647067248, 333.28125021856727, -0.0, -0.0], [48.88415187080702, 84.60201049035742, 334.68750059033323, -0.0, -0.0], [55.11046933868195, 86.71028837562088, 336.09375038714154, -0.0, -0.0], [52.46899705506567, 82.75812325603023, 336.093751277538, -0.0, -0.0], [47.89632014606357, 72.93185122418333, 334.41769431977843, -0.0, -0.0], [49.98783574020601, 72.98822685061805, 334.68750073407034, -0.0, -0.0], [48.92140033425832, 76.84079788798854, 336.0937506122811, -0.0, -0.0], [52.61008018496051, 78.88082097520068, 336.0937505060724, -0.0, -0.0], [46.494137433867024, 78.71598026532864, 336.0937498496585, -0.0, -0.0], [50.97486937567166, 76.8961447053418, 335.3546751805487, -0.0, -0.0], [46.86746409213612, 76.78543759795335, 334.68750012386204, -0.0, -0.0], [50.45291515823635, 82.70378622403362, 333.28124981968136, -0.0, -0.0], [48.53316998587681, 78.77093812533867, 331.87499939756674, -0.0, -0.0], [49.69657801406254, 80.74286516770074, 334.687499708489, -0.0, -0.0], [45.643481844441226, 80.63362576582266, 331.87500084965535, -0.0, -0.0], [44.45471337435643, 78.6610129856283, 336.0937505634877, -0.0, -0.0], [46.67058228902215, 74.8394901039006, 333.2812489244631, -0.0, -0.0], [48.43658849390172, 82.64944333618115, 336.09375007892686, -0.0, -0.0], [46.87524168887518, 84.54786820328374, 334.6875010020095, -0.0, -0.0], [54.9098141404055, 84.76440954646556, 334.68750002812897, -0.0, -0.0], [43.61645673997356, 80.57899271926958, 334.6874990854178, -0.0, -0.0], [49.16975228980925, 88.49062333203285, 335.46310507571707, -0.0, -0.0], [51.72265620273092, 80.7974718267131, 334.68750066339607, -0.0, -0.0], [49.17606777307816, 90.4312046611873, 336.09374980804887, -0.0, -0.0], [52.90143467892915, 84.71028123500598, 334.6875001034233, -0.0, -0.0], [50.892880829763016, 84.65614835620852, 336.09375026702287, -0.0, -0.0], [53.10716851670995, 86.65629754282332, 336.09375015787896, -0.0, -0.0], [41.17591158365914, 90.21560080610787, 334.68749922213186, -0.0, -0.0], [46.420015306458396, 82.59509271997537, 336.0937485969313, -0.0, -0.0], [49.20189825398518, 92.37228099046185, 333.28125076874466, -0.0, -0.0], [45.19854632928653, 92.26439265423947, 334.6875000197484, -0.0, -0.0], [51.17039546010827, 88.54454145531159, 336.093750062259, -0.0, -0.0], [43.088987971534095, 86.38629967541901, 334.68750031691764, -0.0, -0.0], [40.84742616086049, 84.38541040622479, 334.6874998082104, -0.0, -0.0], [40.36880488952832, 82.43200300659889, 337.50000011184864, -0.0, -0.0], [51.103757175597764, 86.60230420594348, 336.09375084881816, -0.0, -0.0], [42.75820051836939, 76.67467989542156, 334.6874998701815, -0.0, -0.0], [42.3861251208842, 82.48637268802736, 333.2812510342641, -0.0, -0.0], [42.3861251208842, 82.48637268802736, 336.2114254559933, -0.0, -0.0], [42.41490162526449, 78.60603439288184, 333.28124909234805, -0.0, -0.0], [42.52580179952784, 74.7277744064862, 333.28124971378963, -0.0, -0.0], [44.40319760103603, 82.54073648404015, 334.68749920659275, -0.0, -0.0], [40.702865295519956, 76.6192834130347, 335.53915428165584, -0.0, -0.0], [40.45258882904207, 74.67189357396991, 336.09375008883467, -0.0, -0.0], [40.37469112097813, 78.55104629335277, 336.0937503869335, -0.0, -0.0], [28.25949061934712, 82.10563477637265, 336.0937499708724, -0.0, -0.0], [29.043985279205202, 72.42370232519055, 336.73113959547123, -0.0, -0.0], [26.082035105586343, 78.16582030453586, 338.90625017219645, -0.0, -0.0], [24.038603453923244, 78.11074374424224, 336.0937486480204, -0.0, -0.0], [24.243008071919462, 76.17563661304364, 337.50000029000967, -0.0, -0.0], [16.681583692662457, 99.25713457197845, 337.4999998218846, -0.0, -0.0], [27.053643998645086, 85.95413292369146, 336.0937507100525, -0.0, -0.0], [33.162684909625995, 88.0592261186007, 337.5000004258998, -0.0, -0.0], [41.58911343938499, 80.52435094759053, 338.90625131664217, -0.0, -0.0], [36.5907766762643, 76.50844881763295, 337.4999995307441, -0.0, -0.0], [41.085001159089586, 86.33229055613145, 338.9062499026314, -0.0, -0.0], [41.1954915045381, 92.15651213279136, 337.35971066734726, -0.0, -0.0], [45.09285336696388, 86.44030549093121, 338.90624984254015, -0.0, -0.0], [47.09660234526038, 86.49430812756614, 333.28125066659646, -0.0, -0.0], [39.56144672210394, 80.46970147557718, 337.50000036634106, -0.0, -0.0], [42.41490162526449, 78.60603439288184, 338.9062504937533, -0.0, -0.0], [50.57181879143158, 78.8258853602938, 338.90624967050184, -0.0, -0.0], [43.711395212569215, 72.81905158053513, 338.9062505513591, -0.0, -0.0], [44.59846643020208, 74.78363935855029, 337.9769896804838, -0.0, -0.0], [44.81306744479457, 76.73006541230473, 337.50000022223264, -0.0, -0.0], [48.74215074093854, 74.89532694780435, 338.9062492344079, -0.0, -0.0], [47.67018815936095, 80.6882492049645, 338.9062497231663, -0.0, -0.0], [38.64705785836594, 76.56387207474546, 340.31249991677555, -0.0, -0.0], [45.80417431215312, 72.87545995140584, 337.50000005505785, -0.0, -0.0], [44.161015708616276, 70.89045303205856, 337.4999995948694, -0.0, -0.0], [50.19193017200831, 69.11226010712458, 336.09374935539626, -0.0, -0.0], [45.909737450248144, 68.99683228214761, 337.50000109946353, -0.0, -0.0]]
    refparams=[[44.4012739843086, 101.94434285155423, 57.656251087500095, -0.0, -0.0], [40.10876621464658, 103.76886166554078, 56.24999859695254, -0.0, -0.0], [32.81515546607329, 113.27272158296553, 53.43749983092755, -0.0, -0.0], [46.917502615482164, 119.4723683533925, 57.656250053386714, -0.0, -0.0], [49.05910923108033, 127.28877610352072, 57.656249990928984, -0.0, -0.0], [45.00866196853852, 138.81497350968104, 52.03124938409917, -0.0, -0.0], [43.40828386726827, 148.46324402758947, 54.84375067374663, -0.0, -0.0], [30.18279383601478, 150.04476642454256, 42.187501343093174, -0.0, -0.0], [28.46455033936273, 144.1851821369091, 43.81433052188976, -0.0, -0.0], [45.58260706254359, 152.39599796333576, 56.249999794000246, -0.0, -0.0], [56.46236230728803, 158.49501847825957, 63.281249026178614, -0.0, -0.0], [70.2736735548736, 156.9315836956123, 78.74999987999286, -0.0, -0.0], [61.954067511784075, 158.6425350493942, 54.84375053018613, -0.0, -0.0], [51.17613894898591, 90.48510647067248, 330.46874982484917, -0.0, -0.0], [50.892880829763016, 84.65614835620852, 330.46875020395197, -0.0, -0.0], [53.10716851670995, 86.65629754282332, 331.87500095509597, -0.0, -0.0], [47.16905631296311, 88.43670393467808, 333.28124891816464, -0.0, -0.0], [51.17039546010827, 88.54454145531159, 333.28125042797717, -0.0, -0.0], [45.16831136248007, 88.38278300096842, 333.2812494892223, -0.0, -0.0], [49.17606777307816, 90.4312046611873, 333.2812498507327, -0.0, -0.0], [45.08512532222889, 100.02254939056965, 333.5160508769778, -0.0, -0.0], [45.17596739278744, 90.32340216396611, 334.6874996009934, -0.0, -0.0], [41.1954915045381, 92.15651213279136, 334.7679740464975, -0.0, -0.0], [45.31893159619136, 94.20798789067081, 333.28124917530556, -0.0, -0.0], [41.59803048705899, 96.04803389535608, 334.6875002725168, -0.0, -0.0], [37.17589074800935, 90.10780021124744, 336.0937495292243, -0.0, -0.0], [40.13082054614313, 97.9487837779021, 335.5160523234348, -0.0, -0.0], [43.05413201572037, 99.96781920338643, 331.8750004859039, -0.0, -0.0], [41.0234784515755, 99.91309702143698, 336.09375081906273, -0.0, -0.0], [42.357014619404566, 101.88925668380978, 331.8749993840262, -0.0, -0.0], [43.05413201572037, 99.96781920338643, 334.68749894773794, -0.0, -0.0], [50.53656405394548, 102.1096729085617, 336.09374996528754, -0.0, -0.0], [46.74159955252878, 105.88774614975061, 335.7283328041299, -0.0, -0.0], [46.28923283750959, 103.93540552225147, 336.09374996989857, -0.0, -0.0], [47.116465288010374, 100.07728975643352, 333.28125031416545, -0.0, -0.0], [42.357014619404566, 101.88925668380978, 335.82693509541537, -0.0, -0.0], [48.82161253760111, 105.94379511749287, 334.68749979118763, -0.0, -0.0], [44.228582821840604, 103.8798779199092, 333.2812504166588, -0.0, -0.0], [42.15048915920923, 98.00321098875033, 334.687499658415, -0.0, -0.0], [47.116465288010374, 100.07728975643352, 334.69638122267565, -0.0, -0.0], [48.491046324392926, 102.05455194331047, 333.2812510174674, -0.0, -0.0], [49.14814731092724, 100.13203827338509, 334.6875007102493, -0.0, -0.0], [49.64485230063906, 96.26488481608796, 333.2812496237367, -0.0, -0.0], [44.170432722469855, 98.05764540192871, 336.0937501928436, -0.0, -0.0], [43.609425873086366, 96.10223830981946, 336.0937502946671, -0.0, -0.0], [44.170432722469855, 98.05764540192871, 334.68750097386874, -0.0, -0.0], [47.632836864107816, 96.210663711916, 334.6875010610388, -0.0, -0.0], [41.30825543100741, 94.09990346967666, 336.0937501433592, -0.0, -0.0], [47.200182853993454, 92.31833581634471, 334.6874998254447, -0.0, -0.0], [46.19064888032082, 98.11208578621446, 336.09374955424073, -0.0, -0.0], [49.20189825398518, 92.37228099046185, 334.68750071708075, -0.0, -0.0], [45.62102877534875, 96.15644831516599, 336.0937500819578, -0.0, -0.0], [47.324476288434795, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [47.17601196506416, 90.37730325905419, 334.6875006879113, -0.0, -0.0], [47.324476288434795, 94.26203540572942, 336.0937501285429, -0.0, -0.0], [52.582507437652254, 102.16480552927928, 336.09375007610106, -0.0, -0.0], [54.27426878916464, 98.32992548208689, 334.6875000566633, -0.0, -0.0], [51.65707835236421, 96.31911179365366, 337.4999992370875, -0.0, -0.0], [46.44595023042399, 101.99944190765075, 333.2812497173458, -0.0, -0.0], [52.47416992257348, 104.10206981604098, 334.6874992489246, -0.0, -0.0], [48.491046324392926, 102.05455194331047, 333.2812510174674, -0.0, -0.0], [49.14814731092724, 100.13203827338509, 334.9395453501846, -0.0, -0.0], [51.18017945427377, 100.18679745907073, 335.4682606383749, -0.0, -0.0], [52.47416992257348, 104.10206981604098, 337.46817000604676, -0.0, -0.0], [52.582507437652254, 102.16480552927928, 336.09375007610106, -0.0, -0.0], [53.21256367831589, 100.24156600446646, 335.177061967494, -0.0, -0.0], [52.98339478273627, 106.05593856487205, 337.4999990285479, -0.0, -0.0], [57.14752635973869, 106.1681465136048, 334.68749996309936, -0.0, -0.0], [63.379796596164226, 100.51554908757866, 336.0937506147592, -0.0, -0.0], [64.73973671841892, 108.31283157939123, 337.5000011532889, -0.0, -0.0], [67.76266544776647, 96.7531345346374, 338.74395800223857, -0.0, -0.0], [62.79250342910589, 104.38011485204639, 334.6875007546264, -0.0, -0.0], [71.01537880973311, 102.66151948900279, 337.50000019790514, -0.0, -0.0], [3.174987977762683, 89.1914720539308, 337.49999929370006, -0.0, -0.0], [1.1784268609936959, 91.07806513923873, 337.50000069390643, -0.0, -0.0], [59.21169731917283, 92.64203991483133, 338.9062500179199, -0.0, -0.0], [57.27838513113349, 100.35113013490589, 337.49999888322805, -0.0, -0.0], [54.62887064233311, 102.2199488043037, 337.50000112094676, -0.0, -0.0], [50.41202107730069, 104.04650135561747, 338.062530319494, -0.0, -0.0], [53.66951484404569, 96.37334432212357, 336.39749174576514, -0.0, -0.0], [51.335995429151524, 94.37014207243301, 338.90624914355226, -0.0, -0.0], [51.20369372374114, 92.42622836878581, 336.093749390464, -0.0, -0.0], [52.252945762867654, 98.27545417435223, 337.9944755313544, -0.0, -0.0], [52.252945762867654, 98.27545417435223, 338.9062495251767, -0.0, -0.0], [48.211137436906284, 98.16653499056041, 337.49999969956434, -0.0, -0.0], [50.23190398446434, 98.22099018564722, 338.9062497196227, -0.0, -0.0], [48.3503785990371, 103.99094749511158, 334.68750073986547, -0.0, -0.0], [44.66216541833097, 105.83171208560515, 338.58120741594587, -0.0, -0.0], [46.44595023042399, 101.99944190765075, 337.4999994056692, -0.0, -0.0], [50.90221176825793, 105.99985960998757, 337.4999994903521, -0.0, -0.0], [52.109863589162075, 107.97251110977513, 338.23471059822435, -0.0, -0.0], [50.90221176825793, 105.99985960998757, 338.90625042411614, -0.0, -0.0], [47.90534450850359, 107.85921828199395, 338.9062505807672, -0.0, -0.0], [52.01846080048625, 109.91011950963757, 334.6875000521202, -0.0, -0.0], [50.26719531842579, 113.74294931960685, 334.68750054434713, -0.0, -0.0]]
    indexes = list(range(95))

    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.orient_params()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.orient_params()
        self.assertEqual(cm_new.exception.message, "orient_params() takes at least 2 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_default_value_returns_AttributeError(self):
        with self.assertRaises(AttributeError)as cm_new:
            fu.orient_params(params=self.params, refparams=self.refparams, indexes= None,symmetry_class=None)
        with self.assertRaises(AttributeError)as cm_old:
            oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= None,symmetry_class=None)

        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'sym'")

    def test_symmetry_class_noneType_returns_AttributeError(self):
        with self.assertRaises(AttributeError)as cm_new:
            fu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=None)
        with self.assertRaises(AttributeError)as cm_old:
            oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=None)

        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'sym'")

    def test_indexes_empty_list(self):
        return_new = fu.orient_params(params=self.params, refparams=self.refparams, indexes= [],symmetry_class=self.symmetry_class)
        return_old = oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= [],symmetry_class=self.symmetry_class)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal(return_new, [[48.05124726579686, 69.05455778294466, 54.84375071663101, -0.0, -0.0], [56.845645784288195, 71.23236296721491, 55.88562174587844, -0.0, -0.0], [63.01890116107117, 69.45801186277717, 53.437500358889054, -0.0, -0.0], [48.78633889868564, 65.19273007904152, 53.437498529745426, -0.0, -0.0], [46.75583767738047, 51.55046889255816, 57.65625043396955, -0.0, -0.0], [46.50278311828407, 41.835647911839025, 53.43749893314299, -0.0, -0.0], [48.56280893139251, 34.12192773635157, 53.437499922907875, -0.0, -0.0], [1.6390995549009233, 28.968938083640264, 39.37499908518612, -0.0, -0.0], [63.02071009501989, 36.45465298624063, 42.18749889520927, -0.0, -0.0], [46.14631640064633, 30.170426205646173, 53.437500136526864, -0.0, -0.0], [48.73457100022452, 24.407400818343685, 57.245970159677995, -0.0, -0.0], [24.285651841215625, 23.7468002673281, 78.08666994272221, -0.0, -0.0], [29.074914300174996, 19.983170959991, 54.314147089510584, -0.0, -0.0], [47.16905631296311, 88.43670393467808, 330.46874988232844, -0.0, -0.0], [39.16575862912984, 88.22101165002283, 331.8749999522553, -0.0, -0.0], [43.31352598088864, 94.15394400739405, 331.8749990026355, -0.0, -0.0], [49.10023899407898, 86.54830776126175, 331.87499930500826, -0.0, -0.0], [47.324476288434795, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [44.86615332639434, 84.49372082999594, 333.28125059192405, -0.0, -0.0], [45.16831136248007, 88.38278300096842, 331.87500062955047, -0.0, -0.0], [55.17633498515568, 90.59291140785325, 335.0065014459757, -0.0, -0.0], [53.17099464029562, 88.59845821263497, 336.09375030281024, -0.0, -0.0], [53.1762286681647, 90.53900870573675, 334.68750144755785, -0.0, -0.0], [51.17613894898591, 90.48510647067248, 333.28125021856727, -0.0, -0.0], [48.88415187080702, 84.60201049035742, 334.68750059033323, -0.0, -0.0], [55.11046933868195, 86.71028837562088, 336.09375038714154, -0.0, -0.0], [52.46899705506567, 82.75812325603023, 336.093751277538, -0.0, -0.0], [47.89632014606357, 72.93185122418333, 334.41769431977843, -0.0, -0.0], [49.98783574020601, 72.98822685061805, 334.68750073407034, -0.0, -0.0], [48.92140033425832, 76.84079788798854, 336.0937506122811, -0.0, -0.0], [52.61008018496051, 78.88082097520068, 336.0937505060724, -0.0, -0.0], [46.494137433867024, 78.71598026532864, 336.0937498496585, -0.0, -0.0], [50.97486937567166, 76.8961447053418, 335.3546751805487, -0.0, -0.0], [46.86746409213612, 76.78543759795335, 334.68750012386204, -0.0, -0.0], [50.45291515823635, 82.70378622403362, 333.28124981968136, -0.0, -0.0], [48.53316998587681, 78.77093812533867, 331.87499939756674, -0.0, -0.0], [49.69657801406254, 80.74286516770074, 334.687499708489, -0.0, -0.0], [45.643481844441226, 80.63362576582266, 331.87500084965535, -0.0, -0.0], [44.45471337435643, 78.6610129856283, 336.0937505634877, -0.0, -0.0], [46.67058228902215, 74.8394901039006, 333.2812489244631, -0.0, -0.0], [48.43658849390172, 82.64944333618115, 336.09375007892686, -0.0, -0.0], [46.87524168887518, 84.54786820328374, 334.6875010020095, -0.0, -0.0], [54.9098141404055, 84.76440954646556, 334.68750002812897, -0.0, -0.0], [43.61645673997356, 80.57899271926958, 334.6874990854178, -0.0, -0.0], [49.16975228980925, 88.49062333203285, 335.46310507571707, -0.0, -0.0], [51.72265620273092, 80.7974718267131, 334.68750066339607, -0.0, -0.0], [49.17606777307816, 90.4312046611873, 336.09374980804887, -0.0, -0.0], [52.90143467892915, 84.71028123500598, 334.6875001034233, -0.0, -0.0], [50.892880829763016, 84.65614835620852, 336.09375026702287, -0.0, -0.0], [53.10716851670995, 86.65629754282332, 336.09375015787896, -0.0, -0.0], [41.17591158365914, 90.21560080610787, 334.68749922213186, -0.0, -0.0], [46.420015306458396, 82.59509271997537, 336.0937485969313, -0.0, -0.0], [49.20189825398518, 92.37228099046185, 333.28125076874466, -0.0, -0.0], [45.19854632928653, 92.26439265423947, 334.6875000197484, -0.0, -0.0], [51.17039546010827, 88.54454145531159, 336.093750062259, -0.0, -0.0], [43.088987971534095, 86.38629967541901, 334.68750031691764, -0.0, -0.0], [40.84742616086049, 84.38541040622479, 334.6874998082104, -0.0, -0.0], [40.36880488952832, 82.43200300659889, 337.50000011184864, -0.0, -0.0], [51.103757175597764, 86.60230420594348, 336.09375084881816, -0.0, -0.0], [42.75820051836939, 76.67467989542156, 334.6874998701815, -0.0, -0.0], [42.3861251208842, 82.48637268802736, 333.2812510342641, -0.0, -0.0], [42.3861251208842, 82.48637268802736, 336.2114254559933, -0.0, -0.0], [42.41490162526449, 78.60603439288184, 333.28124909234805, -0.0, -0.0], [42.52580179952784, 74.7277744064862, 333.28124971378963, -0.0, -0.0], [44.40319760103603, 82.54073648404015, 334.68749920659275, -0.0, -0.0], [40.702865295519956, 76.6192834130347, 335.53915428165584, -0.0, -0.0], [40.45258882904207, 74.67189357396991, 336.09375008883467, -0.0, -0.0], [40.37469112097813, 78.55104629335277, 336.0937503869335, -0.0, -0.0], [28.25949061934712, 82.10563477637265, 336.0937499708724, -0.0, -0.0], [29.043985279205202, 72.42370232519055, 336.73113959547123, -0.0, -0.0], [26.082035105586343, 78.16582030453586, 338.90625017219645, -0.0, -0.0], [24.038603453923244, 78.11074374424224, 336.0937486480204, -0.0, -0.0], [24.243008071919462, 76.17563661304364, 337.50000029000967, -0.0, -0.0], [16.681583692662457, 99.25713457197845, 337.4999998218846, -0.0, -0.0], [27.053643998645086, 85.95413292369146, 336.0937507100525, -0.0, -0.0], [33.162684909625995, 88.0592261186007, 337.5000004258998, -0.0, -0.0], [41.58911343938499, 80.52435094759053, 338.90625131664217, -0.0, -0.0], [36.5907766762643, 76.50844881763295, 337.4999995307441, -0.0, -0.0], [41.085001159089586, 86.33229055613145, 338.9062499026314, -0.0, -0.0], [41.1954915045381, 92.15651213279136, 337.35971066734726, -0.0, -0.0], [45.09285336696388, 86.44030549093121, 338.90624984254015, -0.0, -0.0], [47.09660234526038, 86.49430812756614, 333.28125066659646, -0.0, -0.0], [39.56144672210394, 80.46970147557718, 337.50000036634106, -0.0, -0.0], [42.41490162526449, 78.60603439288184, 338.9062504937533, -0.0, -0.0], [50.57181879143158, 78.8258853602938, 338.90624967050184, -0.0, -0.0], [43.711395212569215, 72.81905158053513, 338.9062505513591, -0.0, -0.0], [44.59846643020208, 74.78363935855029, 337.9769896804838, -0.0, -0.0], [44.81306744479457, 76.73006541230473, 337.50000022223264, -0.0, -0.0], [48.74215074093854, 74.89532694780435, 338.9062492344079, -0.0, -0.0], [47.67018815936095, 80.6882492049645, 338.9062497231663, -0.0, -0.0], [38.64705785836594, 76.56387207474546, 340.31249991677555, -0.0, -0.0], [45.80417431215312, 72.87545995140584, 337.50000005505785, -0.0, -0.0], [44.161015708616276, 70.89045303205856, 337.4999995948694, -0.0, -0.0], [50.19193017200831, 69.11226010712458, 336.09374935539626, -0.0, -0.0], [45.909737450248144, 68.99683228214761, 337.50000109946353, -0.0, -0.0]]))

    def test_params_empty_list_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.orient_params(params=[], refparams=self.refparams, indexes= self.indexes,symmetry_class=self.symmetry_class)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.orient_params(params=[], refparams=self.refparams, indexes= self.indexes,symmetry_class=self.symmetry_class)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_refparams_empty_list_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.orient_params(params=self.params, refparams=[], indexes= self.indexes,symmetry_class=self.symmetry_class)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.orient_params(params=self.params, refparams=[], indexes= self.indexes,symmetry_class=self.symmetry_class)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_refparams_and_params_empty_list_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.orient_params(params=[], refparams=[], indexes=self.indexes,symmetry_class=self.symmetry_class)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.orient_params(params=[], refparams=[], indexes=self.indexes,symmetry_class=self.symmetry_class)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_refparams_and_params_indexes_empty_list_returns_IndexError(self):
        return_new = fu.orient_params(params=[], refparams=[], indexes=[],symmetry_class=self.symmetry_class)
        return_old = oldfu.orient_params(params=[], refparams=[], indexes=[],symmetry_class=self.symmetry_class)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal(return_new, []))

    def test_indexes_None(self):
        return_new = fu.orient_params(params=self.params, refparams=self.refparams, indexes= None,symmetry_class=self.symmetry_class)
        return_old = oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= None,symmetry_class=self.symmetry_class)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal(return_new, [[48.040515511180516, 69.05455778294466, 54.84375071663101, -0.0, -0.0], [56.834914029671836, 71.23236296721491, 55.88562174587844, -0.0, -0.0], [63.00816940645484, 69.45801186277717, 53.437500358889054, -0.0, -0.0], [48.775607144069298, 65.19273007904152, 53.437498529745426, -0.0, -0.0], [46.745105922764125, 51.55046889255816, 57.65625043396955, -0.0, -0.0], [46.492051363667727, 41.835647911839025, 53.43749893314299, -0.0, -0.0], [48.552077176776166, 34.12192773635157, 53.437499922907875, -0.0, -0.0], [1.6283678002845789, 28.968938083640264, 39.37499908518612, -0.0, -0.0], [63.009978340403563, 36.45465298624063, 42.18749889520927, -0.0, -0.0], [46.135584646029983, 30.170426205646173, 53.437500136526864, -0.0, -0.0], [48.723839245608175, 24.407400818343685, 57.245970159677995, -0.0, -0.0], [24.27492008659928, 23.7468002673281, 78.08666994272221, -0.0, -0.0], [29.064182545558651, 19.983170959991, 54.314147089510584, -0.0, -0.0], [47.158324558346763, 88.43670393467808, 330.46874988232844, -0.0, -0.0], [39.155026874513496, 88.22101165002283, 331.8749999522553, -0.0, -0.0], [43.302794226272297, 94.15394400739405, 331.8749990026355, -0.0, -0.0], [49.089507239462634, 86.54830776126175, 331.87499930500826, -0.0, -0.0], [47.313744533818451, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [44.855421571777995, 84.49372082999594, 333.28125059192405, -0.0, -0.0], [45.157579607863724, 88.38278300096842, 331.87500062955047, -0.0, -0.0], [55.165603230539332, 90.59291140785325, 335.0065014459757, -0.0, -0.0], [53.160262885679273, 88.59845821263497, 336.09375030281024, -0.0, -0.0], [53.165496913548353, 90.53900870573675, 334.68750144755785, -0.0, -0.0], [51.165407194369564, 90.48510647067248, 333.28125021856727, -0.0, -0.0], [48.873420116190673, 84.60201049035742, 334.68750059033323, -0.0, -0.0], [55.099737584065608, 86.71028837562088, 336.09375038714154, -0.0, -0.0], [52.458265300449327, 82.75812325603023, 336.093751277538, -0.0, -0.0], [47.885588391447229, 72.93185122418333, 334.41769431977843, -0.0, -0.0], [49.977103985589665, 72.98822685061805, 334.68750073407034, -0.0, -0.0], [48.910668579641978, 76.84079788798854, 336.0937506122811, -0.0, -0.0], [52.599348430344165, 78.88082097520068, 336.0937505060724, -0.0, -0.0], [46.48340567925068, 78.71598026532864, 336.0937498496585, -0.0, -0.0], [50.964137621055315, 76.8961447053418, 335.3546751805487, -0.0, -0.0], [46.856732337519773, 76.78543759795335, 334.68750012386204, -0.0, -0.0], [50.442183403620007, 82.70378622403362, 333.28124981968136, -0.0, -0.0], [48.522438231260466, 78.77093812533867, 331.87499939756674, -0.0, -0.0], [49.685846259446194, 80.74286516770074, 334.687499708489, -0.0, -0.0], [45.632750089824881, 80.63362576582266, 331.87500084965535, -0.0, -0.0], [44.443981619740086, 78.6610129856283, 336.0937505634877, -0.0, -0.0], [46.659850534405805, 74.8394901039006, 333.2812489244631, -0.0, -0.0], [48.425856739285379, 82.64944333618115, 336.09375007892686, -0.0, -0.0], [46.864509934258834, 84.54786820328374, 334.6875010020095, -0.0, -0.0], [54.899082385789157, 84.76440954646556, 334.68750002812897, -0.0, -0.0], [43.605724985357213, 80.57899271926958, 334.6874990854178, -0.0, -0.0], [49.159020535192909, 88.49062333203285, 335.46310507571707, -0.0, -0.0], [51.711924448114573, 80.7974718267131, 334.68750066339607, -0.0, -0.0], [49.165336018461815, 90.4312046611873, 336.09374980804887, -0.0, -0.0], [52.890702924312805, 84.71028123500598, 334.6875001034233, -0.0, -0.0], [50.882149075146671, 84.65614835620852, 336.09375026702287, -0.0, -0.0], [53.096436762093603, 86.65629754282332, 336.09375015787896, -0.0, -0.0], [41.165179829042799, 90.21560080610787, 334.68749922213186, -0.0, -0.0], [46.409283551842051, 82.59509271997537, 336.0937485969313, -0.0, -0.0], [49.191166499368833, 92.37228099046185, 333.28125076874466, -0.0, -0.0], [45.187814574670185, 92.26439265423947, 334.6875000197484, -0.0, -0.0], [51.159663705491923, 88.54454145531159, 336.093750062259, -0.0, -0.0], [43.078256216917751, 86.38629967541901, 334.68750031691764, -0.0, -0.0], [40.836694406244149, 84.38541040622479, 334.6874998082104, -0.0, -0.0], [40.358073134911976, 82.43200300659889, 337.50000011184864, -0.0, -0.0], [51.093025420981419, 86.60230420594348, 336.09375084881816, -0.0, -0.0], [42.747468763753048, 76.67467989542156, 334.6874998701815, -0.0, -0.0], [42.375393366267858, 82.48637268802736, 333.2812510342641, -0.0, -0.0], [42.375393366267858, 82.48637268802736, 336.2114254559933, -0.0, -0.0], [42.404169870648147, 78.60603439288184, 333.28124909234805, -0.0, -0.0], [42.515070044911496, 74.7277744064862, 333.28124971378963, -0.0, -0.0], [44.392465846419682, 82.54073648404015, 334.68749920659275, -0.0, -0.0], [40.692133540903612, 76.6192834130347, 335.53915428165584, -0.0, -0.0], [40.441857074425727, 74.67189357396991, 336.09375008883467, -0.0, -0.0], [40.363959366361783, 78.55104629335277, 336.0937503869335, -0.0, -0.0], [28.248758864730775, 82.10563477637265, 336.0937499708724, -0.0, -0.0], [29.033253524588858, 72.42370232519055, 336.73113959547123, -0.0, -0.0], [26.071303350969998, 78.16582030453586, 338.90625017219645, -0.0, -0.0], [24.0278716993069, 78.11074374424224, 336.0937486480204, -0.0, -0.0], [24.232276317303118, 76.17563661304364, 337.50000029000967, -0.0, -0.0], [16.670851938046113, 99.25713457197845, 337.4999998218846, -0.0, -0.0], [27.042912244028741, 85.95413292369146, 336.0937507100525, -0.0, -0.0], [33.15195315500965, 88.0592261186007, 337.5000004258998, -0.0, -0.0], [41.578381684768644, 80.52435094759053, 338.90625131664217, -0.0, -0.0], [36.580044921647954, 76.50844881763295, 337.4999995307441, -0.0, -0.0], [41.074269404473242, 86.33229055613145, 338.9062499026314, -0.0, -0.0], [41.184759749921753, 92.15651213279136, 337.35971066734726, -0.0, -0.0], [45.082121612347535, 86.44030549093121, 338.90624984254015, -0.0, -0.0], [47.085870590644035, 86.49430812756614, 333.28125066659646, -0.0, -0.0], [39.550714967487593, 80.46970147557718, 337.50000036634106, -0.0, -0.0], [42.404169870648147, 78.60603439288184, 338.9062504937533, -0.0, -0.0], [50.561087036815238, 78.8258853602938, 338.90624967050184, -0.0, -0.0], [43.700663457952871, 72.81905158053513, 338.9062505513591, -0.0, -0.0], [44.587734675585736, 74.78363935855029, 337.9769896804838, -0.0, -0.0], [44.802335690178225, 76.73006541230473, 337.50000022223264, -0.0, -0.0], [48.731418986322197, 74.89532694780435, 338.9062492344079, -0.0, -0.0], [47.659456404744603, 80.6882492049645, 338.9062497231663, -0.0, -0.0], [38.636326103749596, 76.56387207474546, 340.31249991677555, -0.0, -0.0], [45.793442557536778, 72.87545995140584, 337.50000005505785, -0.0, -0.0], [44.150283953999931, 70.89045303205856, 337.4999995948694, -0.0, -0.0], [50.181198417391968, 69.11226010712458, 336.09374935539626, -0.0, -0.0], [45.8990056956318, 68.99683228214761, 337.50000109946353, -0.0, -0.0]]))

    def test_pickle_values(self):
        return_new = fu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=self.symmetry_class)
        return_old = oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=self.symmetry_class)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal(return_new, [[48.040515511180516, 69.05455778294466, 54.84375071663101, -0.0, -0.0], [56.834914029671836, 71.23236296721491, 55.88562174587844, -0.0, -0.0], [63.00816940645484, 69.45801186277717, 53.437500358889054, -0.0, -0.0], [48.775607144069298, 65.19273007904152, 53.437498529745426, -0.0, -0.0], [46.745105922764125, 51.55046889255816, 57.65625043396955, -0.0, -0.0], [46.492051363667727, 41.835647911839025, 53.43749893314299, -0.0, -0.0], [48.552077176776166, 34.12192773635157, 53.437499922907875, -0.0, -0.0], [1.6283678002845789, 28.968938083640264, 39.37499908518612, -0.0, -0.0], [63.009978340403563, 36.45465298624063, 42.18749889520927, -0.0, -0.0], [46.135584646029983, 30.170426205646173, 53.437500136526864, -0.0, -0.0], [48.723839245608175, 24.407400818343685, 57.245970159677995, -0.0, -0.0], [24.27492008659928, 23.7468002673281, 78.08666994272221, -0.0, -0.0], [29.064182545558651, 19.983170959991, 54.314147089510584, -0.0, -0.0], [47.158324558346763, 88.43670393467808, 330.46874988232844, -0.0, -0.0], [39.155026874513496, 88.22101165002283, 331.8749999522553, -0.0, -0.0], [43.302794226272297, 94.15394400739405, 331.8749990026355, -0.0, -0.0], [49.089507239462634, 86.54830776126175, 331.87499930500826, -0.0, -0.0], [47.313744533818451, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [44.855421571777995, 84.49372082999594, 333.28125059192405, -0.0, -0.0], [45.157579607863724, 88.38278300096842, 331.87500062955047, -0.0, -0.0], [55.165603230539332, 90.59291140785325, 335.0065014459757, -0.0, -0.0], [53.160262885679273, 88.59845821263497, 336.09375030281024, -0.0, -0.0], [53.165496913548353, 90.53900870573675, 334.68750144755785, -0.0, -0.0], [51.165407194369564, 90.48510647067248, 333.28125021856727, -0.0, -0.0], [48.873420116190673, 84.60201049035742, 334.68750059033323, -0.0, -0.0], [55.099737584065608, 86.71028837562088, 336.09375038714154, -0.0, -0.0], [52.458265300449327, 82.75812325603023, 336.093751277538, -0.0, -0.0], [47.885588391447229, 72.93185122418333, 334.41769431977843, -0.0, -0.0], [49.977103985589665, 72.98822685061805, 334.68750073407034, -0.0, -0.0], [48.910668579641978, 76.84079788798854, 336.0937506122811, -0.0, -0.0], [52.599348430344165, 78.88082097520068, 336.0937505060724, -0.0, -0.0], [46.48340567925068, 78.71598026532864, 336.0937498496585, -0.0, -0.0], [50.964137621055315, 76.8961447053418, 335.3546751805487, -0.0, -0.0], [46.856732337519773, 76.78543759795335, 334.68750012386204, -0.0, -0.0], [50.442183403620007, 82.70378622403362, 333.28124981968136, -0.0, -0.0], [48.522438231260466, 78.77093812533867, 331.87499939756674, -0.0, -0.0], [49.685846259446194, 80.74286516770074, 334.687499708489, -0.0, -0.0], [45.632750089824881, 80.63362576582266, 331.87500084965535, -0.0, -0.0], [44.443981619740086, 78.6610129856283, 336.0937505634877, -0.0, -0.0], [46.659850534405805, 74.8394901039006, 333.2812489244631, -0.0, -0.0], [48.425856739285379, 82.64944333618115, 336.09375007892686, -0.0, -0.0], [46.864509934258834, 84.54786820328374, 334.6875010020095, -0.0, -0.0], [54.899082385789157, 84.76440954646556, 334.68750002812897, -0.0, -0.0], [43.605724985357213, 80.57899271926958, 334.6874990854178, -0.0, -0.0], [49.159020535192909, 88.49062333203285, 335.46310507571707, -0.0, -0.0], [51.711924448114573, 80.7974718267131, 334.68750066339607, -0.0, -0.0], [49.165336018461815, 90.4312046611873, 336.09374980804887, -0.0, -0.0], [52.890702924312805, 84.71028123500598, 334.6875001034233, -0.0, -0.0], [50.882149075146671, 84.65614835620852, 336.09375026702287, -0.0, -0.0], [53.096436762093603, 86.65629754282332, 336.09375015787896, -0.0, -0.0], [41.165179829042799, 90.21560080610787, 334.68749922213186, -0.0, -0.0], [46.409283551842051, 82.59509271997537, 336.0937485969313, -0.0, -0.0], [49.191166499368833, 92.37228099046185, 333.28125076874466, -0.0, -0.0], [45.187814574670185, 92.26439265423947, 334.6875000197484, -0.0, -0.0], [51.159663705491923, 88.54454145531159, 336.093750062259, -0.0, -0.0], [43.078256216917751, 86.38629967541901, 334.68750031691764, -0.0, -0.0], [40.836694406244149, 84.38541040622479, 334.6874998082104, -0.0, -0.0], [40.358073134911976, 82.43200300659889, 337.50000011184864, -0.0, -0.0], [51.093025420981419, 86.60230420594348, 336.09375084881816, -0.0, -0.0], [42.747468763753048, 76.67467989542156, 334.6874998701815, -0.0, -0.0], [42.375393366267858, 82.48637268802736, 333.2812510342641, -0.0, -0.0], [42.375393366267858, 82.48637268802736, 336.2114254559933, -0.0, -0.0], [42.404169870648147, 78.60603439288184, 333.28124909234805, -0.0, -0.0], [42.515070044911496, 74.7277744064862, 333.28124971378963, -0.0, -0.0], [44.392465846419682, 82.54073648404015, 334.68749920659275, -0.0, -0.0], [40.692133540903612, 76.6192834130347, 335.53915428165584, -0.0, -0.0], [40.441857074425727, 74.67189357396991, 336.09375008883467, -0.0, -0.0], [40.363959366361783, 78.55104629335277, 336.0937503869335, -0.0, -0.0], [28.248758864730775, 82.10563477637265, 336.0937499708724, -0.0, -0.0], [29.033253524588858, 72.42370232519055, 336.73113959547123, -0.0, -0.0], [26.071303350969998, 78.16582030453586, 338.90625017219645, -0.0, -0.0], [24.0278716993069, 78.11074374424224, 336.0937486480204, -0.0, -0.0], [24.232276317303118, 76.17563661304364, 337.50000029000967, -0.0, -0.0], [16.670851938046113, 99.25713457197845, 337.4999998218846, -0.0, -0.0], [27.042912244028741, 85.95413292369146, 336.0937507100525, -0.0, -0.0], [33.15195315500965, 88.0592261186007, 337.5000004258998, -0.0, -0.0], [41.578381684768644, 80.52435094759053, 338.90625131664217, -0.0, -0.0], [36.580044921647954, 76.50844881763295, 337.4999995307441, -0.0, -0.0], [41.074269404473242, 86.33229055613145, 338.9062499026314, -0.0, -0.0], [41.184759749921753, 92.15651213279136, 337.35971066734726, -0.0, -0.0], [45.082121612347535, 86.44030549093121, 338.90624984254015, -0.0, -0.0], [47.085870590644035, 86.49430812756614, 333.28125066659646, -0.0, -0.0], [39.550714967487593, 80.46970147557718, 337.50000036634106, -0.0, -0.0], [42.404169870648147, 78.60603439288184, 338.9062504937533, -0.0, -0.0], [50.561087036815238, 78.8258853602938, 338.90624967050184, -0.0, -0.0], [43.700663457952871, 72.81905158053513, 338.9062505513591, -0.0, -0.0], [44.587734675585736, 74.78363935855029, 337.9769896804838, -0.0, -0.0], [44.802335690178225, 76.73006541230473, 337.50000022223264, -0.0, -0.0], [48.731418986322197, 74.89532694780435, 338.9062492344079, -0.0, -0.0], [47.659456404744603, 80.6882492049645, 338.9062497231663, -0.0, -0.0], [38.636326103749596, 76.56387207474546, 340.31249991677555, -0.0, -0.0], [45.793442557536778, 72.87545995140584, 337.50000005505785, -0.0, -0.0], [44.150283953999931, 70.89045303205856, 337.4999995948694, -0.0, -0.0], [50.181198417391968, 69.11226010712458, 336.09374935539626, -0.0, -0.0], [45.8990056956318, 68.99683228214761, 337.50000109946353, -0.0, -0.0]]))

    def test_symmetryClass_with_msym_1(self):
        symmetry_class = deepcopy(self.symmetry_class)
        symmetry_class.nsym=1
        return_new = fu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=symmetry_class)
        return_old = oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=symmetry_class)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal(return_new, [[44.997331480628951, 111.62609759230635, 54.535845033063595, -0.0, -0.0], [36.180586663137198, 109.3963513877251, 55.475364001251705, -0.0, -0.0], [29.968966956801182, 111.12521269418717, 52.952520506367023, -0.0, -0.0], [44.235178924343359, 115.48411891310595, 53.110764611212005, -0.0, -0.0], [46.185811769289479, 129.13624579981209, 57.307167331330277, -0.0, -0.0], [46.352795236193728, 138.85206064873572, 53.030620672078499, -0.0, -0.0], [44.156725902035426, 146.55529967605494, 52.907132118091681, -0.0, -0.0], [92.069131451412986, 151.70645676803528, 39.999187073513156, -0.0, -0.0], [29.516338873695108, 144.12677740427947, 41.415453895721384, -0.0, -0.0], [46.549639513184374, 150.51857912978323, 52.902203317402268, -0.0, -0.0], [43.758002868446432, 156.26835649610217, 56.514609431517442, -0.0, -0.0], [68.903936313925314, 156.99174069184116, 78.118514807316956, -0.0, -0.0], [63.944680026971767, 160.7537733439986, 54.164716250298056, -0.0, -0.0], [45.982123465872895, 92.24853483443178, 330.19282358018518, -0.0, -0.0], [53.987966969786193, 92.495952364766183, 331.69720974739539, -0.0, -0.0], [49.862047527696667, 86.548335660604835, 331.64556961938501, -0.0, -0.0], [44.040328282382788, 94.127220974427217, 331.5755732591972, -0.0, -0.0], [45.854892560779362, 86.422521075929126, 333.00313557326831, -0.0, -0.0], [48.268763438731661, 96.202010405906648, 333.03183453463475, -0.0, -0.0], [47.983412839519495, 92.311667020830171, 331.62317363509771, -0.0, -0.0], [37.985776432931551, 90.047267396978157, 334.63802083657981, -0.0, -0.0], [39.978926972585064, 92.054185545886369, 335.74767418113777, -0.0, -0.0], [39.985412223056741, 90.113638827109895, 334.3415874249842, -0.0, -0.0], [41.9850799080439, 90.17921478804206, 332.95832368342752, -0.0, -0.0], [44.246442037577907, 96.0746144465664, 334.38972900963142, -0.0, -0.0], [38.026677716427123, 93.930233476021712, 335.72513917079391, -0.0, -0.0], [40.646426169797088, 97.898607904425006, 335.75268408933857, -0.0, -0.0], [45.174696008213381, 107.74963311095252, 334.11909062524296, -0.0, -0.0], [43.075794648611641, 107.68238515448063, 334.36309514971219, -0.0, -0.0], [44.167674453114159, 103.83552685126705, 335.78835028785363, -0.0, -0.0], [40.481459347395585, 101.77500494483064, 335.74700593275395, -0.0, -0.0], [46.611076801052029, 101.97235197775002, 335.8201759908502, -0.0, -0.0], [42.108787406547698, 103.7690898201916, 335.02460028821906, -0.0, -0.0], [46.22717855753865, 103.90110900912285, 334.4071871574144, -0.0, -0.0], [42.665272580934996, 97.96444001022931, 332.96367239577057, -0.0, -0.0], [44.567352265680398, 101.90741241463117, 331.57663036239506, -0.0, -0.0], [43.412063169372601, 99.929432285259196, 334.37718683418223, -0.0, -0.0], [47.472776364911638, 100.05863353018677, 331.61361047268724, -0.0, -0.0], [48.655319224898996, 102.03642921975191, 335.84533559891975, -0.0, -0.0], [46.414819416817551, 105.84796706057003, 333.0008489389603, -0.0, -0.0], [44.684468751377345, 98.029451078681618, 335.80006625547026, -0.0, -0.0], [46.257470510505648, 96.138731836494713, 334.41373492089292, -0.0, -0.0], [38.21496490612261, 95.877357259750269, 334.32006988677392, -0.0, -0.0], [49.503777333392705, 100.1219544172702, 334.45109690332583, -0.0, -0.0], [43.980945382917206, 92.184568906305131, 335.16341832431027, -0.0, -0.0], [41.382352717462396, 99.863577565648669, 334.35331410225695, -0.0, -0.0], [43.984778463306974, 90.243981085940902, 335.79420372597986, -0.0, -0.0], [40.225190621597463, 95.943913033495306, 334.34286795250114, -0.0, -0.0], [42.235683403601215, 96.009670143725884, 335.77234707379739, -0.0, -0.0], [40.031041346052533, 93.996695877284779, 335.74778232664778, -0.0, -0.0], [51.983951604706128, 90.494701053977508, 334.48483100113395, -0.0, -0.0], [46.704013239019112, 98.093629387166629, 335.82433287952324, -0.0, -0.0], [43.969109686333674, 88.302796236666779, 332.98127144647287, -0.0, -0.0], [47.97024421844187, 88.429961991366525, 334.43541689621009, -0.0, -0.0], [41.97988276588854, 92.119781345559673, 335.77067060956858, -0.0, -0.0], [50.055677831802882, 94.316810692498422, 334.46059718406894, -0.0, -0.0], [52.292148276702974, 96.326006517394291, 334.48769804253106, -0.0, -0.0], [52.764713233751451, 98.281038859682539, 337.30533511213793, -0.0, -0.0], [42.035593082999164, 94.062363001708761, 335.77085409706103, -0.0, -0.0], [50.348000047131436, 104.02965099029795, 334.45847171993773, -0.0, -0.0], [50.744137587659736, 98.219434125611258, 333.06137523610823, -0.0, -0.0], [50.744137587659736, 98.219434125611258, 335.99154965783748, -0.0, -0.0], [50.700069827408448, 102.09963385494744, 333.05832505683122, -0.0, -0.0], [50.573104292088217, 105.97743932771408, 333.05309898228563, -0.0, -0.0], [48.723900636858254, 98.156960248246818, 334.44269622731281, -0.0, -0.0], [52.40931094090103, 104.09258556335145, 335.33625825685419, -0.0, -0.0], [52.653286957694398, 106.0408188739068, 335.89222589044067, -0.0, -0.0], [52.745333583127199, 102.16195306986924, 335.8966173430681, -0.0, -0.0], [64.895231176734598, 98.631994480494271, 336.05454252001061, -0.0, -0.0], [64.100514693369036, 108.31332220594511, 336.67967423210922, -0.0, -0.0], [67.076002098264382, 102.57275084351012, 338.89527693207492, -0.0, -0.0], [69.12515457325118, 102.62773939198348, 336.10976737590863, -0.0, -0.0], [68.920641374971879, 104.56289767927639, 337.51342837013624, -0.0, -0.0], [76.461569895609102, 81.47328454166059, 337.61129404334679, -0.0, -0.0], [66.104918114494495, 94.784150277873493, 336.07043733943021, -0.0, -0.0], [59.993575624020167, 92.672388947192928, 337.39822279530586, -0.0, -0.0], [51.535201732517756, 100.18440644050195, 338.69514734718859, -0.0, -0.0], [56.533706258563349, 104.21573605514497, 337.35018011584259, -0.0, -0.0], [52.061174996749529, 94.378307902241417, 338.7041266393669, -0.0, -0.0], [51.971245744185978, 88.553732600323286, 337.15674262439893, -0.0, -0.0], [48.050373090592693, 94.254456790135009, 338.65484983667375, -0.0, -0.0], [46.045257438724391, 94.191256466436172, 333.00566656033618, -0.0, -0.0], [53.56704976265685, 100.2459783042264, 337.31447317793135, -0.0, -0.0], [50.700069827408448, 102.09963385494744, 338.68332645823642, -0.0, -0.0], [42.524143444682437, 101.84162495459336, 338.58348019673628, -0.0, -0.0], [49.37493239482577, 107.88143936362715, 338.66049496516649, -0.0, -0.0], [48.493612050474923, 105.91315198109635, 337.72253562894082, -0.0, -0.0], [48.287286144495859, 103.96582081132897, 337.24515542576307, -0.0, -0.0], [44.336733452283383, 105.78189733818151, 338.60029429730457, -0.0, -0.0], [45.442204226183172, 99.99445604673997, 338.62021225568787, -0.0, -0.0], [54.471215122372591, 104.1546179870038, 340.13601809136225, -0.0, -0.0], [47.274410805605363, 107.8159887919234, 337.22762453361327, -0.0, -0.0], [48.914581585292083, 109.80815502175571, 337.2455707498757, -0.0, -0.0], [42.84719669745153, 111.55717804776639, 335.75885728301432, -0.0, -0.0], [47.148521231924008, 111.69409038896191, 337.21954955062154, -0.0, -0.0]]))

    def test_symmetryClass_not_cn_UnboundLocalError_BUG(self):
        symmetry_class = deepcopy(self.symmetry_class)
        symmetry_class.sym='d'
        with self.assertRaises(UnboundLocalError) as cm_new:
            fu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=symmetry_class)
        with self.assertRaises(UnboundLocalError) as cm_old:
            oldfu.orient_params(params=self.params, refparams=self.refparams, indexes= self.indexes,symmetry_class=symmetry_class)
        self.assertEqual(cm_new.exception.message, "local variable 'out' referenced before assignment")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)



class Test_find_common_subset(unittest.TestCase):
    (projs,target_threshold,minimal_subset_size,symmetry_class) = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.find_common_subset"))[0]

    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.find_common_subset()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.find_common_subset()
        self.assertEqual(cm_new.exception.message, "find_common_subset() takes at least 1 argument (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_emptyProjslist_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.find_common_subset([], target_threshold = self.target_threshold ,minimal_subset_size = self.minimal_subset_size ,symmetry_class =self.symmetry_class)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.find_common_subset([], target_threshold = self.target_threshold ,minimal_subset_size = self.minimal_subset_size ,symmetry_class =self.symmetry_class)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_None_symClass_returns_AttributeError(self):
        with self.assertRaises(AttributeError)as cm_new:
            fu.find_common_subset(projs= self.projs, target_threshold = self.target_threshold ,minimal_subset_size = self.minimal_subset_size ,symmetry_class =None)
        with self.assertRaises(AttributeError)as cm_old:
            oldfu.find_common_subset(projs= self.projs, target_threshold = self.target_threshold ,minimal_subset_size = self.minimal_subset_size ,symmetry_class =None)
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        self.assertEqual(cm_new.exception.message, "'NoneType' object has no attribute 'sym'")

    def test_pickle_file_value(self):
        return_new = fu.find_common_subset(projs= self.projs, target_threshold = self.target_threshold ,minimal_subset_size = self.minimal_subset_size ,symmetry_class =self.symmetry_class)
        return_old = oldfu.find_common_subset(projs= self.projs, target_threshold = self.target_threshold ,minimal_subset_size = self.minimal_subset_size ,symmetry_class =self.symmetry_class)
        self.assertTrue(numpy_array_equal(return_new, return_old))
        self.assertTrue(numpy_array_equal(return_new, ([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 34, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 58, 73, 79, 80, 81], [33.083686646519595, 36.472626714637158, 52.779699911264331, 54.308960491892407, 75.76833530050483, 96.987820484661043, 114.41564496426142, 123.06489103225256, 111.45977728123334, 122.2263087036868, 134.19776481502279, 134.45300283256174, 140.42079694497301, 4.509490307518508, 12.243664500985185, 12.336865776878311, 2.6919223319511714, 6.8950934219761244, 3.9015878865374245, 4.5100745962163709, 13.763317883569021, 8.1678349415420843, 12.075109501245077, 6.9261627056824535, 13.555986327351984, 18.233580075639114, 19.54032049214312, 27.454668944439348, 28.342813866592991, 25.8782497207613, 23.124275076822101, 23.73737447560271, 29.290783127201504, 27.155778169054589, 17.686369691777809, 23.915488250753942, 25.215452871777341, 23.287963788666559, 19.476288707302761, 25.241857896641587, 19.405214134908313, 15.749296548615286, 12.640122455449381, 17.487705393123964, 9.4148673527942623, 18.824218447816506, 5.9783614060025592, 14.899774091936449, 8.499195036983231, 13.367236517230051, 8.3186369116453882, 13.584136662355895, 2.6539794858956793, 2.7406823048498721, 6.883115581414792, 18.390494193349198, 19.3396426075393, 17.885130915806251, 16.075253742808751, 29.069760289120822, 20.491229966131176, 18.890530145543988, 23.277154790206044, 30.981259444626001, 21.246360944415969, 26.691263222176008, 33.738501255105973, 32.224370124383938, 39.518767831843256, 50.206527467495391, 35.413463584107497, 42.125848973965667, 36.413246435144913, 16.792508523015474, 26.349991090017451, 26.452283591428344, 25.241006944375123, 31.325793732252155, 19.989095564825305, 13.142080268698084, 10.095979276373182, 7.2199309077165577, 21.841927678037706, 21.973980480278769, 19.481438026383731, 26.209641295013533, 29.443410480625975, 29.101977160414727, 27.198426802833072, 25.514403748961499, 34.105207936143749, 33.50510292864692, 37.152319653868567, 40.837445380259943, 44.947792091395634], [[[44.4012739843086, 101.94434285155423, 57.656251087500095, -0.0, -0.0], [40.10876621464658, 103.76886166554078, 56.24999859695254, -0.0, -0.0], [32.81515546607329, 113.27272158296553, 53.43749983092755, -0.0, -0.0], [46.917502615482164, 119.4723683533925, 57.656250053386714, -0.0, -0.0], [49.05910923108033, 127.28877610352072, 57.656249990928984, -0.0, -0.0], [45.00866196853852, 138.81497350968104, 52.03124938409917, -0.0, -0.0], [43.40828386726827, 148.46324402758947, 54.84375067374663, -0.0, -0.0], [30.18279383601478, 150.04476642454256, 42.187501343093174, -0.0, -0.0], [28.46455033936273, 144.1851821369091, 43.81433052188976, -0.0, -0.0], [45.58260706254359, 152.39599796333576, 56.249999794000246, -0.0, -0.0], [56.46236230728803, 158.49501847825957, 63.281249026178614, -0.0, -0.0], [70.2736735548736, 156.9315836956123, 78.74999987999286, -0.0, -0.0], [61.954067511784075, 158.6425350493942, 54.84375053018613, -0.0, -0.0], [51.17613894898591, 90.48510647067248, 330.46874982484917, -0.0, -0.0], [50.892880829763016, 84.65614835620852, 330.46875020395197, -0.0, -0.0], [53.10716851670995, 86.65629754282332, 331.87500095509597, -0.0, -0.0], [47.16905631296311, 88.43670393467808, 333.28124891816464, -0.0, -0.0], [51.17039546010827, 88.54454145531159, 333.28125042797717, -0.0, -0.0], [45.16831136248007, 88.38278300096842, 333.2812494892223, -0.0, -0.0], [49.17606777307816, 90.4312046611873, 333.2812498507327, -0.0, -0.0], [45.08512532222889, 100.02254939056965, 333.5160508769778, -0.0, -0.0], [45.17596739278744, 90.32340216396611, 334.6874996009934, -0.0, -0.0], [41.1954915045381, 92.15651213279136, 334.7679740464975, -0.0, -0.0], [45.31893159619136, 94.20798789067081, 333.28124917530556, -0.0, -0.0], [41.59803048705899, 96.04803389535608, 334.6875002725168, -0.0, -0.0], [37.17589074800935, 90.10780021124744, 336.0937495292243, -0.0, -0.0], [40.13082054614313, 97.9487837779021, 335.5160523234348, -0.0, -0.0], [43.05413201572037, 99.96781920338643, 331.8750004859039, -0.0, -0.0], [41.0234784515755, 99.91309702143698, 336.09375081906273, -0.0, -0.0], [42.357014619404566, 101.88925668380978, 331.8749993840262, -0.0, -0.0], [43.05413201572037, 99.96781920338643, 334.68749894773794, -0.0, -0.0], [50.53656405394548, 102.1096729085617, 336.09374996528754, -0.0, -0.0], [46.74159955252878, 105.88774614975061, 335.7283328041299, -0.0, -0.0], [46.28923283750959, 103.93540552225147, 336.09374996989857, -0.0, -0.0], [47.116465288010374, 100.07728975643352, 333.28125031416545, -0.0, -0.0], [42.357014619404566, 101.88925668380978, 335.82693509541537, -0.0, -0.0], [48.82161253760111, 105.94379511749287, 334.68749979118763, -0.0, -0.0], [44.228582821840604, 103.8798779199092, 333.2812504166588, -0.0, -0.0], [42.15048915920923, 98.00321098875033, 334.687499658415, -0.0, -0.0], [47.116465288010374, 100.07728975643352, 334.69638122267565, -0.0, -0.0], [48.491046324392926, 102.05455194331047, 333.2812510174674, -0.0, -0.0], [49.14814731092724, 100.13203827338509, 334.6875007102493, -0.0, -0.0], [49.64485230063906, 96.26488481608796, 333.2812496237367, -0.0, -0.0], [44.170432722469855, 98.05764540192871, 336.0937501928436, -0.0, -0.0], [43.609425873086366, 96.10223830981946, 336.0937502946671, -0.0, -0.0], [44.170432722469855, 98.05764540192871, 334.68750097386874, -0.0, -0.0], [47.632836864107816, 96.210663711916, 334.6875010610388, -0.0, -0.0], [41.30825543100741, 94.09990346967666, 336.0937501433592, -0.0, -0.0], [47.200182853993454, 92.31833581634471, 334.6874998254447, -0.0, -0.0], [46.19064888032082, 98.11208578621446, 336.09374955424073, -0.0, -0.0], [49.20189825398518, 92.37228099046185, 334.68750071708075, -0.0, -0.0], [45.62102877534875, 96.15644831516599, 336.0937500819578, -0.0, -0.0], [47.324476288434795, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [47.17601196506416, 90.37730325905419, 334.6875006879113, -0.0, -0.0], [47.324476288434795, 94.26203540572942, 336.0937501285429, -0.0, -0.0], [52.582507437652254, 102.16480552927928, 336.09375007610106, -0.0, -0.0], [54.27426878916464, 98.32992548208689, 334.6875000566633, -0.0, -0.0], [51.65707835236421, 96.31911179365366, 337.4999992370875, -0.0, -0.0], [46.44595023042399, 101.99944190765075, 333.2812497173458, -0.0, -0.0], [52.47416992257348, 104.10206981604098, 334.6874992489246, -0.0, -0.0], [48.491046324392926, 102.05455194331047, 333.2812510174674, -0.0, -0.0], [49.14814731092724, 100.13203827338509, 334.9395453501846, -0.0, -0.0], [51.18017945427377, 100.18679745907073, 335.4682606383749, -0.0, -0.0], [52.47416992257348, 104.10206981604098, 337.46817000604676, -0.0, -0.0], [52.582507437652254, 102.16480552927928, 336.09375007610106, -0.0, -0.0], [53.21256367831589, 100.24156600446646, 335.177061967494, -0.0, -0.0], [52.98339478273627, 106.05593856487205, 337.4999990285479, -0.0, -0.0], [57.14752635973869, 106.1681465136048, 334.68749996309936, -0.0, -0.0], [63.379796596164226, 100.51554908757866, 336.0937506147592, -0.0, -0.0], [64.73973671841892, 108.31283157939123, 337.5000011532889, -0.0, -0.0], [67.76266544776647, 96.7531345346374, 338.74395800223857, -0.0, -0.0], [62.79250342910589, 104.38011485204639, 334.6875007546264, -0.0, -0.0], [71.01537880973311, 102.66151948900279, 337.50000019790514, -0.0, -0.0], [3.174987977762683, 89.1914720539308, 337.49999929370006, -0.0, -0.0], [1.1784268609936959, 91.07806513923873, 337.50000069390643, -0.0, -0.0], [59.21169731917283, 92.64203991483133, 338.9062500179199, -0.0, -0.0], [57.27838513113349, 100.35113013490589, 337.49999888322805, -0.0, -0.0], [54.62887064233311, 102.2199488043037, 337.50000112094676, -0.0, -0.0], [50.41202107730069, 104.04650135561747, 338.062530319494, -0.0, -0.0], [53.66951484404569, 96.37334432212357, 336.39749174576514, -0.0, -0.0], [51.335995429151524, 94.37014207243301, 338.90624914355226, -0.0, -0.0], [51.20369372374114, 92.42622836878581, 336.093749390464, -0.0, -0.0], [52.252945762867654, 98.27545417435223, 337.9944755313544, -0.0, -0.0], [52.252945762867654, 98.27545417435223, 338.9062495251767, -0.0, -0.0], [48.211137436906284, 98.16653499056041, 337.49999969956434, -0.0, -0.0], [50.23190398446434, 98.22099018564722, 338.9062497196227, -0.0, -0.0], [48.3503785990371, 103.99094749511158, 334.68750073986547, -0.0, -0.0], [44.66216541833097, 105.83171208560515, 338.58120741594587, -0.0, -0.0], [46.44595023042399, 101.99944190765075, 337.4999994056692, -0.0, -0.0], [50.90221176825793, 105.99985960998757, 337.4999994903521, -0.0, -0.0], [52.109863589162075, 107.97251110977513, 338.23471059822435, -0.0, -0.0], [50.90221176825793, 105.99985960998757, 338.90625042411614, -0.0, -0.0], [47.90534450850359, 107.85921828199395, 338.9062505807672, -0.0, -0.0], [52.01846080048625, 109.91011950963757, 334.6875000521202, -0.0, -0.0], [50.26719531842579, 113.74294931960685, 334.68750054434713, -0.0, -0.0]], [[48.040515511180516, 69.05455778294466, 54.84375071663101, -0.0, -0.0], [56.834914029671836, 71.23236296721491, 55.88562174587844, -0.0, -0.0], [63.00816940645484, 69.45801186277717, 53.437500358889054, -0.0, -0.0], [48.775607144069298, 65.19273007904152, 53.437498529745426, -0.0, -0.0], [46.745105922764125, 51.55046889255816, 57.65625043396955, -0.0, -0.0], [46.492051363667727, 41.835647911839025, 53.43749893314299, -0.0, -0.0], [48.552077176776166, 34.12192773635157, 53.437499922907875, -0.0, -0.0], [1.6283678002845789, 28.968938083640264, 39.37499908518612, -0.0, -0.0], [63.009978340403563, 36.45465298624063, 42.18749889520927, -0.0, -0.0], [46.135584646029983, 30.170426205646173, 53.437500136526864, -0.0, -0.0], [48.723839245608175, 24.407400818343685, 57.245970159677995, -0.0, -0.0], [24.27492008659928, 23.7468002673281, 78.08666994272221, -0.0, -0.0], [29.064182545558651, 19.983170959991, 54.314147089510584, -0.0, -0.0], [47.158324558346763, 88.43670393467808, 330.46874988232844, -0.0, -0.0], [39.155026874513496, 88.22101165002283, 331.8749999522553, -0.0, -0.0], [43.302794226272297, 94.15394400739405, 331.8749990026355, -0.0, -0.0], [49.089507239462634, 86.54830776126175, 331.87499930500826, -0.0, -0.0], [47.313744533818451, 94.26203540572942, 333.28124915861184, -0.0, -0.0], [44.855421571777995, 84.49372082999594, 333.28125059192405, -0.0, -0.0], [45.157579607863724, 88.38278300096842, 331.87500062955047, -0.0, -0.0], [55.165603230539332, 90.59291140785325, 335.0065014459757, -0.0, -0.0], [53.160262885679273, 88.59845821263497, 336.09375030281024, -0.0, -0.0], [53.165496913548353, 90.53900870573675, 334.68750144755785, -0.0, -0.0], [51.165407194369564, 90.48510647067248, 333.28125021856727, -0.0, -0.0], [48.873420116190673, 84.60201049035742, 334.68750059033323, -0.0, -0.0], [55.099737584065608, 86.71028837562088, 336.09375038714154, -0.0, -0.0], [52.458265300449327, 82.75812325603023, 336.093751277538, -0.0, -0.0], [47.885588391447229, 72.93185122418333, 334.41769431977843, -0.0, -0.0], [49.977103985589665, 72.98822685061805, 334.68750073407034, -0.0, -0.0], [48.910668579641978, 76.84079788798854, 336.0937506122811, -0.0, -0.0], [52.599348430344165, 78.88082097520068, 336.0937505060724, -0.0, -0.0], [46.48340567925068, 78.71598026532864, 336.0937498496585, -0.0, -0.0], [50.964137621055315, 76.8961447053418, 335.3546751805487, -0.0, -0.0], [46.856732337519773, 76.78543759795335, 334.68750012386204, -0.0, -0.0], [50.442183403620007, 82.70378622403362, 333.28124981968136, -0.0, -0.0], [48.522438231260466, 78.77093812533867, 331.87499939756674, -0.0, -0.0], [49.685846259446194, 80.74286516770074, 334.687499708489, -0.0, -0.0], [45.632750089824881, 80.63362576582266, 331.87500084965535, -0.0, -0.0], [44.443981619740086, 78.6610129856283, 336.0937505634877, -0.0, -0.0], [46.659850534405805, 74.8394901039006, 333.2812489244631, -0.0, -0.0], [48.425856739285379, 82.64944333618115, 336.09375007892686, -0.0, -0.0], [46.864509934258834, 84.54786820328374, 334.6875010020095, -0.0, -0.0], [54.899082385789157, 84.76440954646556, 334.68750002812897, -0.0, -0.0], [43.605724985357213, 80.57899271926958, 334.6874990854178, -0.0, -0.0], [49.159020535192909, 88.49062333203285, 335.46310507571707, -0.0, -0.0], [51.711924448114573, 80.7974718267131, 334.68750066339607, -0.0, -0.0], [49.165336018461815, 90.4312046611873, 336.09374980804887, -0.0, -0.0], [52.890702924312805, 84.71028123500598, 334.6875001034233, -0.0, -0.0], [50.882149075146671, 84.65614835620852, 336.09375026702287, -0.0, -0.0], [53.096436762093603, 86.65629754282332, 336.09375015787896, -0.0, -0.0], [41.165179829042799, 90.21560080610787, 334.68749922213186, -0.0, -0.0], [46.409283551842051, 82.59509271997537, 336.0937485969313, -0.0, -0.0], [49.191166499368833, 92.37228099046185, 333.28125076874466, -0.0, -0.0], [45.187814574670185, 92.26439265423947, 334.6875000197484, -0.0, -0.0], [51.159663705491923, 88.54454145531159, 336.093750062259, -0.0, -0.0], [43.078256216917751, 86.38629967541901, 334.68750031691764, -0.0, -0.0], [40.836694406244149, 84.38541040622479, 334.6874998082104, -0.0, -0.0], [40.358073134911976, 82.43200300659889, 337.50000011184864, -0.0, -0.0], [51.093025420981419, 86.60230420594348, 336.09375084881816, -0.0, -0.0], [42.747468763753048, 76.67467989542156, 334.6874998701815, -0.0, -0.0], [42.375393366267858, 82.48637268802736, 333.2812510342641, -0.0, -0.0], [42.375393366267858, 82.48637268802736, 336.2114254559933, -0.0, -0.0], [42.404169870648147, 78.60603439288184, 333.28124909234805, -0.0, -0.0], [42.515070044911496, 74.7277744064862, 333.28124971378963, -0.0, -0.0], [44.392465846419682, 82.54073648404015, 334.68749920659275, -0.0, -0.0], [40.692133540903612, 76.6192834130347, 335.53915428165584, -0.0, -0.0], [40.441857074425727, 74.67189357396991, 336.09375008883467, -0.0, -0.0], [40.363959366361783, 78.55104629335277, 336.0937503869335, -0.0, -0.0], [28.248758864730775, 82.10563477637265, 336.0937499708724, -0.0, -0.0], [29.033253524588858, 72.42370232519055, 336.73113959547123, -0.0, -0.0], [26.071303350969998, 78.16582030453586, 338.90625017219645, -0.0, -0.0], [24.0278716993069, 78.11074374424224, 336.0937486480204, -0.0, -0.0], [24.232276317303118, 76.17563661304364, 337.50000029000967, -0.0, -0.0], [16.670851938046113, 99.25713457197845, 337.4999998218846, -0.0, -0.0], [27.042912244028741, 85.95413292369146, 336.0937507100525, -0.0, -0.0], [33.15195315500965, 88.0592261186007, 337.5000004258998, -0.0, -0.0], [41.578381684768644, 80.52435094759053, 338.90625131664217, -0.0, -0.0], [36.580044921647954, 76.50844881763295, 337.4999995307441, -0.0, -0.0], [41.074269404473242, 86.33229055613145, 338.9062499026314, -0.0, -0.0], [41.184759749921753, 92.15651213279136, 337.35971066734726, -0.0, -0.0], [45.082121612347535, 86.44030549093121, 338.90624984254015, -0.0, -0.0], [47.085870590644035, 86.49430812756614, 333.28125066659646, -0.0, -0.0], [39.550714967487593, 80.46970147557718, 337.50000036634106, -0.0, -0.0], [42.404169870648147, 78.60603439288184, 338.9062504937533, -0.0, -0.0], [50.561087036815238, 78.8258853602938, 338.90624967050184, -0.0, -0.0], [43.700663457952871, 72.81905158053513, 338.9062505513591, -0.0, -0.0], [44.587734675585736, 74.78363935855029, 337.9769896804838, -0.0, -0.0], [44.802335690178225, 76.73006541230473, 337.50000022223264, -0.0, -0.0], [48.731418986322197, 74.89532694780435, 338.9062492344079, -0.0, -0.0], [47.659456404744603, 80.6882492049645, 338.9062497231663, -0.0, -0.0], [38.636326103749596, 76.56387207474546, 340.31249991677555, -0.0, -0.0], [45.793442557536778, 72.87545995140584, 337.50000005505785, -0.0, -0.0], [44.150283953999931, 70.89045303205856, 337.4999995948694, -0.0, -0.0], [50.181198417391968, 69.11226010712458, 336.09374935539626, -0.0, -0.0], [45.8990056956318, 68.99683228214761, 337.50000109946353, -0.0, -0.0]]])))




class Test_ali3d_multishc(unittest.TestCase):
    (stack, ref_vol, ali3d_options, symmetry_class) =  get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.ali3d_multishc"))[0]
    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.ali3d_multishc()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.ali3d_multishc()
        self.assertEqual(cm_new.exception.message, "ali3d_multishc() takes at least 4 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)


    @unittest.skip("ZeroDivisionError on the pickle file value")
    def test_pickleFile(self):


        mpi_barrier(MPI_COMM_WORLD)

        return_new = fu.ali3d_multishc(stack= self.stack, ref_vol = self.ref_vol, ali3d_options=self.ali3d_options, symmetry_class = self.symmetry_class, number_of_runs=1)

        mpi_barrier(MPI_COMM_WORLD)

        return_old = oldfu.ali3d_multishc(stack= self.stack, ref_vol = self.ref_vol, ali3d_options=self.ali3d_options, symmetry_class = self.symmetry_class, number_of_runs=1)

        mpi_barrier(MPI_COMM_WORLD)

        # mpi_finalize()

        if (return_old is not None and return_new is not None):
            self.assertTrue(return_new, return_old)




class Test_ali3d_multishc_2(unittest.TestCase):
    (stack, ref_vol, ali3d_options, symmetry_class) =  get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.ali3d_multishc_2"))[0]
    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.ali3d_multishc_2()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.ali3d_multishc_2()
        self.assertEqual(cm_new.exception.message, "ali3d_multishc_2() takes at least 4 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    @unittest.skip("PROBLEM: not have the same results")
    def test_pickleFile(self):
        mpi_barrier(MPI_COMM_WORLD)
        return_new = fu.ali3d_multishc_2(stack= self.stack, ref_vol = self.ref_vol, ali3d_options=self.ali3d_options, symmetry_class = self.symmetry_class, mpi_comm = None, log = None)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.ali3d_multishc_2(stack= self.stack, ref_vol = self.ref_vol, ali3d_options=self.ali3d_options, symmetry_class = self.symmetry_class,  mpi_comm = None, log = None)
        mpi_barrier(MPI_COMM_WORLD)

        self.assertTrue(allclose(return_new[0], return_old[0], atol=TOLERANCE))
        self.assertTrue(allclose(return_old[1].get_3dview(), return_new[1].get_3dview(), atol=TOLERANCE))
        self.assertTrue(allclose(return_new[2], return_old[2], atol=TOLERANCE))
        self.assertTrue(allclose(return_new[3], return_old[3], atol=TOLERANCE))



class Test_multi_shc(unittest.TestCase):
    (projs, target_threshold, minimal_subset_size, symmetry_class) = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.multi_shc"))[0]
    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.multi_shc()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.multi_shc()
        self.assertEqual(cm_new.exception.message, "multi_shc() takes at least 5 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)



class Test_mirror_and_reduce_dsym(unittest.TestCase):
    (projs, target_threshold, minimal_subset_size, symmetry_class)  = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.find_common_subset"))[0]
    subset = list(range( len(projs[0]) ))

    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.mirror_and_reduce_dsym()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.mirror_and_reduce_dsym()
        self.assertEqual(cm_new.exception.message, "mirror_and_reduce_dsym() takes exactly 3 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_No_params_returns_IndexError(self):
        with self.assertRaises(IndexError) as cm_new:
            fu.mirror_and_reduce_dsym(params = [], indexes=self.subset, symmetry_class=self.symmetry_class)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.mirror_and_reduce_dsym(params =[], indexes=self.subset, symmetry_class=self.symmetry_class)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_No_subset(self):
        new_copy=deepcopy(self.projs)
        old_copy = deepcopy(self.projs)
        return_new = fu.mirror_and_reduce_dsym(params = self.projs, indexes=[], symmetry_class=self.symmetry_class)

        return_old = oldfu.mirror_and_reduce_dsym(params =self.projs, indexes=[], symmetry_class=self.symmetry_class)
        self.assertTrue(numpy_array_equal(new_copy,old_copy))
        self.assertTrue(numpy_array_equal(new_copy, self.projs))
        self.assertEqual(return_new,None)
        self.assertEqual(return_new,return_old)

    def test_sym_C5(self):
        new_copy=deepcopy(self.projs)
        old_copy = deepcopy(self.projs)
        return_new = fu.mirror_and_reduce_dsym(params = self.projs, indexes=self.subset, symmetry_class=self.symmetry_class)

        return_old = oldfu.mirror_and_reduce_dsym(params =self.projs, indexes=self.subset, symmetry_class=self.symmetry_class)
        self.assertTrue(numpy_array_equal(new_copy,old_copy))
        self.assertTrue(numpy_array_equal(new_copy, self.projs))
        self.assertEqual(return_new,None)
        self.assertEqual(return_new,return_old)

    def test_sym_D1(self):
        symmetry_class=deepcopy(self.symmetry_class)
        symmetry_class.sym ="d"
        symmetry_class.nsym =1
        new_copy=deepcopy(self.projs)
        old_copy = deepcopy(self.projs)
        return_new = fu.mirror_and_reduce_dsym(params = self.projs, indexes=self.subset, symmetry_class=symmetry_class)

        return_old = oldfu.mirror_and_reduce_dsym(params =self.projs, indexes=self.subset, symmetry_class=symmetry_class)
        self.assertTrue(numpy_array_equal(new_copy,old_copy))
        self.assertTrue(numpy_array_equal(new_copy, self.projs))
        self.assertEqual(return_new,None)
        self.assertEqual(return_new,return_old)



class Test_do_volume(unittest.TestCase):
    (data, options, iter) = get_arg_from_pickle_file(path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume"))[0]

    def test_wrong_number_params_returns_TypeError_too_few_parameters(self):
        with self.assertRaises(TypeError) as cm_new:
            fu.do_volume()
        with self.assertRaises(TypeError) as cm_old:
            oldfu.do_volume()
        self.assertEqual(cm_new.exception.message, "do_volume() takes exactly 4 arguments (0 given)")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_pickle_file(self):
        mpi_barrier(MPI_COMM_WORLD)
        return_new = fu.do_volume(data=self.data,options=self.options,iter=None, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.do_volume(data=self.data,options=self.options,iter=None, mpi_comm = MPI_COMM_WORLD)
        self.assertTrue(allclose(return_old.get_3dview(), return_new.get_3dview(), atol=TOLERANCE))

    def test_emptyData_returns_RuntimeError(self):
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(RuntimeError) as cm_new:
            fu.do_volume(data=[EMData()],options=self.options,iter=None, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(RuntimeError) as cm_old:
            oldfu.do_volume(data=[EMData()],options=self.options,iter=None, mpi_comm = MPI_COMM_WORLD)
        msg = cm_new.exception.message.split("'")
        msg_old = cm_old.exception.message.split("'")
        self.assertEqual(msg[0].split(" ")[0], "NotExistingObjectException")
        self.assertEqual(msg[3], "The requested key does not exist")
        self.assertEqual(msg[0].split(" ")[0], msg_old[0].split(" ")[0])
        self.assertEqual(msg[3], msg_old[3])
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)

    def test_no_data_returns_indexError(self):
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(IndexError) as cm_new:
            fu.do_volume(data=[],options=self.options,iter=None, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(IndexError) as cm_old:
            oldfu.do_volume(data=[],options=self.options,iter=None, mpi_comm = MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, "list index out of range")
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)


    def test_empty_options_returns_AttributeError(self):
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(AttributeError) as cm_new:
            fu.do_volume(data=self.data,options={},iter=None, mpi_comm = MPI_COMM_WORLD)
        mpi_barrier(MPI_COMM_WORLD)
        with self.assertRaises(AttributeError) as cm_old:
            oldfu.do_volume(data=self.data,options={},iter=None, mpi_comm = MPI_COMM_WORLD)
        self.assertEqual(cm_new.exception.message, cm_old.exception.message)
        self.assertEqual(cm_new.exception.message, "'dict' object has no attribute 'sym'")






"""
@unittest.skip("original adnans tests")
class Test_lib_multi_shc_compare(unittest.TestCase):

    def test_orient_params_true_should_return_equal_objects(self):
        from sphire.libpy import sparx_fundamentals
        print(ABSOLUTE_PATH)
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.orient_params")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        (params, refparams, indexes) = argum[0]
        symmetry_class = argum[1]['symmetry_class']


        return_new = fu.orient_params(params, refparams,indexes,symmetry_class)

        return_old = oldfu.orient_params(params, refparams,indexes,symmetry_class)

        self.assertEqual(return_new, return_old)


    def test_find_common_subset_true_should_return_equal_objects(self):
        print(ABSOLUTE_PATH)
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.find_common_subset")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (projs,target_threshold,minimal_subset_size,symmetry_class) = argum[0]

        print(symmetry_class)

        print(symmetry_class.sym)

        return_new = fu.find_common_subset(projs,target_threshold,minimal_subset_size,symmetry_class)

        return_old = oldfu.find_common_subset(projs, target_threshold, minimal_subset_size,symmetry_class)

        self.assertEqual(return_new, return_old)


    #    Cannot Work without proper value of mpi_comm . have to ask markus for this
    
    def test_ali3d_multishc_true_should_return_equal_objects(self):

        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.ali3d_multishc")
        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum)

        (stack, ref_vol, ali3d_options, symmetry_class) = argum[0]

        (dd) = argum[1]

        print('stack values are', stack)
        print('refvol values are', ref_vol)
        print('ali3d_option are ', ali3d_options)
        print('symmetry_class are', symmetry_class)
        print('argument 1 are' , dd)


        mpi_barrier(MPI_COMM_WORLD)

        return_new = fu.ali3d_multishc(stack, ref_vol, ali3d_options, symmetry_class, number_of_runs=1)

        mpi_barrier(MPI_COMM_WORLD)

        return_old = oldfu.ali3d_multishc(stack, ref_vol, ali3d_options, symmetry_class, number_of_runs=1)

        mpi_barrier(MPI_COMM_WORLD)

        # mpi_finalize()

        if (return_old is not None  and return_new is not None) :
            self.assertTrue(return_new, return_old)



    # def test_ali3d_multishc_2_true_should_return_equal_objects(self):
    #
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.ali3d_multishc_2")
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     print(argum)
    #
    #     (stack, ref_vol, ali3d_options, symmetry_class) = argum[0]
    #
    #     (dd) = argum[1]
    #
    #     print('stack values are', stack)
    #     print('refvol values are', ref_vol)
    #     print('ali3d_option are ', ali3d_options)
    #     print('symmetry_class are', symmetry_class)
    #     print('argument 1 are' , dd)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_new = fu.ali3d_multishc_2(stack, ref_vol, ali3d_options, symmetry_class)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_old = oldfu.ali3d_multishc_2(stack, ref_vol, ali3d_options, symmetry_class)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #

    #    Cannot Work without proper value of mpi_comm . have to ask markus for this

    # def test_multi_shc_true_should_return_equal_object(self):
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc.multi_shc")
    #     import sparx_fundamentals
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #         print(argum[0])
    #
    #     (all_projs, subset, runs_count, ali3d_options) = argum[0]
    #
    #
    #     filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc.find_common_subset")
    #     import sparx_fundamentals
    #     with open(filepath, 'rb') as rb:
    #         argum = pickle.load(rb)
    #
    #     (projs, target_threshold, minimal_subset_size, symmetry_class) = argum[0]
    #
    #     all_projs = projs
    #     n = len(projs[0])
    #     subset = list(range(n))
    #
    #     print(type(subset))
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_new = fu.multi_shc(all_projs, subset, runs_count, ali3d_options, mpi_comm=MPI_COMM_WORLD)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     return_old = fu.multi_shc(all_projs, subset, runs_count, ali3d_options, mpi_comm=MPI_COMM_WORLD)
    #
    #     mpi_barrier(MPI_COMM_WORLD)
    #
    #     if (return_old is not None  and return_new is not None) :
    #         self.assertTrue(return_new, return_old)


    def test_mirror_and_reduce_dsym_true_should_return_equal_object(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.find_common_subset")

        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        (projs, target_threshold, minimal_subset_size, symmetry_class) = argum[0]
        n = len(projs[0])
        subset = list(range(n))

        return_new = fu.mirror_and_reduce_dsym(projs, subset, symmetry_class)
        return_old = oldfu.mirror_and_reduce_dsym(projs, subset, symmetry_class)

        if (return_old is not None and return_new is not None):
            self.assertTrue(return_new, return_old)


    
    #Cannot Work without proper value of mpi_comm . have to ask markus for this
    

    def test_do_volume_true_should_return_equal_object(self):
        filepath = os.path.join(ABSOLUTE_PATH, "pickle files/multi_shc/multi_shc.do_volume")

        with open(filepath, 'rb') as rb:
            argum = pickle.load(rb)

        print(argum[0])

        (data,options,iter) = argum[0]

        mpi_barrier(MPI_COMM_WORLD)
        return_new = fu.do_volume(data,options,iter, mpi_comm = MPI_COMM_WORLD)

        mpi_barrier(MPI_COMM_WORLD)
        return_old = oldfu.do_volume(data, options, iter, mpi_comm = MPI_COMM_WORLD)

        if (return_old is not None  and return_new is not None) :
            self.assertTrue(return_new, return_old)
"""

if __name__ == '__main__':
    unittest.main()
    mpi_finalize()

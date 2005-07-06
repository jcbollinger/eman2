
// Boost Includes ==============================================================
#include <boost/python.hpp>
#include <boost/cstdint.hpp>

// Includes ====================================================================
#include <emdata.h>
#include <emutil.h>
#include <imageio.h>
#include <testutil.h>
#include <util.h>
#include <xydata.h>

// Using =======================================================================
using namespace boost::python;

// Declarations ================================================================
namespace  {

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_Util_find_max_overloads_3_4, EMAN::Util::find_max, 3, 4)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_Util_find_min_and_max_overloads_4_6, EMAN::Util::find_min_and_max, 4, 6)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_Util_printMatI3D_overloads_1_3, EMAN::Util::printMatI3D, 1, 3)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_Util_voea_overloads_1_5, EMAN::Util::voea, 1, 5)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_Util_quadri_overloads_5_6, EMAN::Util::quadri, 5, 6)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_EMUtil_get_imageio_overloads_2_3, EMAN::EMUtil::get_imageio, 2, 3)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_EMUtil_process_region_io_overloads_7_13, EMAN::EMUtil::process_region_io, 7, 13)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_TestUtil_check_image_overloads_1_2, EMAN::TestUtil::check_image, 1, 2)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_TestUtil_make_image_file_overloads_2_6, EMAN::TestUtil::make_image_file, 2, 6)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_TestUtil_verify_image_file_overloads_2_6, EMAN::TestUtil::verify_image_file, 2, 6)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_TestUtil_make_image_file2_overloads_2_6, EMAN::TestUtil::make_image_file2, 2, 6)

BOOST_PYTHON_FUNCTION_OVERLOADS(EMAN_TestUtil_verify_image_file2_overloads_2_6, EMAN::TestUtil::verify_image_file2, 2, 6)


}// namespace 


// Module ======================================================================
BOOST_PYTHON_MODULE(libpyUtils2)
{
    scope* EMAN_Util_scope = new scope(
    class_< EMAN::Util >("Util", init<  >())
        .def(init< const EMAN::Util& >())
        .def("ap2ri", &EMAN::Util::ap2ri)
        .def("flip_complex_phase", &EMAN::Util::flip_complex_phase)
        .def("file_lock_wait", &EMAN::Util::file_lock_wait)
        .def("check_file_by_magic", &EMAN::Util::check_file_by_magic)
        .def("is_file_exist", &EMAN::Util::is_file_exist)
        .def("flip_image", &EMAN::Util::flip_image)
        .def("sstrncmp", &EMAN::Util::sstrncmp)
        .def("int2str", &EMAN::Util::int2str)
        .def("get_line_from_string", &EMAN::Util::get_line_from_string)
        .def("get_str_float", (bool (*)(const char*, const char*, float*))&EMAN::Util::get_str_float)
        .def("get_str_float", (bool (*)(const char*, const char*, float*, float*))&EMAN::Util::get_str_float)
        .def("get_str_float", (bool (*)(const char*, const char*, int*, float*, float*))&EMAN::Util::get_str_float)
        .def("get_str_int", (bool (*)(const char*, const char*, int*))&EMAN::Util::get_str_int)
        .def("get_str_int", (bool (*)(const char*, const char*, int*, int*))&EMAN::Util::get_str_int)
        .def("get_str_int", (bool (*)(const char*, const char*, int*, int*, int*))&EMAN::Util::get_str_int)
        .def("change_filename_ext", &EMAN::Util::change_filename_ext)
        .def("remove_filename_ext", &EMAN::Util::remove_filename_ext)
        .def("get_filename_ext", &EMAN::Util::get_filename_ext)
        .def("sbasename", &EMAN::Util::sbasename)
        .def("calc_least_square_fit", &EMAN::Util::calc_least_square_fit)
        .def("save_data", (void (*)(const std::vector<float,std::allocator<float> >&, const std::vector<float,std::allocator<float> >&, const std::string&))&EMAN::Util::save_data)
        .def("save_data", (void (*)(float, float, const std::vector<float,std::allocator<float> >&, const std::string&))&EMAN::Util::save_data)
        .def("save_data", (void (*)(float, float, float*, size_t, const std::string&))&EMAN::Util::save_data)
        .def("get_frand", (float (*)(int, int))&EMAN::Util::get_frand)
        .def("get_frand", (float (*)(float, float))&EMAN::Util::get_frand)
        .def("get_frand", (float (*)(double, double))&EMAN::Util::get_frand)
        .def("get_gauss_rand", &EMAN::Util::get_gauss_rand)
        .def("round", (int (*)(float))&EMAN::Util::round)
        .def("round", (int (*)(double))&EMAN::Util::round)
        .def("bilinear_interpolate", &EMAN::Util::bilinear_interpolate)
        .def("trilinear_interpolate", &EMAN::Util::trilinear_interpolate)
        .def("find_max", &EMAN::Util::find_max, EMAN_Util_find_max_overloads_3_4())
        .def("find_min_and_max", &EMAN::Util::find_min_and_max, EMAN_Util_find_min_and_max_overloads_4_6())
        .def("calc_best_fft_size", &EMAN::Util::calc_best_fft_size)
        .def("square", (int (*)(int))&EMAN::Util::square)
        .def("square", (float (*)(float))&EMAN::Util::square)
        .def("square", (float (*)(double))&EMAN::Util::square)
        .def("square_sum", &EMAN::Util::square_sum)
        .def("hypot3", (float (*)(int, int, int))&EMAN::Util::hypot3)
        .def("hypot3", (float (*)(float, float, float))&EMAN::Util::hypot3)
        .def("hypot3", (float (*)(double, double, double))&EMAN::Util::hypot3)
        .def("fast_floor", &EMAN::Util::fast_floor)
        .def("agauss", &EMAN::Util::agauss)
        .def("get_min", (int (*)(int, int))&EMAN::Util::get_min)
        .def("get_min", (int (*)(int, int, int))&EMAN::Util::get_min)
        .def("get_min", (float (*)(float, float))&EMAN::Util::get_min)
        .def("get_min", (float (*)(float, float, float))&EMAN::Util::get_min)
        .def("get_min", (float (*)(float, float, float, float))&EMAN::Util::get_min)
        .def("get_max", (float (*)(float, float))&EMAN::Util::get_max)
        .def("get_max", (float (*)(float, float, float))&EMAN::Util::get_max)
        .def("get_max", (float (*)(float, float, float, float))&EMAN::Util::get_max)
        .def("angle_sub_2pi", &EMAN::Util::angle_sub_2pi)
        .def("angle_sub_pi", &EMAN::Util::angle_sub_pi)
        .def("goodf", &EMAN::Util::goodf)
        .def("get_time_label", &EMAN::Util::get_time_label)
        .def("set_log_level", &EMAN::Util::set_log_level)
        .def("eman_copysign", &EMAN::Util::eman_copysign)
        .def("eman_erfc", &EMAN::Util::eman_erfc)
        .def("printMatI3D", &EMAN::Util::printMatI3D, EMAN_Util_printMatI3D_overloads_1_3())
        .def("voea", &EMAN::Util::voea, EMAN_Util_voea_overloads_1_5())
        .def("triquad", &EMAN::Util::triquad)
        .def("quadri", &EMAN::Util::quadri, EMAN_Util_quadri_overloads_5_6())
        .staticmethod("flip_complex_phase")
        .staticmethod("square")
        .staticmethod("get_min")
        .staticmethod("int2str")
        .staticmethod("get_str_int")
        .staticmethod("printMatI3D")
        .staticmethod("calc_best_fft_size")
        .staticmethod("get_frand")
        .staticmethod("angle_sub_pi")
        .staticmethod("get_time_label")
        .staticmethod("trilinear_interpolate")
        .staticmethod("file_lock_wait")
        .staticmethod("triquad")
        .staticmethod("is_file_exist")
        .staticmethod("save_data")
        .staticmethod("set_log_level")
        .staticmethod("get_str_float")
        .staticmethod("change_filename_ext")
        .staticmethod("angle_sub_2pi")
        .staticmethod("get_gauss_rand")
        .staticmethod("remove_filename_ext")
        .staticmethod("quadri")
        .staticmethod("find_max")
        .staticmethod("fast_floor")
        .staticmethod("ap2ri")
        .staticmethod("sstrncmp")
        .staticmethod("get_max")
        .staticmethod("check_file_by_magic")
        .staticmethod("agauss")
        .staticmethod("eman_erfc")
        .staticmethod("voea")
        .staticmethod("bilinear_interpolate")
        .staticmethod("calc_least_square_fit")
        .staticmethod("find_min_and_max")
        .staticmethod("get_filename_ext")
        .staticmethod("get_line_from_string")
        .staticmethod("square_sum")
        .staticmethod("goodf")
        .staticmethod("hypot3")
        .staticmethod("flip_image")
        .staticmethod("eman_copysign")
        .staticmethod("sbasename")
        .staticmethod("round")
    );

    class_< EMAN::Util::KaiserBessel >("KaiserBessel", init< const EMAN::Util::KaiserBessel& >())
        .def(init< optional< float, float, float > >())
        .def("kb1d", &EMAN::Util::KaiserBessel::kb1d)
        .def("kbtf1d", &EMAN::Util::KaiserBessel::kbtf1d)
    ;

    delete EMAN_Util_scope;

    scope* EMAN_EMUtil_scope = new scope(
    class_< EMAN::EMUtil >("EMUtil", init<  >())
        .def(init< const EMAN::EMUtil& >())
        .def("vertical_acf", &EMAN::EMUtil::vertical_acf, return_value_policy< manage_new_object >())
        .def("make_image_median", &EMAN::EMUtil::make_image_median, return_value_policy< manage_new_object >())
        .def("get_image_ext_type", &EMAN::EMUtil::get_image_ext_type)
        .def("get_image_type", &EMAN::EMUtil::get_image_type)
        .def("get_image_count", &EMAN::EMUtil::get_image_count)
        .def("get_imageio", &EMAN::EMUtil::get_imageio, EMAN_EMUtil_get_imageio_overloads_2_3()[ return_internal_reference< 1 >() ])
        .def("get_imagetype_name", &EMAN::EMUtil::get_imagetype_name)
        .def("get_datatype_string", &EMAN::EMUtil::get_datatype_string)
        .def("process_region_io", &EMAN::EMUtil::process_region_io, EMAN_EMUtil_process_region_io_overloads_7_13())
        .def("process_ascii_region_io", &EMAN::EMUtil::process_ascii_region_io)
        .def("dump_dict", &EMAN::EMUtil::dump_dict)
        .def("is_same_size", &EMAN::EMUtil::is_same_size)
        .def("is_same_ctf", &EMAN::EMUtil::is_same_ctf)
        .def("is_complex_type", &EMAN::EMUtil::is_complex_type)
        .def("jump_lines", &EMAN::EMUtil::jump_lines)
        .def("get_euler_names", &EMAN::EMUtil::get_euler_names)
        .staticmethod("vertical_acf")
        .staticmethod("get_datatype_string")
        .staticmethod("dump_dict")
        .staticmethod("get_imageio")
        .staticmethod("get_image_count")
        .staticmethod("get_imagetype_name")
        .staticmethod("get_image_type")
        .staticmethod("is_same_size")
        .staticmethod("make_image_median")
        .staticmethod("jump_lines")
        .staticmethod("process_region_io")
        .staticmethod("is_same_ctf")
        .staticmethod("get_euler_names")
        .staticmethod("get_image_ext_type")
        .staticmethod("process_ascii_region_io")
        .staticmethod("is_complex_type")
    );

    enum_< EMAN::EMUtil::EMDataType >("EMDataType")
        .value("EM_SHORT_COMPLEX", EMAN::EMUtil::EM_SHORT_COMPLEX)
        .value("EM_SHORT", EMAN::EMUtil::EM_SHORT)
        .value("EM_UCHAR", EMAN::EMUtil::EM_UCHAR)
        .value("EM_FLOAT_COMPLEX", EMAN::EMUtil::EM_FLOAT_COMPLEX)
        .value("EM_CHAR", EMAN::EMUtil::EM_CHAR)
        .value("EM_INT", EMAN::EMUtil::EM_INT)
        .value("EM_USHORT", EMAN::EMUtil::EM_USHORT)
        .value("EM_USHORT_COMPLEX", EMAN::EMUtil::EM_USHORT_COMPLEX)
        .value("EM_UNKNOWN", EMAN::EMUtil::EM_UNKNOWN)
        .value("EM_UINT", EMAN::EMUtil::EM_UINT)
        .value("EM_DOUBLE", EMAN::EMUtil::EM_DOUBLE)
        .value("EM_FLOAT", EMAN::EMUtil::EM_FLOAT)
    ;


    enum_< EMAN::EMUtil::ImageType >("ImageType")
        .value("IMAGE_XPLOR", EMAN::EMUtil::IMAGE_XPLOR)
        .value("IMAGE_MRC", EMAN::EMUtil::IMAGE_MRC)
        .value("IMAGE_GATAN2", EMAN::EMUtil::IMAGE_GATAN2)
        .value("IMAGE_ICOS", EMAN::EMUtil::IMAGE_ICOS)
        .value("IMAGE_UNKNOWN", EMAN::EMUtil::IMAGE_UNKNOWN)
        .value("IMAGE_LST", EMAN::EMUtil::IMAGE_LST)
        .value("IMAGE_DM3", EMAN::EMUtil::IMAGE_DM3)
        .value("IMAGE_TIFF", EMAN::EMUtil::IMAGE_TIFF)
        .value("IMAGE_SAL", EMAN::EMUtil::IMAGE_SAL)
        .value("IMAGE_IMAGIC", EMAN::EMUtil::IMAGE_IMAGIC)
        .value("IMAGE_VTK", EMAN::EMUtil::IMAGE_VTK)
        .value("IMAGE_HDF", EMAN::EMUtil::IMAGE_HDF)
        .value("IMAGE_SPIDER", EMAN::EMUtil::IMAGE_SPIDER)
        .value("IMAGE_EMIM", EMAN::EMUtil::IMAGE_EMIM)
        .value("IMAGE_SINGLE_SPIDER", EMAN::EMUtil::IMAGE_SINGLE_SPIDER)
        .value("IMAGE_PNG", EMAN::EMUtil::IMAGE_PNG)
        .value("IMAGE_PGM", EMAN::EMUtil::IMAGE_PGM)
        .value("IMAGE_EM", EMAN::EMUtil::IMAGE_EM)
        .value("IMAGE_PIF", EMAN::EMUtil::IMAGE_PIF)
        .value("IMAGE_AMIRA", EMAN::EMUtil::IMAGE_AMIRA)
    ;

    delete EMAN_EMUtil_scope;

    class_< EMAN::ImageSort >("ImageSort", init< const EMAN::ImageSort& >())
        .def(init< int >())
        .def("sort", &EMAN::ImageSort::sort)
        .def("set", &EMAN::ImageSort::set)
        .def("get_index", &EMAN::ImageSort::get_index)
        .def("get_score", &EMAN::ImageSort::get_score)
        .def("size", &EMAN::ImageSort::size)
    ;

    class_< EMAN::TestUtil >("TestUtil", init<  >())
        .def(init< const EMAN::TestUtil& >())
        .def_readonly("EMDATA_HEADER_EXT", &EMAN::TestUtil::EMDATA_HEADER_EXT)
        .def_readonly("EMDATA_DATA_EXT", &EMAN::TestUtil::EMDATA_DATA_EXT)
        .def("get_debug_int", &EMAN::TestUtil::get_debug_int)
        .def("get_debug_float", &EMAN::TestUtil::get_debug_float)
        .def("get_debug_string", &EMAN::TestUtil::get_debug_string)
        .def("get_debug_image", &EMAN::TestUtil::get_debug_image)
        .def("get_golden_image", &EMAN::TestUtil::get_golden_image)
        .def("to_emobject", &EMAN::TestUtil::to_emobject)
        .def("emobject_to_py", (EMAN::EMObject (*)(int))&EMAN::TestUtil::emobject_to_py)
        .def("emobject_to_py", (EMAN::EMObject (*)(float))&EMAN::TestUtil::emobject_to_py)
        .def("emobject_to_py", (EMAN::EMObject (*)(double))&EMAN::TestUtil::emobject_to_py)
        .def("emobject_to_py", (EMAN::EMObject (*)(const std::string&))&EMAN::TestUtil::emobject_to_py)
        .def("emobject_to_py", (EMAN::EMObject (*)(EMAN::EMData*))&EMAN::TestUtil::emobject_to_py)
        .def("emobject_to_py", (EMAN::EMObject (*)(EMAN::XYData*))&EMAN::TestUtil::emobject_to_py)
        .def("emobject_farray_to_py", &EMAN::TestUtil::emobject_farray_to_py)
        .def("emobject_strarray_to_py", &EMAN::TestUtil::emobject_strarray_to_py)
        .def("test_IntPoint", &EMAN::TestUtil::test_IntPoint)
        .def("test_FloatPoint", &EMAN::TestUtil::test_FloatPoint)
        .def("test_IntSize", &EMAN::TestUtil::test_IntSize)
        .def("test_FloatSize", &EMAN::TestUtil::test_FloatSize)
        .def("test_Vec3i", &EMAN::TestUtil::test_Vec3i)
        .def("test_Vec3f", &EMAN::TestUtil::test_Vec3f)
        .def("test_vector_int", &EMAN::TestUtil::test_vector_int)
        .def("test_vector_float", &EMAN::TestUtil::test_vector_float)
        .def("test_vector_long", &EMAN::TestUtil::test_vector_long)
        .def("test_vector_string", &EMAN::TestUtil::test_vector_string)
        .def("test_vector_emdata", &EMAN::TestUtil::test_vector_emdata)
        .def("test_vector_pixel", &EMAN::TestUtil::test_vector_pixel)
        .def("test_map_int", &EMAN::TestUtil::test_map_int)
        .def("test_map_long", &EMAN::TestUtil::test_map_long)
        .def("test_map_float", &EMAN::TestUtil::test_map_float)
        .def("test_map_string", &EMAN::TestUtil::test_map_string)
        .def("test_map_emobject", &EMAN::TestUtil::test_map_emobject)
        .def("test_map_vecstring", &EMAN::TestUtil::test_map_vecstring)
        .def("test_dict", &EMAN::TestUtil::test_dict)
        .def("dump_image_from_file", &EMAN::TestUtil::dump_image_from_file)
        .def("dump_emdata", &EMAN::TestUtil::dump_emdata)
        .def("check_image", &EMAN::TestUtil::check_image, EMAN_TestUtil_check_image_overloads_1_2())
        .def("set_progname", &EMAN::TestUtil::set_progname)
        .def("make_image_file", &EMAN::TestUtil::make_image_file, EMAN_TestUtil_make_image_file_overloads_2_6())
        .def("verify_image_file", &EMAN::TestUtil::verify_image_file, EMAN_TestUtil_verify_image_file_overloads_2_6())
        .def("make_image_file2", &EMAN::TestUtil::make_image_file2, EMAN_TestUtil_make_image_file2_overloads_2_6())
        .def("verify_image_file2", &EMAN::TestUtil::verify_image_file2, EMAN_TestUtil_verify_image_file2_overloads_2_6())
        .staticmethod("test_Vec3f")
        .staticmethod("verify_image_file2")
        .staticmethod("test_vector_float")
        .staticmethod("test_Vec3i")
        .staticmethod("test_map_int")
        .staticmethod("make_image_file")
        .staticmethod("emobject_strarray_to_py")
        .staticmethod("test_map_vecstring")
        .staticmethod("get_debug_int")
        .staticmethod("test_vector_long")
        .staticmethod("emobject_farray_to_py")
        .staticmethod("test_IntPoint")
        .staticmethod("dump_image_from_file")
        .staticmethod("dump_emdata")
        .staticmethod("set_progname")
        .staticmethod("test_map_float")
        .staticmethod("test_IntSize")
        .staticmethod("test_FloatSize")
        .staticmethod("verify_image_file")
        .staticmethod("test_map_long")
        .staticmethod("to_emobject")
        .staticmethod("make_image_file2")
        .staticmethod("emobject_to_py")
        .staticmethod("get_golden_image")
        .staticmethod("test_map_string")
        .staticmethod("test_vector_int")
        .staticmethod("test_vector_emdata")
        .staticmethod("test_vector_string")
        .staticmethod("get_debug_string")
        .staticmethod("test_FloatPoint")
        .staticmethod("test_dict")
        .staticmethod("test_map_emobject")
        .staticmethod("test_vector_pixel")
        .staticmethod("get_debug_float")
        .staticmethod("get_debug_image")
        .staticmethod("check_image")
    ;

}


/**
 * $Id$
 */
#ifndef eman__vtkio_h__
#define eman__vtkio_h__ 1

#include "imageio.h"
#include <stdio.h>

namespace EMAN
{

	/** VtkIO reads/writes VTK image file.
	 *
	 * VTK is a file format used by the Visual Toolkit.
	 * (http://public.kitware.com/VTK/)
	 *
	 * There are 2 VTK formats: ASCII or Binary.
	 *
	 * ASCII format
	 * It has 5 parts:
	 *
	 * - 1) file version and indentifier. It is one line with 
	 *      "# vtk DataFile Version x.x".
	 *
	 * - 2) description header. one line. it is 256 char maximum. 
	 *
	 * - 3) file format. one word. either "ASCII" or "BINARY".
	 *
	 * - 4) dataset structure. it describes the geometry and topology
	 *      of the dataset. It must be one line containing the keyword
	 *      "DATASET" followed by a keyword describing the type of
	 *      dataset. This part is optional.
	 *
	 *
	 * - 5) dataset attributes. It begins with POINT_DATA or CELL_DATA.
	 *
	 * Binary format
	 * It has the same 5 parts like ASCII format,followed by data in
	 * binary format. The data are stored in big endian by default..
	 * 
	 *
	 * A VTK file contains 1 2D or 3D image.
	 */
	class VtkIO:public ImageIO
	{
	  public:
		VtkIO(const string & filename, IOMode rw_mode = READ_ONLY);
		~VtkIO();

		DEFINE_IMAGEIO_FUNC;
		static bool is_valid(const void *first_block);

	  private:
		static const char *MAGIC;

		enum VtkType
		{
			VTK_UNKNOWN,
			VTK_ASCII,
			VTK_BINARY			
		};

		enum DataType
		{
			DATATYPE_UNKNOWN,
			BIT,
			UNSIGNED_CHAR,
			CHAR,			
			UNSIGNED_SHORT,
			SHORT,
			UNSIGNED_INT,
			INT,
			UNSIGNED_LONG,
			LONG,
			FLOAT,
			DOUBLE
		};

		enum DatasetType
		{
			DATASET_UNKNOWN,
			STRUCTURED_POINTS,
			STRUCTURED_GRID,
			RECTILINEAR_GRID,
			UNSTRUCTURED_GRID,
			POLYDATA
		};
		
		int to_em_datatype(int vtk_datatype);
		int get_mode_size(DataType d);
		
		void read_dataset(DatasetType dstype);
		DataType get_datatype_from_name(const string& datatype_name);
		DatasetType get_datasettype_from_name(const string& dataset_name);
	
		string filename;
		IOMode rw_mode;
		FILE *vtk_file;
		bool is_big_endian;
		bool is_new_file;
		bool initialized;

		DataType datatype;
		VtkType filetype;
		int nx;
		int ny;
		int nz;
		float originx;
		float originy;
		float originz;
		float spacingx;
		float spacingy;
		float spacingz;
		off_t file_offset;

	};

}

#endif

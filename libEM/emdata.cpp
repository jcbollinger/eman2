/**
 * $Id$
 */

#include "emdata.h"
#include "all_imageio.h"
#include "ctf.h"
#include "processor.h"
#include "aligner.h"
#include "cmp.h"
#include "emfft.h"
#include "projector.h"

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>

#include <iomanip>
#include <complex>

#ifdef WIN32
	#define M_PI 3.14159265358979323846f
#endif	//WIN32

using namespace EMAN;
using namespace std;
using namespace boost;

int EMData::totalalloc=0;		// mainly used for debugging/memory leak purposes

EMData::EMData()
{
	ENTERFUNC;

	rdata = 0;
	supp = 0;
	ctf = 0;

	flags =0;
	// used to replace cube 'pixel'
	attr_dict["apix_x"] = 1.0f;
	attr_dict["apix_y"] = 1.0f;
	attr_dict["apix_z"] = 1.0f;

	attr_dict["is_complex"] = int(0);
	attr_dict["is_complex_x"] = int(0);
	attr_dict["is_complex_ri"] = int(1);

	changecount=0;

	nx = 0;
	ny = 0;
	nz = 0;
	xoff = yoff = zoff = 0;

	EMData::totalalloc++;
#ifdef MEMDEBUG
	printf("EMDATA+  %4d    %p\n",EMData::totalalloc,this);
#endif
	EXITFUNC;
}

EMData::~EMData()
{
	ENTERFUNC;
	if (rdata) {
		free(rdata);
		rdata = 0;
	}

	if (supp) {
		free(supp);
		supp = 0;
	}

	if (ctf) {
		delete ctf;
		ctf = 0;
	}

	EMData::totalalloc--;
#ifdef MEMDEBUG
	printf("EMDATA-  %4d    %p\n",EMData::totalalloc,this);
#endif
	EXITFUNC;
}


EMData *EMData::get_clip(const Region & area)
{
	ENTERFUNC;
	if (get_ndim() != area.get_ndim()) {
		LOGERR("cannot get %dD clip out of %dD image", get_ndim(), area.get_ndim());
		return 0;
	}

	EMData *result = new EMData();
	int zsize = (int)area.size[2];
	if (zsize == 0 || nz <= 1) {
		zsize = 1;
	}
	int ysize = (ny<=1 && (int)area.size[1]==0 ? 1 : (int)area.size[1]);

	result->set_size((int)area.size[0], ysize, zsize);

	int x0 = (int) area.origin[0];
	x0 = x0 < 0 ? 0 : x0;

	int y0 = (int) area.origin[1];
	y0 = y0 < 0 ? 0 : y0;

	int z0 = (int) area.origin[2];
	z0 = z0 < 0 ? 0 : z0;

	int x1 = (int) (area.origin[0] + area.size[0]);
	x1 = x1 > nx ? nx : x1;

	int y1 = (int) (area.origin[1] + area.size[1]);
	y1 = y1 > ny ? ny : y1;

	int z1 = (int) (area.origin[2] + area.size[2]);
	z1 = z1 > nz ? nz : z1;
	if (z1 <= 0) {
		z1 = 1;
	}

	int xd0 = (int) (area.origin[0] < 0 ? -area.origin[0] : 0);
	int yd0 = (int) (area.origin[1] < 0 ? -area.origin[1] : 0);
	int zd0 = (int) (area.origin[2] < 0 ? -area.origin[2] : 0);

	size_t clipped_row_size = (x1-x0) * sizeof(float);
	int src_secsize = nx * ny;
	int dst_secsize = (int)(area.size[0] * area.size[1]);

	float *src = rdata + z0 * src_secsize + y0 * nx + x0;
	float *dst = result->get_data();
	dst += zd0 * dst_secsize + yd0 * (int)area.size[0] + xd0;

	int src_gap = src_secsize - (y1-y0) * nx;
	int dst_gap = dst_secsize - (y1-y0) * (int)area.size[0];

	for (int i = z0; i < z1; i++) {
		for (int j = y0; j < y1; j++) {
			memcpy(dst, src, clipped_row_size);
			src += nx;
			dst += (int)area.size[0];
		}
		src += src_gap;
		dst += dst_gap;
	}

	result->done_data();

	if( attr_dict.has_key("apix_x") && attr_dict.has_key("apix_y") &&
		attr_dict.has_key("apix_z") )
	{
		result->attr_dict["apix_x"] = attr_dict["apix_x"];
		result->attr_dict["apix_y"] = attr_dict["apix_y"];
		result->attr_dict["apix_z"] = attr_dict["apix_z"];

		if( attr_dict.has_key("origin_row") && attr_dict.has_key("origin_col") &&
		    attr_dict.has_key("origin_sec") )
		{
			float xorigin = attr_dict["origin_row"];
			float yorigin = attr_dict["origin_col"];
			float zorigin = attr_dict["origin_sec"];

			float apix_x = attr_dict["apix_x"];
			float apix_y = attr_dict["apix_y"];
			float apix_z = attr_dict["apix_z"];

			result->set_xyz_origin(xorigin + apix_x * area.origin[0],
							   	   yorigin + apix_y * area.origin[1],
							       zorigin + apix_z * area.origin[2]);
		}
	}

	result->update();

	result->set_path(path);
	result->set_pathnum(pathnum);

	EXITFUNC;
	return result;
}


EMData *EMData::get_top_half() const
{
	ENTERFUNC;

	if (get_ndim() != 3) {
		throw ImageDimensionException("3D only");
	}

	EMData *half = new EMData();
	half->attr_dict = attr_dict;
	half->set_size(nx, ny, nz / 2);

	float *half_data = half->get_data();
	memcpy(half_data, &rdata[nz / 2 * nx * ny], sizeof(float) * nx * ny * nz / 2);
	half->done_data();

	float apix_z = attr_dict["apix_z"];
	float origin_sec = attr_dict["origin_sec"];
	origin_sec += apix_z * nz / 2;
	half->attr_dict["origin_sec"] = origin_sec;
	half->update();

	EXITFUNC;
	return half;
}


EMData *EMData::get_rotated_clip(const Transform3D &xform,
								 const IntSize &size, float scale)
{
	EMData *result = new EMData();
	result->set_size(size[0],size[1],size[2]);

	for (int z=-size[2]/2; z<size[2]/2; z++) {
		for (int y=-size[1]/2; y<size[1]/2; y++) {
			for (int x=-size[0]/2; x<size[0]/2; x++) {
				Vec3f xv=Vec3f((float)x,(float)y,(float)z)*xform;
				float v = 0;

				if (xv[0]<0||xv[1]<0||xv[2]<0||xv[0]>nx-2||xv[1]>ny-2||xv[2]>nz-2) v=0.;
				else v=sget_value_at_interp(xv[0],xv[1],xv[2]);
				result->set_value_at(x+size[0]/2,y+size[1]/2,z+size[2]/2,v);
			}
		}
	}
	result->update();

	return result;
}


EMData* EMData::window_center(int l) {
	ENTERFUNC;
	// sanity checks
	int n = nx;
	if (is_complex()) {
		LOGERR("Need real-space data for window_center()");
		throw ImageFormatException(
			"Complex input image; real-space expected.");
	}
	if (is_fftpadded()) {
		// image has been fft-padded, compute the real-space size
		n -= (2 - int(is_fftodd()));
	}
	int corner = n/2 - l/2;
	int ndim = get_ndim();
	EMData* ret;
	switch (ndim) {
		case 3:
			if ((n != ny) || (n != nz)) {
				LOGERR("Need the real-space image to be cubic.");
				throw ImageFormatException(
						"Need cubic real-space image.");
			}
			ret = get_clip(Region(corner, corner, corner, l, l, l));
			break;
		case 2:
			if (n != ny) {
				LOGERR("Need the real-space image to be square.");
				throw ImageFormatException(
						"Need square real-space image.");
			}
			ret = get_clip(Region(corner, corner, l, l));
			break;
		case 1:
			ret = get_clip(Region(corner, l));
			break;
		default:
			throw ImageDimensionException(
					"window_center only supports 1-d, 2-d, and 3-d images");
	}
	return ret;
	EXITFUNC;
}


float *EMData::setup4slice(bool redo)
{
	ENTERFUNC;

	if (!is_complex()) {
		throw ImageFormatException("complex image only");
	}

	if (get_ndim() != 3) {
		throw ImageDimensionException("3D only");
	}

	if (supp) {
		if (redo) {
			free(supp);
			supp = 0;
		}
		else {
			EXITFUNC;
			return supp;
		}
	}

	const int SUPP_ROW_SIZE = 8;
	const int SUPP_ROW_OFFSET = 4;
	const int supp_size = SUPP_ROW_SIZE + SUPP_ROW_OFFSET;

	supp = (float *) calloc(supp_size * ny * nz, sizeof(float));
	int nxy = nx * ny;
	int supp_xy = supp_size * ny;

	for (int z = 0; z < nz; z++) {
		int cur_z1 = z * nxy;
		int cur_z2 = z * supp_xy;

		for (int y = 0; y < ny; y++) {
			int cur_y1 = y * nx;
			int cur_y2 = y * supp_size;

			for (int x = 0; x < SUPP_ROW_SIZE; x++) {
				int k = (x + SUPP_ROW_OFFSET) + cur_y2 + cur_z2;
				supp[k] = rdata[x + cur_y1 + cur_z1];
			}
		}
	}

	for (int z = 1, zz = nz - 1; z < nz; z++, zz--) {
		int cur_z1 = zz * nxy;
		int cur_z2 = z * supp_xy;

		for (int y = 1, yy = ny - 1; y < ny; y++, yy--) {
			supp[y * 12 + cur_z2] = rdata[4 + yy * nx + cur_z1];
			supp[1 + y * 12 + cur_z2] = -rdata[5 + yy * nx + cur_z1];
			supp[2 + y * 12 + cur_z2] = rdata[2 + yy * nx + cur_z1];
			supp[3 + y * 12 + cur_z2] = -rdata[3 + yy * nx + cur_z1];
		}
	}

	EXITFUNC;
	return supp;
}


void EMData::scale(float s)
{
	ENTERFUNC;
	Transform3D t;
	t.set_scale(s);
	rotate_translate(t);
	EXITFUNC;
}


void EMData::translate(int dx, int dy, int dz)
{
	ENTERFUNC;
	translate(Vec3i(dx, dy, dz));
	EXITFUNC;
}


void EMData::translate(float dx, float dy, float dz)
{
	ENTERFUNC;
	int dx_ = Util::round(dx);
	int dy_ = Util::round(dy);
	int dz_ = Util::round(dz);
	if( ( (dx-dx_) == 0 ) && ( (dy-dy_) == 0 ) && ( (dz-dz_) == 0 )) {
		translate(dx_, dy_, dz_);
	}
	else {
		translate(Vec3f(dx, dy, dz));
	}
	EXITFUNC;
}


void EMData::translate(const Vec3i &translation)
{
	ENTERFUNC;

	//if traslation is 0, do nothing
	if( translation[0] == 0 && translation[1] == 0 && translation[2] == 0) {
		EXITFUNC;
		return;
	}

	float *this_data = get_data();
	int data_size = sizeof(float)*get_xsize()*get_ysize()*get_zsize();
	float *tmp_data = (float *)malloc(data_size);
	memcpy(tmp_data, this_data, data_size);

	int x0, x1, x2;
	if( translation[0] < 0 ) {
		x0 = 0;
		x1 = nx;
		x2 = 1;
	}
	else {
		x0 = nx-1;
		x1 = -1;
		x2 = -1;
	}

	int y0, y1, y2;
	if( translation[1] < 0 ) {
		y0 = 0;
		y1 = ny;
		y2 = 1;
	}
	else {
		y0 = ny-1;
		y1 = -1;
		y2 = -1;
	}

	int z0, z1, z2;
	if( translation[2] < 0 ) {
		z0 = 0;
		z1 = nz;
		z2 = 1;
	}
	else {
		z0 = nz-1;
		z1 = -1;
		z2 = -1;
	}

	int xp, yp, zp;
	int tx = translation[0];
	int ty = translation[1];
	int tz = translation[2];
	for (int y = y0; y != y1; y += y2) {
		for (int x = x0; x != x1; x += x2) {
			for (int z = z0; z != z1; z+=z2) {
				xp = x - tx;
				yp = y - ty;
				zp = z - tz;
				if (xp < 0 || yp < 0 || zp<0 || xp >= nx || yp >= ny || zp >= nz) {
					this_data[x + y * nx + z * nx * ny] = 0;
				}
				else {
					this_data[x + y * nx + z * nx * ny] = tmp_data[xp + yp * nx + zp * nx * ny];
				}
			}
		}
	}

	if( tmp_data ) {
		free(tmp_data);
		tmp_data = 0;
	}

	done_data();
	all_translation += translation;

	EXITFUNC;
}


void EMData::translate(const Vec3f &translation)
{
	ENTERFUNC;

	//if traslation is 0, do nothing
	if( translation[0] == 0.0f && translation[1] == 0.0f && translation[2] == 0.0f ) {
		EXITFUNC;
		return;
	}

	float *this_data = get_data();
	EMData *tmp_emdata = copy();

	int x0, x1, x2;
	if( translation[0] < 0 ) {
		x0 = 0;
		x1 = nx;
		x2 = 1;
	}
	else {
		x0 = nx-1;
		x1 = -1;
		x2 = -1;
	}

	int y0, y1, y2;
	if( translation[1] < 0 ) {
		y0 = 0;
		y1 = ny;
		y2 = 1;
	}
	else {
		y0 = ny-1;
		y1 = -1;
		y2 = -1;
	}

	int z0, z1, z2;
	if( translation[2] < 0 ) {
		z0 = 0;
		z1 = nz;
		z2 = 1;
	}
	else {
		z0 = nz-1;
		z1 = -1;
		z2 = -1;
	}

	if( nz == 1 ) 	//2D translation
	{
		int tx = Util::round(translation[0]);
		int ty = Util::round(translation[1]);
		float ftx = translation[0];
		float fty = translation[1];
		int xp, yp;
		for (int y = y0; y != y1; y += y2) {
			for (int x = x0; x != x1; x += x2) {
				xp = x - tx;
				yp = y - ty;

				if (xp < 0 || yp < 0 || xp >= nx || yp >= ny) {
					this_data[x + y * nx] = 0;
				}
				else {
					float fxp = static_cast<float>(x) - ftx;
					float fyp = static_cast<float>(y) - fty;
					this_data[x + y * nx] = tmp_emdata->sget_value_at_interp(fxp, fyp);
				}
			}
		}
	}
	else 	//3D translation
	{
		int tx = Util::round(translation[0]);
		int ty = Util::round(translation[1]);
		int tz = Util::round(translation[2]);
		float ftx = translation[0];
		float fty = translation[1];
		float ftz = translation[2];
		int xp, yp, zp;
		for (int z = z0; z != z1; z += z2) {
			for (int y = y0; y != y1; y += y2) {
				for (int x = x0; x != x1; x += x2) {
					xp = x - tx;
					yp = y - ty;
					zp = z - tz;
					if (xp < 0 || yp < 0 || zp<0 || xp >= nx || yp >= ny || zp >= nz) {
						this_data[x + y * nx] = 0;
					}
					else {
						float fxp = static_cast<float>(x) - ftx;
						float fyp = static_cast<float>(y) - fty;
						float fzp = static_cast<float>(z) - ftz;
						this_data[x + y * nx + z * nx * ny] = tmp_emdata->sget_value_at_interp(fxp, fyp, fzp);
					}
				}
			}
		}

	}

	if( tmp_emdata ) {
		delete tmp_emdata;
		tmp_emdata = 0;
	}
	done_data();
	update();
	all_translation += translation;
	EXITFUNC;
}


void EMData::rotate(float az, float alt, float phi)
{
	Transform3D t(az, alt, phi);
	rotate_translate(t);
}


void EMData::rotate(const Transform3D & t)
{
	rotate_translate(t);
}


void EMData::rotate_translate(float az, float alt, float phi, float dx, float dy, float dz)
{
	Transform3D t(Vec3f(dx, dy, dz),  az, alt, phi);
	rotate_translate(t);
}


void EMData::rotate_translate(float az, float alt, float phi, float dx, float dy,
							  float dz, float pdx, float pdy, float pdz)
{
	Transform3D t(Vec3f(dx, dy, dz), Vec3f(pdx,pdy,pdz), az, alt, phi);
	rotate_translate(t);
}



void EMData::rotate_translate(const Transform3D & xform)
{
	ENTERFUNC;

#ifdef DEBUG	
	std::cout << "start rotate_translate..." << std::endl;
#endif

	float scale = xform.get_scale();
	Vec3f dcenter = xform.get_center();
	Vec3f translation = xform.get_posttrans();
	Dict rotation = xform.get_rotation(Transform3D::EMAN);

#ifdef DEBUG
	vector<string> keys = rotation.keys();
	vector<string>::const_iterator it;
	for(it=keys.begin(); it!=keys.end(); ++it) {
//		std::cout << *it << " : " << rotation[*it] << std::endl;
		std::cout << *it << " : " << (float)rotation.get(*it) << std::endl;
	} 
#endif

	int nx2 = nx;
	int ny2 = ny;
	float inv_scale = 1.0f;

	if (scale != 0) {
		inv_scale = 1.0f / scale;
	}

	float *src_data = 0;
	float *des_data = 0;

	src_data = get_data();
	des_data = (float *) malloc(nx * ny * nz * sizeof(float));

	if (nz == 1) {
		float mx0 = inv_scale * cos((M_PI/180.0f)*(float)rotation["phi"]);
		float mx1 = inv_scale * sin((M_PI/180.0f)*(float)rotation["phi"]);

		float x2c = nx / 2.0f - dcenter[0] - translation[0];
		float y2c = ny / 2.0f - dcenter[1] - translation[1];
		float y = -ny / 2.0f + dcenter[0];

		for (int j = 0; j < ny; j++, y += 1.0f) {
			float x = -nx / 2.0f + dcenter[1];

			for (int i = 0; i < nx; i++, x += 1.0f) {
				float x2 = (mx0 * x + mx1 * y) + x2c;
				float y2 = (-mx1 * x + mx0 * y) + y2c;

				if (x2 < 0 || x2 > nx2 - 1 || y2 < 0 || y2 > ny2 - 1) {
					des_data[i + j * nx] = 0;
				}
				else {
					int ii = Util::fast_floor(x2);
					int jj = Util::fast_floor(y2);
					int k0 = ii + jj * nx;
					int k1 = k0 + 1;
					int k2 = k0 + nx + 1;
					int k3 = k0 + nx;

					if (ii == nx2 - 1) {
						k1--;
						k2--;
					}
					if (jj == ny2 - 1) {
						k2 -= nx2;
						k3 -= nx2;
					}

					float t = x2 - ii;
					float u = y2 - jj;
					float tt = 1 - t;
					float uu = 1 - u;

					float p0 = src_data[k0] * tt * uu;
					float p1 = src_data[k1] * t * uu;
					float p3 = src_data[k3] * tt * u;
					float p2 = src_data[k2] * t * u;

					des_data[i + j * nx] = p0 + p1 + p2 + p3;
				}
			}
		}
	}

	else if (nx == (nx / 2 * 2 + 1) && nx == ny && (2 * nz - 1) == nx) { // square, odd image, with nz= (nx+1)/2
		// make sure this is right
		Transform3D mx = xform;
		mx.set_scale(inv_scale);
		int nxy = nx * ny;
		int l = 0;

		for (int k = 0; k < nz; k++) {
			for (int j = -ny / 2; j < ny - ny / 2; j++) {
				for (int i = -nx / 2; i < nx - nx / 2; i++, l++) {
					float x2 = mx[0][0] * i + mx[0][1] * j + mx[0][2] * k + nx / 2;
					float y2 = mx[1][0] * i + mx[1][1] * j + mx[1][2] * k + ny / 2;
					float z2 = mx[2][0] * i + mx[2][1] * j + mx[2][2] * k + 0 / 2;

					if (x2 >= 0 && y2 >= 0 && z2 > -(nz - 1) && x2 < nx && y2 < ny && z2 < nz - 1) {
						if (z2 < 0) {
							x2 = nx - 1 - x2;
							z2 = -z2;
						}

						int x = Util::fast_floor(x2);
						int y = Util::fast_floor(y2);
						int z = Util::fast_floor(z2);

						float t = x2 - x;
						float u = y2 - y;
						float v = z2 - z;

						int ii = (int) (x + y * nx + z * nxy);

						des_data[l] += Util::trilinear_interpolate(src_data[ii], src_data[ii + 1],
																   src_data[ii + nx],
																   src_data[ii + nx + 1],
																   src_data[ii + nx * ny],
																   src_data[ii + nxy + 1],
																   src_data[ii + nxy + nx],
																   src_data[ii + nxy + nx + 1], t,
																   u, v);
					}
				}
			}
		}
	}
	else {
#ifdef DEBUG
		std::cout << "I am in this case..." << std::endl;
#endif

		Transform3D mx = xform;
		mx.set_scale(inv_scale);

		Vec3f dcenter2 = Vec3f((float)nx,(float)ny,(float)nz)/(-2.0f) + dcenter;
		Vec3f v4 = dcenter2 * mx  - dcenter2 - translation; // verify this

#ifdef DEBUG
		std::cout << v4[0] << " " << v4[1]<< " " << v4[2]<< " "
			<< dcenter2[0]<< " "<< dcenter2[1]<< " "<< dcenter2[2] << std::endl;
#endif

		int nxy = nx * ny;
		int l = 0;

		for (int k = 0; k < nz; k++) {
			Vec3f v3 = v4;

			for (int j = 0; j < ny; j++) {
				Vec3f v2 = v3;

                
				for (int i = 0; i < nx; i++, l++) {

					if (v2[0] < 0 || v2[1] < 0 || v2[2] < 0 ||
						v2[0] >= nx - 1 || v2[1] >= ny - 1 || v2[2] >= nz - 1) {
						des_data[l] = 0;
#ifdef DEBUG
                		std::cout << l <<" weird if statement..." << std::endl;
                		std::cout << v2[0] << " "<< v2[1] << " " << v2[2] << " "  << std::endl;
#endif
					}
					else {
						int x = Util::fast_floor(v2[0]);
						int y = Util::fast_floor(v2[1]);
						int z = Util::fast_floor(v2[2]);
						Vec3f tuv = v2 - Vec3f((float)x,(float)y,(float)z);
						int ii = x + y * nx + z * nxy;

						des_data[l] = Util::trilinear_interpolate(src_data[ii],
							  src_data[ii + 1],
							  src_data[ii + nx],
							  src_data[ii + nx + 1],
							  src_data[ii + nx * ny],
							  src_data[ii + nxy + 1],
							  src_data[ii + nxy + nx],
							  src_data[ii + nxy + nx + 1],
							  tuv[0],
							  tuv[1],
							  tuv[2]);
#ifdef DEBUG
						std::cout << src_data[ii] << std::endl;
#endif
					}

					v2 += mx.get_matrix3_col(0);
				}
				v3 += mx.get_matrix3_col(1);
			}
			v4 += mx.get_matrix3_col(2); //  or should it be row?   PRB April 2005
		}
	}

	if( rdata )
	{
		free(rdata);
		rdata = 0;
	}
	rdata = des_data;

	scale_pixel(inv_scale);

	attr_dict["origin_row"] = (float) attr_dict["origin_row"] * inv_scale;
	attr_dict["origin_col"] = (float) attr_dict["origin_col"] * inv_scale;
	attr_dict["origin_sec"] = (float) attr_dict["origin_sec"] * inv_scale;

	done_data();
	update();


	all_translation += translation;
	EXITFUNC;
}


void EMData::rotate_x(int dx)
{
	ENTERFUNC;

	if (get_ndim() > 2) {
		throw ImageDimensionException("no 3D image");
	}

	float *tmp = new float[nx];
	size_t row_size = nx * sizeof(float);

	for (int y = 0; y < ny; y++) {
		int y_nx = y * nx;
		for (int x = 0; x < nx; x++) {
			tmp[x] = rdata[y_nx + (x + dx) % nx];
		}
		memcpy(&rdata[y_nx], tmp, row_size);
	}

	done_data();
	if( tmp )
	{
		delete[]tmp;
		tmp = 0;
	}
	EXITFUNC;
}


void EMData::rotate_180()
{
	ENTERFUNC;

	if (nx != ny) {
		throw ImageFormatException("non-square image");
	}

	if (get_ndim() != 2) {
		throw ImageDimensionException("2D only");
	}

	float *d = get_data();

	for (int x = 1; x < nx; x++) {
		int y = 0;
		for (y = 1; y < ny; y++) {
			if (x == nx / 2 && y == ny / 2) {
				break;
			}
			int i = x + y * nx;
			int k = nx - x + (ny - y) * nx;

			float t = d[i];
			d[i] = d[k];
			d[k] = t;
		}
		if (x == nx / 2 && y == ny / 2) {
			break;
		}
	}

	done_data();
	EXITFUNC;
}


double EMData::dot_rotate_translate(EMData * with, float dx, float dy, float da)
{
	ENTERFUNC;

	if (!EMUtil::is_same_size(this, with)) {
		LOGERR("images not same size");
		throw ImageFormatException("images not same size");
	}

	if (get_ndim() == 3) {
		LOGERR("1D/2D Images only");
		throw ImageDimensionException("1D/2D only");
	}

	float *this_data = 0;

	this_data = get_data();

	float *with_data = with->get_data();
	float mx0 = cos(da);
	float mx1 = sin(da);
	float y = -ny / 2.0f;
	float my0 = mx0 * (-nx / 2.0f - 1.0f) + nx / 2.0f - dx;
	float my1 = -mx1 * (-nx / 2.0f - 1.0f) + ny / 2.0f - dy;
	double result = 0;

	for (int j = 0; j < ny; j++) {
		float x2 = my0 + mx1 * y;
		float y2 = my1 + mx0 * y;

		int ii = Util::fast_floor(x2);
		int jj = Util::fast_floor(y2);
		float t = x2 - ii;
		float u = y2 - jj;

		for (int i = 0; i < nx; i++) {
			t += mx0;
			u -= mx1;

			if (t >= 1.0f) {
				ii++;
				t -= 1.0f;
			}

			if (u >= 1.0f) {
				jj++;
				u -= 1.0f;
			}

			if (t < 0) {
				ii--;
				t += 1.0f;
			}

			if (u < 0) {
				jj--;
				u += 1.0f;
			}

			if (ii >= 0 && ii <= nx - 2 && jj >= 0 && jj <= ny - 2) {
				int k0 = ii + jj * nx;
				int k1 = k0 + 1;
				int k2 = k0 + nx + 1;
				int k3 = k0 + nx;

				float tt = 1 - t;
				float uu = 1 - u;

				result += (this_data[k0] * tt * uu + this_data[k1] * t * uu +
						   this_data[k2] * t * u + this_data[k3] * tt * u) * with_data[i + j * nx];
			}
		}
		y += 1.0f;
	}

	EXITFUNC;
	return result;
}


EMData *EMData::little_big_dot(EMData * with, bool do_sigma)
{
	ENTERFUNC;

	if (get_ndim() > 2) {
		throw ImageDimensionException("1D/2D only");
	}

	EMData *ret = copy_head();
	ret->to_zero();

	int nx2 = with->get_xsize();
	int ny2 = with->get_ysize();
	float em = with->get_edge_mean();

	float *data = get_data();
	float *with_data = with->get_data();
	float *ret_data = ret->get_data();

	float sum2 = (Util::square((float)with->get_attr("sigma")) +
				  Util::square((float)with->get_attr("mean")));

	if (do_sigma) {
		for (int j = ny2 / 2; j < ny - ny2 / 2; j++) {
			for (int i = nx2 / 2; i < nx - nx2 / 2; i++) {
				float sum = 0;
				float sum1 = 0;
				float summ = 0;
				int k = 0;

				for (int jj = j - ny2 / 2; jj < j + ny2 / 2; jj++) {
					for (int ii = i - nx2 / 2; ii < i + nx2 / 2; ii++) {
						int l = ii + jj * nx;
						sum1 += Util::square(data[l]);
						summ += data[l];
						sum += data[l] * with_data[k];
						k++;
					}
				}
				float tmp_f1 = (sum1 / 2.0f - sum) / (nx2 * ny2);
				float tmp_f2 = Util::square((float)with->get_attr("mean") -
											summ / (nx2 * ny2));
				ret_data[i + j * nx] = sum2 + tmp_f1 - tmp_f2;
			}
		}
	}
	else {
		for (int j = ny2 / 2; j < ny - ny2 / 2; j++) {
			for (int i = nx2 / 2; i < nx - nx2 / 2; i++) {
				float eml = 0;
				float dot = 0;
				float dot2 = 0;

				for (int ii = i - nx2 / 2; ii < i + nx2 / 2; ii++) {
					eml += data[ii + (j - ny2 / 2) * nx] + data[ii + (j + ny2 / 2 - 1) * nx];
				}

				for (int jj = j - ny2 / 2; jj < j + ny2 / 2; jj++) {
					eml += data[i - nx2 / 2 + jj * nx] + data[i + nx2 / 2 - 1 + jj * nx];
				}

				eml /= (nx2 + ny2) * 2.0f;
				int k = 0;

				for (int jj = j - ny2 / 2; jj < j + ny2 / 2; jj++) {
					for (int ii = i - nx2 / 2; ii < i + nx2 / 2; ii++) {
						dot += (data[ii + jj * nx] - eml) * (with_data[k] - em);
						dot2 += Util::square(data[ii + jj * nx] - eml);
						k++;
					}
				}

				dot2 = sqrt(dot2);

				if (dot2 == 0) {
					ret_data[i + j * nx] = 0;
				}
				else {
					ret_data[i + j * nx] = dot / (nx2 * ny2 * dot2 * (float)with->get_attr("sigma"));
				}
			}
		}
	}

	ret->done_data();

	EXITFUNC;
	return ret;
}


EMData *EMData::do_radon()
{
	ENTERFUNC;

	if (get_ndim() != 2) {
		throw ImageDimensionException("2D only");
	}

	if (nx != ny) {
		throw ImageFormatException("square image only");
	}

	EMData *result = new EMData();
	result->set_size(nx, ny, 1);
	result->to_zero();
	float *result_data = result->get_data();

	EMData *this_copy = this;
	this_copy = copy();

	for (int i = 0; i < nx; i++) {
		this_copy->rotate(M_PI * 2.0f * i / nx, 0, 0);

		float *copy_data = this_copy->get_data();

		for (int y = 0; y < nx; y++) {
			for (int x = 0; x < nx; x++) {
				if (Util::square(x - nx / 2) + Util::square(y - nx / 2) <= nx * nx / 4) {
					result_data[i + y * nx] += copy_data[x + y * nx];
				}
			}
		}

		this_copy->done_data();
	}

	result->done_data();

	if( this_copy )
	{
		delete this_copy;
		this_copy = 0;
	}

	EXITFUNC;
	return result;
}


EMData *EMData::calc_ccf(EMData * with, fp_flag fpflag) {
	if( with == 0 ) {
		return autocorrelation(this,fpflag);
	}
	else if ( with == this ){
		return autocorrelation(this,fpflag);
	}
	else {
		return correlation(this, with, fpflag);
	}
}


EMData *EMData::calc_ccfx(EMData * with, int y0, int y1, bool no_sum)
{
	ENTERFUNC;

	if (!with) {
		LOGERR("NULL 'with' image. ");
		throw NullPointerException("NULL input image");
	}

	if (!EMUtil::is_same_size(this, with)) {
		LOGERR("images not same size: (%d,%d,%d) != (%d,%d,%d)",
			   nx, ny, nz,
			   with->get_xsize(), with->get_ysize(), with->get_zsize());
		throw ImageFormatException("images not same size");
	}
	if (get_ndim() > 2) {
		LOGERR("2D images only");
		throw ImageDimensionException("2D images only");
	}

	if (y1 <= y0) {
		y1 = ny;
	}

	if (y0 >= y1) {
		y0 = 0;
	}

	if (y0 < 0) {
		y0 = 0;
	}

	if (y1 > ny) {
		y1 = ny;
	}

	EMData *cf = new EMData();
	if (no_sum) {
		cf->set_size(nx, y1 - y0 , 1);
	}
	else {
		cf->set_size(nx, 1, 1);
	}

	cf->set_attr("label", "CCFx");
	cf->set_path("/tmp/eman.ccf");


	if (no_sum) {
		float *cfd = cf->get_data();
		float *with_data = with->get_data();

		for (int y = y0; y < y1; y++) {
			int cur_y = y * nx;

			for (int x = 0; x < nx; x++) {
				float dot = 0;
				for (int i = 0; i < nx; i++) {
					int k1 = (i + x) % nx + cur_y;
					dot += rdata[i + cur_y] * with_data[k1];
				}
				cfd[x + (y - y0) * nx] = dot;
			}
		}

		cf->done_data();
		return cf;
	}
	else {
		float *f1 = (float *) calloc(nx, sizeof(float));
		float *f2 = (float *) calloc(nx, sizeof(float));

		float *cfd = cf->get_data();
		float *d1 = get_data();
		float *d2 = with->get_data();
		size_t row_size = nx * sizeof(float);

		if (!is_complex_x()) {
			for (int j = 0; j < ny; j++) {
				EMfft::real_to_complex_1d(d1 + j * nx, f1, nx);
				memcpy(d1 + j * nx, f1, row_size);
			}

			set_complex_x(true);
		}
		if (!with->is_complex_x()) {
			for (int j = 0; j < with->get_ysize(); j++) {
				EMfft::real_to_complex_1d(d2 + j * nx, f2, nx);
				memcpy(d2 + j * nx, f2, row_size);
			}

			with->set_complex_x(true);
		}

		for (int j = y0; j < y1; j++) {
			float *f1a = d1 + j * nx;
			float *f2a = d2 + j * nx;

			f1[0] = f1a[0] * f2a[0];
			f1[nx / 2] = f1a[nx / 2] * f2a[nx / 2];

			for (int i = 1; i < nx / 2; i++) {
				float re1 = f1a[i];
				float re2 = f2a[i];
				float im1 = f1a[nx - i];
				float im2 = f2a[nx - i];

				f1[i] = re1 * re2 + im1 * im2;
				f1[nx - i] = im1 * re2 - re1 * im2;
			}

			EMfft::complex_to_real_1d(f1, f2, nx);

			if (no_sum) {
				for (int i = 0; i < nx; i++) {
					cfd[i + nx * (j - y0)] = f2[i];
				}
			}
			else {
				for (int i = 0; i < nx; i++) {
					cfd[i] += f2[i];
				}
			}
		}

		if( f1 )
		{
			free(f1);
			f1 = 0;
		}
		if( f2 )
		{
			free(f2);
			f2 = 0;
		}
	}

	cf->done_data();
	done_data();
	with->done_data();


	EXITFUNC;
	return cf;
}


EMData *EMData::make_rotational_footprint(bool premasked, bool unwrap)
{
	ENTERFUNC;

	static EMData obj_filt;
	EMData *rfp=NULL;
	EMData* filt = &obj_filt;
	filt->set_complex(true);

	if (nx & 1) {
		LOGERR("even image xsize only");
		throw ImageFormatException("even image xsize only");
	}

	int cs = (((nx * 7 / 4) & 0xfffff8) - nx) / 2;

	EMData *tmp2 = 0;
	Region r1;
	if (nz == 1) {
		r1 = Region(-cs, -cs, nx + 2 * cs, ny + 2 * cs);
	}
	else {
		r1 = Region(-cs, -cs, -cs, nx + 2 * cs, ny + 2 * cs, nz + 2 * cs);
	}
	tmp2 = get_clip(r1);

	if (!premasked) {
		tmp2->process_inplace("eman1.mask.sharp", Dict("outer_radius", nx / 2, "value", 0));
	}

	if (filt->get_xsize() != tmp2->get_xsize() + 2 || filt->get_ysize() != tmp2->get_ysize() ||
		filt->get_zsize() != tmp2->get_zsize()) {
		filt->set_size(tmp2->get_xsize() + 2, tmp2->get_ysize(), tmp2->get_zsize());
		filt->to_one();

		filt->process_inplace("eman1.filter.highpass.gaussian", Dict("highpass", 1.5/nx));
	}

	EMData *tmp = tmp2->calc_mutual_correlation(tmp2, true, filt);
	if( tmp2 )
	{
		delete tmp2;
		tmp2 = 0;
	}

	Region r2;
	if (nz == 1) {
		r2 = Region(cs - nx / 4, cs - ny / 4, nx * 3 / 2, ny * 3 / 2);
	}
	else {
		r2 = Region(cs - nx / 4, cs - ny / 4, cs - nz / 4, nx * 3 / 2, ny * 3 / 2, nz * 3 / 2);
	}
	tmp2 = tmp->get_clip(r2);
	rfp = tmp2;

	if( tmp )
	{
		delete tmp;
		tmp = 0;
	}

	EMData * result = rfp;

	if (nz == 1) {
		if (!unwrap) {
			tmp2->process_inplace("eman1.mask.sharp", Dict("outer_radius", -1, "value", 0));
			rfp = 0;
			result = tmp2;
		}
		else {
			rfp = tmp2->unwrap();
			if( tmp2 )
			{
				delete tmp2;
				tmp2 = 0;
			}
			result = rfp;
		}
	}

	EXITFUNC;
	return result;
}


EMData *EMData::make_footprint() 
{
	//EMData *ccf=calc_ccf(this);
	EMData *ccf=calc_mutual_correlation(this);
	ccf->process_inplace("eman1.xform.phaseorigin");
	ccf->process_inplace("eman1.normalize.edgemean");
	EMData *un=ccf->unwrap();
	EMData *tmp=un->get_clip(Region(0,4,un->get_xsize()/2,un->get_ysize()-6));	// 4 and 6 are empirical
	EMData *cx=tmp->calc_ccfx(tmp,0,-1,1);
	delete ccf;
	delete un;
	delete tmp;
	return cx;
}


EMData *EMData::calc_mutual_correlation(EMData * with, bool tocorner, EMData * filter)
{
	ENTERFUNC;

	if (with && !EMUtil::is_same_size(this, with)) {
		LOGERR("images not same size");
		throw ImageFormatException( "images not same size");
	}

	EMData *this_fft = 0;
	this_fft = do_fft();

	if (!this_fft) {
		LOGERR("FFT returns NULL image");
		throw NullPointerException("FFT returns NULL image");
	}

	this_fft->ap2ri();
	EMData *cf = 0;

	if (with) {
		cf = with->do_fft();
		if (!cf) {
			LOGERR("FFT returns NULL image");
			throw NullPointerException("FFT returns NULL image");
		}
		cf->ap2ri();
	}
	else {
		cf = this_fft->copy();
	}

	if (filter) {
		if (!EMUtil::is_same_size(filter, cf)) {
			LOGERR("improperly sized filter");
			throw ImageFormatException("improperly sized filter");
		}

		cf->mult(*filter);
		this_fft->mult(*filter);
	}

	float *rdata1 = this_fft->get_data();
	float *rdata2 = cf->get_data();
	int this_fft_size = this_fft->get_xsize() * this_fft->get_ysize() * this_fft->get_zsize();

	if (with == this) {
		for (int i = 0; i < this_fft_size; i += 2) {
			rdata2[i] = sqrt(rdata1[i] * rdata2[i] + rdata1[i + 1] * rdata2[i + 1]);
			rdata2[i + 1] = 0;
		}

		this_fft->done_data();
		cf->done_data();
	}
	else {
		for (int i = 0; i < this_fft_size; i += 2) {
			rdata2[i] = (rdata1[i] * rdata2[i] + rdata1[i + 1] * rdata2[i + 1]);
			rdata2[i + 1] = (rdata1[i + 1] * rdata2[i] - rdata1[i] * rdata2[i + 1]);
		}

		this_fft->done_data();
		cf->done_data();
		rdata1 = cf->get_data();

		for (int i = 0; i < this_fft_size; i += 2) {
			float t = Util::square(rdata1[i]) + Util::square(rdata1[i + 1]);
			if (t != 0) {
				t = pow(t, (float) 0.25);
				rdata1[i] /= t;
				rdata1[i + 1] /= t;
			}
		}
		cf->done_data();
	}

	if (tocorner) {
		cf->process_inplace("eman1.xform.phaseorigin");
	}

	EMData *f2 = cf->do_ift();

	if( cf )
	{
		delete cf;
		cf = 0;
	}

	if( this_fft )
	{
		delete this_fft;
		this_fft = 0;
	}

	f2->set_attr("label", "MCF");
	f2->set_path("/tmp/eman.mcf");

	EXITFUNC;
	return f2;
}











vector < float > EMData::calc_hist(int hist_size, float histmin, float histmax)
{
	ENTERFUNC;

	static size_t prime[] = { 1, 3, 7, 11, 17, 23, 37, 59, 127, 253, 511 };

	if (histmin == histmax) {
		histmin = get_attr("minimum");
		histmax = get_attr("maximum");
	}

	vector <float> hist(256, 0.0);

	int p0 = 0;
	int p1 = 0;
	size_t size = nx * ny * nz;
	if (size < 300000) {
		p0 = 0;
		p1 = 0;
	}
	else if (size < 2000000) {
		p0 = 2;
		p1 = 3;
	}
	else if (size < 8000000) {
		p0 = 4;
		p1 = 6;
	}
	else {
		p0 = 7;
		p1 = 9;
	}

	if (is_complex() && p0 > 0) {
		p0++;
		p1++;
	}

	size_t di = 0;
	float norm = 0;
	size_t n = hist.size();

	for (int k = p0; k <= p1; ++k) {
		if (is_complex()) {
			di = prime[k] * 2;
		}
		else {
			di = prime[k];
		}

		norm += (float)size / (float) di;
		float w = (float)n / (histmax - histmin);

		for(size_t i=0; i<=size-di; i += di) {
			int j = Util::round((rdata[i] - histmin) * w);
			if (j >= 0 && j < (int) n) {
				hist[j] += 1;
			}
		}
	}

	for (size_t i = 0; i < hist.size(); ++i) {
		if (norm != 0) {
			hist[i] = hist[i] / norm;
		}
	}
	
	return hist;
	
	EXITFUNC;
}





vector<float> EMData::calc_az_dist(int n, float a0, float da, float rmin, float rmax)
{
	ENTERFUNC;

	if (get_ndim() > 2) {
		throw ImageDimensionException("no 3D image");
	}

	float *yc = new float[n];

	vector<float>	vd(n);
	for (int i = 0; i < n; i++) {
		yc[i] = 0.00001f;
	}

	if (is_complex()) {
		int c = 0;
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x += 2, c += 2) {
				float x1 = x / 2.0f;
				float y1 = y - ny / 2.0f;
				float r = (float)hypot(x1, y1);

				if (r >= rmin && r <= rmax) {
					float a = 0;

					if (y != ny / 2 || x != 0) {
						a = (atan2(y1, x1) - a0) / da;
					}

					int i = static_cast < int >(floor(a));
					a -= i;

					if (i == 0) {
						vd[0] += rdata[c] * (1.0f - a);
						yc[0] += (1.0f - a);
					}
					else if (i == n - 1) {
						vd[n - 1] += rdata[c] * a;
						yc[n - 1] += a;
					}
					else if (i > 0 && i < (n - 1)) {
						float h = 0;
						if (is_ri()) {
							h = (float)hypot(rdata[c], rdata[c + 1]);
						}
						else {
							h = rdata[c];
						}

						vd[i] += h * (1.0f - a);
						yc[i] += (1.0f - a);
						vd[i + 1] += h * a;
						yc[i + 1] += a;
					}
				}
			}
		}
	}
	else {
		int c = 0;
		float half_nx = (nx - 1) / 2.0f;
		float half_ny = (ny - 1) / 2.0f;

		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++, c++) {
				float y1 = y - half_ny;
				float x1 = x - half_nx;
				float r = (float)hypot(x1, y1);

				if (r >= rmin && r <= rmax) {
					float a = 0;
					if (x1 != 0 || y1 != 0) {
						a = atan2(y1, x1);
						if (a < 0) {
							a += M_PI * 2;
						}
					}

					a = (a - a0) / da;
					int i = static_cast < int >(floor(a));
					a -= i;

					if (i == 0) {
						vd[0] += rdata[c] * (1.0f - a);
						yc[0] += (1.0f - a);
					}
					else if (i == n - 1) {
						vd[n - 1] += rdata[c] * a;
						yc[n - 1] += (a);
					}
					else if (i > 0 && i < (n - 1)) {
						vd[i] += rdata[c] * (1.0f - a);
						yc[i] += (1.0f - a);
						vd[i + 1] += rdata[c] * a;
						yc[i + 1] += a;
					}
				}
			}
		}
	}


	for (int i = 0; i < n; i++) {
		vd[i] /= yc[i];
	}

	if( yc )
	{
		delete[]yc;
		yc = 0;
	}
	
	return vd;

	EXITFUNC;
}


EMData *EMData::unwrap(int r1, int r2, int xs, int dx, int dy, bool do360)
{
	ENTERFUNC;

	if (get_ndim() != 2) {
		throw ImageDimensionException("2D image only");
	}

	int p = 1;
	if (do360) {
		p = 2;
	}

	EMData *ret = new EMData();

	if (xs < 1) {
		xs = (int) floor(p * M_PI * ny / 4);
		xs -= xs % 8;
		if (xs<=8) xs=16;
//		xs = Util::calc_best_fft_size(xs);
	}

	if (r1 < 0) {
		r1 = 4;
	}

	int rr = ny / 2 - 2 - (int) floor(hypot(dx, dy));
	rr-=rr%2;
	if (r2 <= r1 || r2 > rr) {
		r2 = rr;
	}

	ret->set_size(xs, r2 - r1, 1);
	float *d = get_data();
	float *dd = ret->get_data();

	for (int x = 0; x < xs; x++) {
		float si = sin(x * M_PI * p / xs);
		float co = cos(x * M_PI * p / xs);

		for (int y = 0; y < r2 - r1; y++) {
			float xx = (y + r1) * co + nx / 2 + dx;
			float yy = (y + r1) * si + ny / 2 + dy;
			float t = xx - floor(xx);
			float u = yy - floor(yy);
			int k = (int) floor(xx) + (int) (floor(yy)) * nx;
			dd[x + y * xs] =
				Util::bilinear_interpolate(d[k], d[k + 1], d[k + nx + 1], d[k + nx], t,u) * (y + r1);
		}
	}
	done_data();
	ret->done_data();

	EXITFUNC;
	return ret;
}


void EMData::mean_shrink(float shrink_factor0)
{
	ENTERFUNC;
	int shrink_factor = int(shrink_factor0);

	if (shrink_factor0 <= 1.0F || ((shrink_factor0 != shrink_factor) && (shrink_factor0 != 1.5F) ) ) {
		throw InvalidValueException(shrink_factor0,
									"mean shrink: shrink factor must be >1 integer or 1.5");
	}

/*	if ((nx % shrink_factor != 0) || (ny % shrink_factor != 0) ||
		(nz > 1 && (nz % shrink_factor != 0))) {
		throw InvalidValueException(shrink_factor,
									"Image size not divisible by shrink factor");
	}*/

	// here handle the special averaging by 1.5 for 2D case
	if (shrink_factor0==1.5 ) {
		if (nz > 1 ) throw InvalidValueException(shrink_factor0, "mean shrink: only support 2D images for shrink factor = 1.5");

		int shrinked_nx = (int(nx / 1.5)+1)/2*2;	// make sure the output size is even
		int shrinked_ny = (int(ny / 1.5)+1)/2*2;
		int nx0 = nx, ny0 = ny;	// the original size

		EMData* orig = copy();
		set_size(shrinked_nx, shrinked_ny, 1);	// now nx = shrinked_nx, ny = shrinked_ny
		to_zero();

		float *data = get_data(), *data0 = orig->get_data();

		for (int j = 0; j < ny; j++) {
			int jj = int(j * 1.5);
			float jw0 = 1.0F, jw1 = 0.5F;	// 3x3 -> 2x2, so each new pixel should have 2.25 of the old pixels
			if ( j%2 ) {
				jw0 = 0.5F;
				jw1 = 1.0F;
			}
			for (int i = 0; i < nx; i++) {
				int ii = int(i * 1.5);
				float iw0 = 1.0F, iw1 = 0.5F;
				float w = 0.0F;

				if ( i%2 ) {
					iw0 = 0.5F;
					iw1 = 1.0F;
				}
				if ( jj < ny0 ) {
					if ( ii < nx0 ) {
						data[j * nx + i] = data0[ jj * nx0 + ii ] * jw0 * iw0 ;
						w += jw0 * iw0 ;
						if ( ii+1 < nx0 ) {
							data[j * nx + i] += data0[ jj * nx0 + ii + 1] * jw0 * iw1;
							w += jw0 * iw1;
						}
					}
					if ( jj +1 < ny0 ) {
						if ( ii < nx0 ) {
							data[j * nx + i] += data0[ (jj+1) * nx0 + ii ] * jw1 * iw0;
							w += jw1 * iw0;
							if ( ii+1 < nx0 ) {
								data[j * nx + i] += data0[ (jj+1) * nx0 + ii + 1] * jw1 * iw1;
								w += jw1 * iw1;
							}
						}
					}
				}
				if ( w>0 ) data[j * nx + i] /= w;
			}
		}
		orig->done_data();
		if( orig )
		{
			delete orig;
			orig = 0;
		}
		done_data();
		update();

		return;
	}


	int shrinked_nx = nx / shrink_factor;
	int shrinked_ny = ny / shrink_factor;
	int shrinked_nz = 1;


	int threed_shrink_factor = shrink_factor * shrink_factor;
	int z_shrink_factor = 1;

	if (nz > 1) {
		shrinked_nz = nz / shrink_factor;
		threed_shrink_factor *= shrink_factor;
		z_shrink_factor = shrink_factor;
	}

	float *data = get_data();
	int nxy = nx * ny;
	int shrinked_nxy = shrinked_nx * shrinked_ny;

	for (int k = 0; k < shrinked_nz; k++) {
		int k_min = k * shrink_factor;
		int k_max = k * shrink_factor + z_shrink_factor;
		int cur_k = k * shrinked_nxy;

		for (int j = 0; j < shrinked_ny; j++) {
			int j_min = j * shrink_factor;
			int j_max = j * shrink_factor + shrink_factor;
			int cur_j = j * shrinked_nx + cur_k;

			for (int i = 0; i < shrinked_nx; i++) {
				int i_min = i * shrink_factor;
				int i_max = i * shrink_factor + shrink_factor;

				float sum = 0;
				for (int kk = k_min; kk < k_max; kk++) {
					int cur_kk = kk * nxy;

					for (int jj = j_min; jj < j_max; jj++) {
						int cur_jj = jj * nx + cur_kk;
						for (int ii = i_min; ii < i_max; ii++) {
							sum += data[ii + cur_jj];
						}
					}
				}
				data[i + cur_j] = sum / threed_shrink_factor;
			}
		}
	}

	done_data();
	set_size(shrinked_nx, shrinked_ny, shrinked_nz);
	scale_pixel((float)shrink_factor);
	EXITFUNC;
}


void EMData::median_shrink(int shrink_factor)
{
	ENTERFUNC;

	if (shrink_factor <= 1) {
		throw InvalidValueException(shrink_factor,
									"median shrink: shrink factor must > 1");
	}

	if ((nx % shrink_factor != 0) || (ny % shrink_factor != 0) ||
		(nz > 1 && (nz % shrink_factor != 0))) {
		throw InvalidValueException(shrink_factor,
									"Image size not divisible by shrink factor");
	}

	int threed_shrink_factor = shrink_factor * shrink_factor;
	int size = nx * ny;
	int nx_old = nx;
	int ny_old = ny;

	int shrinked_nx = nx / shrink_factor;
	int shrinked_ny = ny / shrink_factor;
	int shrinked_nz = 1;

	int z_shrink_factor = 1;

	if (nz > 1) {
		threed_shrink_factor *= shrink_factor;
		size *= nz;
		shrinked_nz = nz / shrink_factor;
		z_shrink_factor = shrink_factor;
	}

	float *mbuf = new float[threed_shrink_factor];
	float *data_copy = new float[size];

	memcpy(data_copy, get_data(), size * sizeof(float));
	set_size(shrinked_nx, shrinked_ny, shrinked_nz);
	scale_pixel((float)shrink_factor);

	int nxy_old = nx_old * ny_old;
	int nxy_new = nx * ny;

	for (int l = 0; l < nz; l++) {
		int l_min = l * shrink_factor;
		int l_max = l * shrink_factor + z_shrink_factor;
		int cur_l = l * nxy_new;

		for (int j = 0; j < ny; j++) {
			int j_min = j * shrink_factor;
			int j_max = (j + 1) * shrink_factor;
			int cur_j = j * nx + cur_l;

			for (int i = 0; i < nx; i++) {
				int i_min = i * shrink_factor;
				int i_max = (i + 1) * shrink_factor;

				int k = 0;
				for (int l2 = l_min; l2 < l_max; l2++) {
					int cur_l2 = l2 * nxy_old;

					for (int j2 = j_min; j2 < j_max; j2++) {
						int cur_j2 = j2 * nx_old + cur_l2;

						for (int i2 = i_min; i2 < i_max; i2++) {
							mbuf[k] = data_copy[i2 + cur_j2];
							k++;
						}
					}
				}


				for (k = 0; k < threed_shrink_factor / 2 + 1; k++) {
					for (int i2 = k + 1; i2 < threed_shrink_factor; i2++) {
						if (mbuf[i2] < mbuf[k]) {
							float f = mbuf[i2];
							mbuf[i2] = mbuf[k];
							mbuf[k] = f;
						}
					}
				}

				rdata[i + cur_j] = mbuf[threed_shrink_factor / 2];
			}
		}
	}

	done_data();

	if( data_copy )
	{
		delete[]data_copy;
		data_copy = 0;
	}

	if( mbuf )
	{
		delete[]mbuf;
		mbuf = 0;
	}
	EXITFUNC;
}


// NOTE : x axis is from 0 to 0.5  (Nyquist), and thus properly handles non-square images
// complex only
void EMData::apply_radial_func(float x0, float step, vector < float >array, bool interp)
{
	ENTERFUNC;

	if (!is_complex()) throw ImageFormatException("apply_radial_func requires a complex image");

	int n = static_cast < int >(array.size());

//	printf("%f %f %f\n",array[0],array[25],array[50]);

	ap2ri();

	size_t ndims = get_ndim();

	if (ndims == 2) {
		int k = 0;
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i += 2, k += 2) {
				float r;
				if (j<ny/2) r = (float)hypot(i/(float)(nx*2), j/(float)ny);
				else r = (float)hypot(i/(float)(nx*2), (ny-j)/(float)ny);
				r = (r - x0) / step;

				int l = 0;
				if (interp) {
					l = (int) floor(r);
				}
				else {
					l = (int) floor(r + 1);
				}


				float f = 0;
				if (l >= n - 2) {
					f = array[n - 1];
				}
				else {
					if (interp) {
						r -= l;
						f = (array[l] * (1.0f - r) + array[l + 1] * r);
					}
					else {
						f = array[l];
					}
				}

				rdata[k] *= f;
				rdata[k + 1] *= f;
			}
		}
	}
	else if (ndims == 3) {
		int k = 0;
		for (int m = 0; m < nz; m++) {
			float mnz;
			if (m<nz/2) mnz=m*m/(float)(nz*nz);
			else { mnz=(nz-m)/(float)nz; mnz*=mnz; }

			for (int j = 0; j < ny; j++) {
				float jny;
				if (j<ny/2) jny= j*j/(float)(ny*ny);
				else { jny=(ny-j)/(float)ny; jny*=jny; }

				for (int i = 0; i < nx; i += 2, k += 2) {
					float r = sqrt((i * i / (nx*nx*4.0f)) + jny + mnz);
					r = (r - x0) / step;

					int l = 0;
					if (interp) {
						l = (int) floor(r);
					}
					else {
						l = (int) floor(r + 1);
					}


					float f = 0;
					if (l >= n - 2) {
						f = array[n - 1];
					}
					else {
						if (interp) {
							r -= l;
							f = (array[l] * (1.0f - r) + array[l + 1] * r);
						}
						else {
							f = array[l];
						}
					}

					rdata[k] *= f;
					rdata[k + 1] *= f;
				}
			}
		}

	}

	done_data();
	update();
	EXITFUNC;
}

vector < float >EMData::calc_radial_dist(int n, float x0, float dx, bool inten)
{
	ENTERFUNC;

	vector<float>ret(n);
	vector<float>norm(n);

	int x,y,z,i;
	int step=is_complex()?2:1;

	for (i=0; i<n; i++) ret[i]=norm[i]=0.0;

	// We do 2D separately to avoid the hypot3 call
	if (nz==1) {
		for (y=i=0; y<ny; y++) {
			for (x=0; x<nx; x+=step,i+=step) {
				float r,v;
				if (is_complex()) {
					r=hypot(x/2.0,float(y<ny/2?y:ny-y));		// origin at 0,0; periodic
					if (inten) {
						if (is_ri()) v=hypot(rdata[i],rdata[i+1]);	// real/imag, compute amplitude
						else v=rdata[i];							// amp/phase, just get amp
					} else {
						if (is_ri()) v=rdata[i]*rdata[i]+rdata[i+1]*rdata[i+1];
						else v=rdata[i]*rdata[i];
					}
				}
				else {
					r=hypot(float(x-nx/2),float(y-ny/2));
					if (inten) v=rdata[i]*rdata[i];
					else v=rdata[i];
				}
				r=(r-x0)/dx;
				int f=int(r);	// safe truncation, so floor isn't needed
				r-=float(f);	// r is now the fractional spacing between bins
				if (f>=0 && f<n) {
					ret[f]+=v*(1.0-r);
					norm[f]+=(1.0-r);
					if (f<n-1) {
						ret[f+1]+=v*r;
						norm[f+1]+=r;
					}
				}
			}
		}
	}
	else {
		for (z=i=0; z<nz; z++) {
			for (y=0; y<ny; y++) {
				for (x=0; x<nx; x+=step,i+=step) {
					float r,v;
					if (is_complex()) {
						r=Util::hypot3(x/2,y<ny/2?y:ny-y,z<nz/2?z:nz-z);	// origin at 0,0; periodic
						if (inten) {
							if (is_ri()) v=hypot(rdata[i],rdata[i+1]);	// real/imag, compute amplitude
							else v=rdata[i];							// amp/phase, just get amp
						} else {
							if (is_ri()) v=rdata[i]*rdata[i]+rdata[i+1]*rdata[i+1];
							else v=rdata[i]*rdata[i];
						}
					}
					else {
						r=Util::hypot3(x-nx/2,y-ny/2,z-nz/2);
						if (inten) v=rdata[i]*rdata[i];
						else v=rdata[i];
					}
					r=(r-x0)/dx;
					int f=int(r);	// safe truncation, so floor isn't needed
					r-=float(f);	// r is now the fractional spacing between bins
					if (f>=0 && f<n) {
						ret[f]+=v*(1.0-r);
						norm[f]+=(1.0-r);
						if (f<n-1) {
							ret[f+1]+=v*r;
							norm[f+1]+=r;
						}
					}
				}
			}
		}
	}

	for (i=0; i<n; i++) ret[i]/=norm[i]?norm[i]:1.0;	// Normalize
	EXITFUNC;

	return ret;
}
	
vector < float >EMData::calc_radial_dist(int n, float x0, float dx, int nwedge, bool inten)
{
	ENTERFUNC;

	if (nz > 1) {
		LOGERR("2D images only.");
		throw ImageDimensionException("2D images only");
	}

	vector<float>ret(n*nwedge);
	vector<float>norm(n*nwedge);

	int x,y,i;
	int step=is_complex()?2:1;
	float astep=M_PI*2.0/float(nwedge);

	for (i=0; i<n*nwedge; i++) ret[i]=norm[i]=0.0;

	// We do 2D separately to avoid the hypot3 call
	for (y=i=0; y<ny; y++) {
		for (x=0; x<nx; x+=step,i+=step) {
			float r,v,a;
			if (is_complex()) {
				r=hypot(x/2.0,float(y<ny/2?y:ny-y));		// origin at 0,0; periodic
				a=atan2(float(y<ny/2?y:ny-y),x/2.0f);
				if (inten) {
					if (is_ri()) v=hypot(rdata[i],rdata[i+1]);	// real/imag, compute amplitude
					else v=rdata[i];							// amp/phase, just get amp
				} else {
					if (is_ri()) v=rdata[i]*rdata[i]+rdata[i+1]*rdata[i+1];
					else v=rdata[i]*rdata[i];
				}
			}
			else {
				r=hypot(float(x-nx/2),float(y-ny/2));
				a=atan2(float(y-ny/2),float(x-nx/2));
				if (inten) v=rdata[i]*rdata[i];
				else v=rdata[i];
			}
			int bin=n*int((a+M_PI)/astep);
			if (bin>=nwedge) bin=nwedge-1;
			r=(r-x0)/dx;
			int f=int(r);	// safe truncation, so floor isn't needed
			r-=float(f);	// r is now the fractional spacing between bins
			if (f>=0 && f<n) {
				ret[f+bin]+=v*(1.0-r);
				norm[f+bin]+=(1.0-r);
				if (f<n-1) {
					ret[f+1+bin]+=v*r;
					norm[f+1+bin]+=r;
				}
			}
		}
	}

	for (i=0; i<n*nwedge; i++) ret[i]/=norm[i]?norm[i]:1.0;	// Normalize
	EXITFUNC;

	return ret;
}

void EMData::cconj() {
	ENTERFUNC;
	if (!is_complex() || !is_ri())
		throw ImageFormatException("EMData::conj requires a complex, ri image");
	int nxreal = nx -2 + int(is_fftodd());
	int nxhalf = nxreal/2;
	for (int iz = 0; iz < nz; iz++)
		for (int iy = 0; iy < ny; iy++)
			for (int ix = 0; ix <= nxhalf; ix++)
				cmplx(ix,iy,iz) = conj(cmplx(ix,iy,iz));
	EXITFUNC;
}


void EMData::update_stat()
{
	ENTERFUNC;

	if (!(flags & EMDATA_NEEDUPD)) {
		return;
	}

	float max = -FLT_MAX;
	float min = -max;

	double sum = 0;
	double square_sum = 0;

	int step = 1;
	if (is_complex() && !is_ri()) {
		step = 2;
	}

	int n_nonzero = 0;

	for (int i = 0; i < nx*ny*nz; i += step) {
		float v = rdata[i];
	#ifdef _WIN32
		max = _cpp_max(max,v);
		min = _cpp_min(min,v);
	#else
		max=std::max<float>(max,v);
		min=std::min<float>(min,v);
	#endif	//_WIN32
		sum += v;
		square_sum += v * (double)(v);
		if (v != 0) n_nonzero++;
	}

	int n = nx * ny * nz / step;
	double mean = sum / n;

#ifdef _WIN32
	float sigma = (float)sqrt( _cpp_max(0.0,(square_sum - sum*sum / n)/(n-1)));
	n_nonzero = _cpp_max(1,n_nonzero);
	double sigma_nonzero = sqrt( _cpp_max(0,(square_sum  - sum*sum/n_nonzero)/(n_nonzero-1)));
#else
	float sigma = (float)sqrt(std::max<double>(0.0,(square_sum - sum*sum / n)/(n-1)));
	n_nonzero = std::max<int>(1,n_nonzero);
	double sigma_nonzero = sqrt(std::max<double>(0,(square_sum  - sum*sum/n_nonzero)/(n_nonzero-1)));
#endif	//_WIN32
	double mean_nonzero = sum / n_nonzero; // previous version overcounted! G2
	
	attr_dict["minimum"] = min;
	attr_dict["maximum"] = max;
	attr_dict["mean"] = (float)(mean);
	attr_dict["sigma"] = (float)(sigma);
	attr_dict["square_sum"] = (float)(square_sum);
	attr_dict["mean_nonzero"] = (float)(mean_nonzero);
	attr_dict["sigma_nonzero"] = (float)(sigma_nonzero);
	attr_dict["is_complex"] = (int) is_complex();
	attr_dict["is_complex_ri"] = (int) is_ri();

	flags &= ~EMDATA_NEEDUPD;
	EXITFUNC;
}














EMData * EMAN::operator+(const EMData & em, float n)
{
	EMData * r = em.copy();
	r->add(n);
	return r;
}

EMData * EMAN::operator-(const EMData & em, float n)
{
	EMData* r = em.copy();
	r->sub(n);
	return r;
}

EMData * EMAN::operator*(const EMData & em, float n)
{
	EMData* r = em.copy();
	r ->mult(n);
	return r;
}

EMData * EMAN::operator/(const EMData & em, float n)
{
	EMData * r = em.copy();
	r->div(n);
	return r;
}


EMData * EMAN::operator+(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->add(n);
	return r;
}

EMData * EMAN::operator-(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->mult(-1.0f);
	r->add(n);
	return r;
}

EMData * EMAN::operator*(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->mult(n);
	return r;
}

EMData * EMAN::operator/(float n, const EMData & em)
{
	EMData * r = em.copy();
	r->to_one();
	r->mult(n);
	r->div(em);

	return r;
}


EMData * EMAN::operator+(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->add(b);
	return r;
}

EMData * EMAN::operator-(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->sub(b);
	return r;
}

EMData * EMAN::operator*(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->mult(b);
	return r;
}

EMData * EMAN::operator/(const EMData & a, const EMData & b)
{
	EMData * r = a.copy();
	r->div(b);
	return r;
}




void EMData::set_xyz_origin(float origin_x, float origin_y, float origin_z)
{
	attr_dict["origin_row"] = origin_x;
	attr_dict["origin_col"] = origin_y;
	attr_dict["origin_sec"] = origin_z;
}


void EMData::to_zero()
{
	ENTERFUNC;

	if (is_complex()) {
		set_ri(true);
	}
	else {
		set_ri(false);
	}

	memset(rdata, 0, nx * ny * nz * sizeof(float));
	done_data();
	EXITFUNC;
}















#if 0
void EMData::calc_rcf(EMData * with, vector < float >&sum_array)
{
	ENTERFUNC;

	int array_size = sum_array.size();
	float da = 2 * M_PI / array_size;
	float *dat = new float[array_size + 2];
	float *dat2 = new float[array_size + 2];
	int nx2 = nx * 9 / 20;

	float lim = 0;
	if (fabs(translation[0]) < fabs(translation[1])) {
		lim = fabs(translation[1]);
	}
	else {
		lim = fabs(translation[0]);
	}

	nx2 -= static_cast < int >(floor(lim));

	for (int i = 0; i < array_size; i++) {
		sum_array[i] = 0;
	}

	float sigma = attr_dict["sigma"];
	float with_sigma = with->get_attr_dict().get("sigma");

	vector<float> vdata, vdata2;
	for (int i = 8; i < nx2; i += 6) {
		vdata = calc_az_dist(array_size, 0, da, i, i + 6);
		vdata2 = with->calc_az_dist(array_size, 0, da, i, i + 6);
		assert(vdata.size() <= array_size + 2);
		assert(cdata2.size() <= array_size + 2);
		std::copy(vdata.begin(), vdata.end(), dat);
		std::copy(vdata2.begin(), vdata2.end(), dat2);

		EMfft::real_to_complex_1d(dat, dat, array_size);
		EMfft::real_to_complex_1d(dat2, dat2, array_size);

		for (int j = 0; j < array_size + 2; j += 2) {
			float max = dat[j] * dat2[j] + dat[j + 1] * dat2[j + 1];
			float max2 = dat[j + 1] * dat2[j] - dat2[j + 1] * dat[j];
			dat[j] = max;
			dat[j + 1] = max2;
		}

		EMfft::complex_to_real_1d(dat, dat, array_size);
		float norm = array_size * array_size * (4.0f * sigma) * (4.0f * with_sigma);

		for (int j = 0; j < array_size; j++) {
			sum_array[j] += dat[j] * (float) i / norm;
		}
	}

	if( dat )
	{
		delete[]dat;
		dat = 0;
	}

	if( dat2 )
	{
		delete[]dat2;
		dat2 = 0;
	}
	EXITFUNC;
}

#endif


void EMData::to_one()
{
	ENTERFUNC;

	if (is_complex()) {
		set_ri(true);
	}
	else {
		set_ri(false);
	}

	for (int i = 0; i < nx * ny * nz; i++) {
		rdata[i] = 1.0f;
	}

	update();
	EXITFUNC;
}






void EMData::add_incoherent(EMData * obj)
{
	ENTERFUNC;

	if (!obj) {
		LOGERR("NULL image");
		throw NullPointerException("NULL image");
	}

	if (!obj->is_complex() || !is_complex()) {
		throw ImageFormatException("complex images only");
	}

	if (!EMUtil::is_same_size(this, obj)) {
		throw ImageFormatException("images not same size");
	}

	ri2ap();
	obj->ri2ap();

	float *dest = get_data();
	float *src = obj->get_data();
	int size = nx * ny * nz;
	for (int j = 0; j < size; j += 2) {
		dest[j] = (float) hypot(src[j], dest[j]);
		dest[j + 1] = 0;
	}

	obj->done_data();
	done_data();
	update();
	EXITFUNC;
}




float EMData::calc_dist(EMData * second_img, int y_index) const
{
	ENTERFUNC;

	if (get_ndim() != 1) {
		throw ImageDimensionException("'this' image is 1D only");
	}

	if (second_img->get_xsize() != nx || ny != 1) {
		throw ImageFormatException("image xsize not same");
	}

	if (y_index > second_img->get_ysize() || y_index < 0) {
		return -1;
	}

	float ret = 0;
	float *d1 = get_data();
	float *d2 = second_img->get_data() + second_img->get_xsize() * y_index;

	for (int i = 0; i < nx; i++) {
		ret += Util::square(d1[i] - d2[i]);
	}
	EXITFUNC;
	return sqrt(ret);
}


//  The following code looks strange - does anybody know it?  Please let me know, pawel.a.penczek@uth.tmc.edu  04/09/06.
//This function does not work, it fails with simple function call in unit test. --Grant Tang
EMData *EMData::calc_flcf(EMData * with, int radius, const string & mask_filter)
{
	ENTERFUNC;

	if (!with) {
		LOGERR("input image is NULL");
		throw NullPointerException("input image is NULL");
	}

	Dict filter_dict;
	if (mask_filter == "eman1.mask.sharp") {
		filter_dict["value"] = 0;
	}

	EMData *img1 = this->copy();
	EMData *img2 = with->copy();

	int img1_nx = img1->get_xsize();
	int img1_ny = img1->get_ysize();
	int img1_nz = img1->get_zsize();
	int img1_size = img1_nx * img1_ny * img1_nz;

	float img1min = img1->get_attr("minimum");
	img1->add(-img1min);

	float img2min = img2->get_attr("minimum");
	img2->add(-img2min);

	filter_dict["outer_radius"] = radius;

	EMData *img1_copy = img1->copy();
	img1_copy->to_one();
	img1_copy->process_inplace(mask_filter, filter_dict);
	img1_copy->process_inplace("eman1.xform.phaseorigin");

	int num = 0;
	float *img1_copy_data = img1_copy->get_data();

	for (int i = 0; i < img1_size; i++) {
		if (img1_copy_data[i] == 1) {
			num++;
		}
	}

	img2->process_inplace(mask_filter, filter_dict);

	float *img2_data = img2->get_data();
	double lsum = 0;
	double sumsq = 0;

	for (int i = 0; i < img1_size; i++) {
		lsum += img2_data[i];
		sumsq += img2_data[i] * img2_data[i];
	}

	float sq = (float)((num * sumsq - lsum * lsum) / (num * num));
	if (sq < 0) {
		LOGERR("sigma < 0");
		throw ImageFormatException("image sigma < 0");
	}

	float mean = (float)lsum / num;
	float sigma = sqrt(sq);
	float th = 0.00001f;

	if (sq > th) {
		for (int i = 0; i < img1_size; i++) {
			img2_data[i] = (img2_data[i] - mean) / sigma;
		}
	}
	else {
		for (int i = 0; i < img1_size; i++) {
			img2_data[i] -= mean;
		}
	}

	img2->done_data();

	EMData *img2_copy = img2->copy();
	if( img2 )
	{
		delete img2;
		img2 = 0;
	}

	img2_copy->process_inplace(mask_filter, filter_dict);
	img2_copy->process_inplace("eman1.xform.phaseorigin");

	if( img1_copy )
	{
		delete img1_copy;
		img1_copy = 0;
	}

	EMData *img1_copy2 = img1->copy();

	img1_copy2->process_inplace("eman1.math.squared");

	EMData *ccf = img1->calc_ccf(img2_copy);
	if( img2_copy )
	{
		delete img2_copy;
		img2_copy = 0;
	}

	ccf->mult(img1_size);

	EMData *conv1 = img1->convolute(img1_copy2);
	if( img1 )
	{
		delete img1;
		img1 = 0;
	}

	conv1->mult(img1_size);
	conv1->mult(1.0f / num);

	EMData *conv2 = img1_copy2->convolute(img1_copy2);
	if( img1_copy2 )
	{
		delete img1_copy2;
		img1_copy2 = 0;
	}

	conv2->mult(img1_size);
	conv1->process_inplace("eman1.math.squared");
	conv1->mult(1.0f / (num * num));

	EMData *conv2_copy = conv2->copy();
	if( conv2 )
	{
		delete conv2;
		conv2 = 0;
	}

	conv2_copy->sub(*conv1);
	if( conv1 )
	{
		delete conv1;
		conv1 = 0;
	}

	conv2_copy->mult(1.0f / num);
	conv2_copy->process_inplace("eman1.math.sqrt");

	EMData *ccf_copy = ccf->copy();
	if( ccf )
	{
		delete ccf;
		ccf = 0;
	}

	ccf_copy->mult(1.0f / num);

	float *lcfd = ccf_copy->get_data();
	float *vdd = conv2_copy->get_data();

	for (int i = 0; i < img1_size; i++) {
		if (vdd[i] > 0) {
			lcfd[i] /= vdd[i];
		}
	}
	if( conv2_copy )
	{
		delete conv2_copy;
		conv2_copy = 0;
	}

	ccf_copy->done_data();
	EMData *lcf = ccf_copy->copy();
	if( ccf_copy )
	{
		delete ccf_copy;
		ccf_copy = 0;
	}

	EXITFUNC;
	return lcf;
}

EMData *EMData::convolute(EMData * with)
{
	ENTERFUNC;

	EMData *f1 = do_fft();
	if (!f1) {
		LOGERR("FFT returns NULL image");
		throw NullPointerException("FFT returns NULL image");
	}

	f1->ap2ri();

	EMData *cf = 0;
	if (with) {
		cf = with->do_fft();
		if (!cf) {
			LOGERR("FFT returns NULL image");
			throw NullPointerException("FFT returns NULL image");
		}
		cf->ap2ri();
	}
	else {
		cf = f1->copy();
	}

	if (with && !EMUtil::is_same_size(f1, cf)) {
		LOGERR("images not same size");
		throw ImageFormatException("images not same size");
	}

	float *rdata1 = f1->get_data();
	float *rdata2 = cf->get_data();
	int cf_size = cf->get_xsize() * cf->get_ysize() * cf->get_zsize();

	float re,im;
	for (int i = 0; i < cf_size; i += 2) {
		re = rdata1[i] * rdata2[i] - rdata1[i + 1] * rdata2[i + 1];
		im = rdata1[i + 1] * rdata2[i] + rdata1[i] * rdata2[i + 1];
		rdata2[i]=re;
		rdata2[i+1]=im;
	}

	cf->done_data();
	EMData *f2 = cf->do_ift();

	if( cf )
	{
		delete cf;
		cf = 0;
	}

	if( f1 )
	{
		delete f1;
		f1=0;
	}

	EXITFUNC;
	return f2;
}


void EMData::common_lines(EMData * image1, EMData * image2,
						  int mode, int steps, bool horizontal)
{
	ENTERFUNC;

	if (!image1 || !image2) {
		throw NullPointerException("NULL image");
	}

	if (mode < 0 || mode > 2) {
		throw OutofRangeException(0, 2, mode, "invalid mode");
	}

	if (!image1->is_complex()) {
		image1 = image1->do_fft();
	}
	if (!image2->is_complex()) {
		image2 = image2->do_fft();
	}

	image1->ap2ri();
	image2->ap2ri();

	if (!EMUtil::is_same_size(image1, image2)) {
		throw ImageFormatException("images not same sizes");
	}

	int image2_nx = image2->get_xsize();
	int image2_ny = image2->get_ysize();

	int rmax = image2_ny / 4 - 1;
	int array_size = steps * rmax * 2;
	float *im1 = new float[array_size];
	float *im2 = new float[array_size];
	for (int i = 0; i < array_size; i++) {
		im1[i] = 0;
		im2[i] = 0;
	}

	set_size(steps * 2, steps * 2, 1);

	float *image1_data = image1->get_data();
	float *image2_data = image2->get_data();

	float da = M_PI / steps;
	float a = -M_PI / 2.0f + da / 2.0f;
	int jmax = 0;

	for (int i = 0; i < steps * 2; i += 2, a += da) {
		float s1 = 0;
		float s2 = 0;
		int i2 = i * rmax;
		int j = 0;

		for (float r = 3.0f; r < rmax - 3.0f; j += 2, r += 1.0f) {
			float x = r * cos(a);
			float y = r * sin(a);

			if (x < 0) {
				x = -x;
				y = -y;
				LOGERR("CCL ERROR %d, %f !\n", i, -x);
			}

			int k = (int) (floor(x) * 2 + floor(y + image2_ny / 2) * image2_nx);
			int l = i2 + j;
			float x2 = x - floor(x);
			float y2 = y - floor(y);

			im1[l] = Util::bilinear_interpolate(image1_data[k],
												image1_data[k + 2],
												image1_data[k + 2 + image2_nx],
												image1_data[k + image2_nx], x2, y2);

			im2[l] = Util::bilinear_interpolate(image2_data[k],
												image2_data[k + 2],
												image2_data[k + 2 + image2_nx],
												image2_data[k + image2_nx], x2, y2);

			k++;

			im1[l + 1] = Util::bilinear_interpolate(image1_data[k],
													image1_data[k + 2],
													image1_data[k + 2 + image2_nx],
													image1_data[k + image2_nx], x2, y2);

			im2[l + 1] = Util::bilinear_interpolate(image2_data[k],
													image2_data[k + 2],
													image2_data[k + 2 + image2_nx],
													image2_data[k + image2_nx], x2, y2);

			s1 += Util::square_sum(im1[l], im1[l + 1]);
			s2 += Util::square_sum(im2[l], im2[l + 1]);
		}

		jmax = j - 1;
		float sqrt_s1 = sqrt(s1);
		float sqrt_s2 = sqrt(s2);

		int l = 0;
		for (float r = 1; r < rmax; r += 1.0f) {
			int i3 = i2 + l;
			im1[i3] /= sqrt_s1;
			im1[i3 + 1] /= sqrt_s1;
			im2[i3] /= sqrt_s2;
			im2[i3 + 1] /= sqrt_s2;
			l += 2;
		}
	}

	switch (mode) {
	case 0:
		for (int m1 = 0; m1 < 2; m1++) {
			for (int m2 = 0; m2 < 2; m2++) {

				if (m1 == 0 && m2 == 0) {
					for (int i = 0; i < steps; i++) {
						int i2 = i * rmax * 2;
						for (int j = 0; j < steps; j++) {
							int l = i + j * steps * 2;
							int j2 = j * rmax * 2;
							rdata[l] = 0;
							for (int k = 0; k < jmax; k++) {
								rdata[l] += im1[i2 + k] * im2[j2 + k];
							}
						}
					}
				}
				else {
					int steps2 = steps * m2 + steps * steps * 2 * m1;

					for (int i = 0; i < steps; i++) {
						int i2 = i * rmax * 2;
						for (int j = 0; j < steps; j++) {
							int j2 = j * rmax * 2;
							int l = i + j * steps * 2 + steps2;
							rdata[l] = 0;

							for (int k = 0; k < jmax; k += 2) {
								i2 += k;
								j2 += k;
								rdata[l] += im1[i2] * im2[j2];
								rdata[l] += -im1[i2 + 1] * im2[j2 + 1];
							}
						}
					}
				}
			}
		}

		break;
	case 1:
		for (int m1 = 0; m1 < 2; m1++) {
			for (int m2 = 0; m2 < 2; m2++) {
				int steps2 = steps * m2 + steps * steps * 2 * m1;
				int p1_sign = 1;
				if (m1 != m2) {
					p1_sign = -1;
				}

				for (int i = 0; i < steps; i++) {
					int i2 = i * rmax * 2;

					for (int j = 0; j < steps; j++) {
						int j2 = j * rmax * 2;

						int l = i + j * steps * 2 + steps2;
						rdata[l] = 0;
						float a = 0;

						for (int k = 0; k < jmax; k += 2) {
							i2 += k;
							j2 += k;

							float a1 = (float) hypot(im1[i2], im1[i2 + 1]);
							float p1 = atan2(im1[i2 + 1], im1[i2]);
							float p2 = atan2(im2[j2 + 1], im2[j2]);

							rdata[l] += Util::angle_sub_2pi(p1_sign * p1, p2) * a1;
							a += a1;
						}

						rdata[l] /= (float)(a * M_PI / 180.0f);
					}
				}
			}
		}

		break;
	case 2:
		for (int m1 = 0; m1 < 2; m1++) {
			for (int m2 = 0; m2 < 2; m2++) {
				int steps2 = steps * m2 + steps * steps * 2 * m1;

				for (int i = 0; i < steps; i++) {
					int i2 = i * rmax * 2;

					for (int j = 0; j < steps; j++) {
						int j2 = j * rmax * 2;
						int l = i + j * steps * 2 + steps2;
						rdata[l] = 0;

						for (int k = 0; k < jmax; k += 2) {
							i2 += k;
							j2 += k;
							rdata[l] += (float) (hypot(im1[i2], im1[i2 + 1]) * hypot(im2[j2], im2[j2 + 1]));
						}
					}
				}
			}
		}

		break;
	default:
		break;
	}

	if (horizontal) {
		float *tmp_array = new float[ny];
		for (int i = 1; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				tmp_array[j] = get_value_at(i, j);
			}
			for (int j = 0; j < ny; j++) {
				set_value_at(i, j, 0, tmp_array[(j + i) % ny]);
			}
		}
		if( tmp_array )
		{
			delete[]tmp_array;
			tmp_array = 0;
		}
	}

	if( im1 )
	{
		delete[]im1;
		im1 = 0;
	}

	if( im2 )
	{
		delete im2;
		im2 = 0;
	}


	image1->done_data();
	image2->done_data();
	if( image1 )
	{
		delete image1;
		image1 = 0;
	}
	if( image2 )
	{
		delete image2;
		image2 = 0;
	}
	done_data();
	update();
	EXITFUNC;
}



void EMData::common_lines_real(EMData * image1, EMData * image2,
							   int steps, bool horiz)
{
	ENTERFUNC;

	if (!image1 || !image2) {
		throw NullPointerException("NULL image");
	}

	if (!EMUtil::is_same_size(image1, image2)) {
		throw ImageFormatException("images not same size");
	}

	int steps2 = steps * 2;
	int image_ny = image1->get_ysize();
	EMData *image1_copy = image1->copy();
	EMData *image2_copy = image2->copy();

	float *im1 = new float[steps2 * image_ny];
	float *im2 = new float[steps2 * image_ny];

	EMData *images[] = { image1_copy, image2_copy };
	float *ims[] = { im1, im2 };

	for (int m = 0; m < 2; m++) {
		float *im = ims[m];
		float a = M_PI / steps2;

		for (int i = 0; i < steps2; i++) {
			images[i]->rotate(-a, 0, 0);
			float *data = images[i]->get_data();

			for (int j = 0; j < image_ny; j++) {
				float sum = 0;
				for (int k = 0; k < image_ny; k++) {
					sum += data[j * image_ny + k];
				}
				im[i * image_ny + j] = sum;
			}

			float sum1 = 0;
			float sum2 = 0;
			for (int j = 0; j < image_ny; j++) {
				int l = i * image_ny + j;
				sum1 += im[l];
				sum2 += im[l] * im[l];
			}

			float mean = sum1 / image_ny;
			float sigma = sqrt(sum2 / image_ny - sum1 * sum1);

			for (int j = 0; j < image_ny; j++) {
				int l = i * image_ny + j;
				im[l] = (im[l] - mean) / sigma;
			}

			images[i]->done_data();
			a += M_PI / steps;
		}
	}

	set_size(steps2, steps2, 1);
	float *data1 = get_data();

	if (horiz) {
		for (int i = 0; i < steps2; i++) {
			for (int j = 0, l = i; j < steps2; j++, l++) {
				if (l == steps2) {
					l = 0;
				}

				float sum = 0;
				for (int k = 0; k < image_ny; k++) {
					sum += im1[i * image_ny + k] * im2[l * image_ny + k];
				}
				data1[i + j * steps2] = sum;
			}
		}
	}
	else {
		for (int i = 0; i < steps2; i++) {
			for (int j = 0; j < steps2; j++) {
				float sum = 0;
				for (int k = 0; k < image_ny; k++) {
					sum += im1[i * image_ny + k] * im2[j * image_ny + k];
				}
				data1[i + j * steps2] = sum;
			}
		}
	}

	done_data();

	if( image1_copy )
	{
		delete image1_copy;
		image1_copy = 0;
	}

	if( image2_copy )
	{
		delete image2_copy;
		image2_copy = 0;
	}

	if( im1 )
	{
		delete[]im1;
		im1 = 0;
	}

	if( im2 )
	{
		delete[]im2;
		im2 = 0;
	}
	EXITFUNC;
}


void EMData::cut_slice(const EMData * map, float dz, Transform3D * ort,
					   bool interpolate, float dx, float dy)
{
	ENTERFUNC;

	if (!map) {
		throw NullPointerException("NULL image");
	}

	Transform3D r(0, 0, 0); // EMAN by default
	if (!ort) {
		ort = &r;
	}

	float *sdata = map->get_data();
	float *ddata = get_data();

	int map_nx = map->get_xsize();
	int map_ny = map->get_ysize();
	int map_nz = map->get_zsize();
	int map_nxy = map_nx * map_ny;

	float mdz0 = dz * (*ort)[0][2] + map_nx / 2;
	float mdz1 = dz * (*ort)[1][2] + map_ny / 2;
	float mdz2 = dz * (*ort)[2][2] + map_nz / 2;

	for (int y = 0; y < ny; y++) {
		int y2 = (int) (y - ny / 2 - dy);
		float my2_0 = y2 * (*ort)[0][1] * y2 + mdz0;
		float my2_1 = y2 * (*ort)[1][1] * y2 + mdz1;
		float my2_2 = y2 * (*ort)[2][1] * y2 + mdz2;

		for (int x = 0; x < nx; x++) {
			int x2 = (int) (x - nx / 2 - dx);
			float xx = x2 * (*ort)[0][0] + my2_0;
			float yy = x2 * (*ort)[1][0] + my2_1;
			float zz = x2 * (*ort)[2][0] + my2_2;
			int l = x + y * nx;

			if (xx < 0 || yy < 0 || zz < 0 || xx > map_nx - 2 ||
				yy > map_ny - 2 || zz > map_nz - 2) {
				ddata[l] = 0;
			}
			else {
				float t = xx - floor(xx);
				float u = yy - floor(yy);
				float v = zz - floor(zz);

				if (interpolate) {
					int k = (int) (floor(xx) + floor(yy) * map_nx + floor(zz) * map_nxy);

					ddata[l] = Util::trilinear_interpolate(sdata[k],
														   sdata[k + 1],
														   sdata[k + map_nx],
														   sdata[k + map_nx + 1],
														   sdata[k + map_nxy],
														   sdata[k + map_nxy + 1],
														   sdata[k + map_nx + map_nxy],
														   sdata[k + map_nx + map_nxy + 1],
														   t, u, v);
				}
				else {
					int k = Util::round(xx) + Util::round(yy) * map_nx + Util::round(zz) * map_nxy;
					ddata[l] = sdata[k];
				}
			}
		}
	}

	done_data();

	EXITFUNC;
}


void EMData::uncut_slice(EMData * map, float dz, Transform3D * ort, float dx, float dy)
{
	ENTERFUNC;

	if (!map) {
		throw NullPointerException("NULL image");
	}

	Transform3D r( 0, 0, 0); // EMAN by default
	if (!ort) {
		ort = &r;
	}

	float *ddata = map->get_data();
	float *sdata = get_data();

	int map_nx = map->get_xsize();
	int map_ny = map->get_ysize();
	int map_nz = map->get_zsize();
	int map_nxy = map_nx * map_ny;

	float mdz0 = dz * (*ort)[0][2] + map_nx / 2;
	float mdz1 = dz * (*ort)[1][2] + map_ny / 2;
	float mdz2 = dz * (*ort)[2][2] + map_nz / 2;

	for (int y = 0; y < ny; y++) {
		int y2 = (int) (y - ny / 2 - dy);

		float my2_0 = y2 * (*ort)[0][1] + mdz0;
		float my2_1 = y2 * (*ort)[1][1] + mdz1;
		float my2_2 = y2 * (*ort)[2][1] + mdz2;

		for (int x = 0; x < nx; x++) {
			int x2 = (int) (x - nx / 2 - dx);

			float xx = x2 * (*ort)[0][0] + my2_0;
			float yy = x2 * (*ort)[1][0] + my2_1;
			float zz = x2 * (*ort)[2][0] + my2_2;

			if (xx >= 0 && yy >= 0 && zz >= 0 && xx <= map_nx - 2 && yy <= map_ny - 2
				&& zz <= map_nz - 2) {
				int k = Util::round(xx) + Util::round(yy) * map_nx + Util::round(zz) * map_nxy;
				ddata[k] = sdata[x + y * nx];
			}
		}
	}

	done_data();
	map->done_data();
	EXITFUNC;
}






void EMData::save_byteorder_to_dict(ImageIO * imageio)
{
	string image_endian = "ImageEndian";
	string host_endian = "HostEndian";

	if (imageio->is_image_big_endian()) {
		attr_dict[image_endian] = "big";
	}
	else {
		attr_dict[image_endian] = "little";
	}

	if (ByteOrder::is_host_big_endian()) {
		attr_dict[host_endian] = "big";
	}
	else {
		attr_dict[host_endian] = "little";
	}
}

/* vim: set ts=4 noet nospell: */

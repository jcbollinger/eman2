/*
 * Author: Steven Ludtke, 04/10/2003 (sludtke@bcm.edu)
 * Copyright (c) 2000-2006 Baylor College of Medicine
 * 
 * This software is issued under a joint BSD/GNU license. You may use the
 * source code in this file under either license. However, note that the
 * complete EMAN2 and SPARX software packages have some GPL dependencies,
 * so you are responsible for compliance with the licenses of these packages
 * if you opt to use BSD licensing. The warranty disclaimer below holds
 * in either instance.
 * 
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * */

#include "emdata.h"
#include "testutil.h"
using namespace EMAN;

int main()
{
	EMData * image = new EMData();
	EMData * image2 = new EMData();
	
	string test_imagename = TestUtil::get_debug_image("search.dm3");
	int err = 0;
	
	try {
		image->read_image(test_imagename);
		image2->read_image(test_imagename);

		image->align("rotate_translate_flip", image2, Dict());

	}
	catch(E2Exception & e) {
		err = 1;
		printf("%s\n", e.what());
	}

	if( image )
	{
		delete image;
		image = 0;
	}
	if( image2 )
	{
		delete image2;
		image2 = 0;
	}
	
	return err;
}
	

	
		
		

#! /usr/bin/env python
#
# Author: Markus Stabrin 2019 (markus.stabrin@mpi-dortmund.mpg.de)
# Author: Fabian Schoenfeld 2019 (fabian.schoenfeld@mpi-dortmund.mpg.de)
# Author: Thorsten Wagner 2019 (thorsten.wagner@mpi-dortmund.mpg.de)
# Author: Tapu Shaikh 2019 (tapu.shaikh@mpi-dortmund.mpg.de)
# Author: Adnan Ali 2019 (adnan.ali@mpi-dortmund.mpg.de)
# Author: Luca Lusnig 2019 (luca.lusnig@mpi-dortmund.mpg.de)
# Author: Toshio Moriya 2019 (toshio.moriya@kek.jp)
# Author: Pawel A.Penczek, 09/09/2006 (Pawel.A.Penczek@uth.tmc.edu)
#
# Copyright (c) 2019 Max Planck Institute of Molecular Physiology
# Copyright (c) 2000-2006 The University of Texas - Houston Medical School
#
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
#
#


import os
import sp_global_def
from sp_global_def import sxprint, ERROR

from sp_global_def import *
from optparse import OptionParser
import sys

def main():
	progname = os.path.basename(sys.argv[0])
	usage = progname + " stack outdir <maskfile> --ou=outer_radius --br=brackets --center=center_type --eps=epsilon --maxit=max_iter --CTF --snr=SNR --function=user_function_name"
	parser = OptionParser(usage,version=SPARXVERSION)
	parser.add_option("--ou", type="float", default=-1, help="  outer radius for a 2-D mask within which the alignment is performed")
	parser.add_option("--br", type="float", default=1.75, help="  brackets for the search of orientation parameters (each parameter will be checked +/- bracket (set to 1.75)")
	parser.add_option("--center", type="float", default=1, help="  0 - if you do not want the average to be centered, 1 - center the average (default=1)")
	parser.add_option("--eps", type="float", default=0.001, help="  stopping criterion, program will terminate when the relative increase of the criterion is less than epsilon  ")
	parser.add_option("--maxit", type="float", default=10, help="  maximum number of iterations (set to 10) ")
	parser.add_option("--CTF", action="store_true", default=False, help="  Consider CTF correction during the alignment ")
	parser.add_option("--snr", type="float", default=1.0, help="  Signal-to-Noise Ratio of the data")	
	parser.add_option("--function", type="string", default="ref_ali2d", help="  name of the reference preparation function")
	(options, args) = parser.parse_args()
	if len(args) < 2 or len(args) >3:
		sxprint( "Usage: " + usage )
		sxprint( "Please run '" + progname + " -h' for detailed options" )
		ERROR( "Invalid number of parameters used. Please see usage information above." )
		return
		
	else:
		if len(args) == 2:
			mask = None
		else:
			mask = args[2]
		
		from sp_applications import local_ali2d

		if sp_global_def.CACHE_DISABLE:
			from sp_utilities import disable_bdb_cache
			disable_bdb_cache()
		
		sp_global_def.BATCH = True	
		local_ali2d(args[0], args[1], mask, options.ou, options.br, options.center, options.eps, options.maxit, options.CTF, options.snr, options.function)
		sp_global_def.BATCH = False	

if __name__ == "__main__":
	sp_global_def.print_timestamp( "Start" )
	sp_global_def.write_command()
	main()
	sp_global_def.print_timestamp( "Finish" )

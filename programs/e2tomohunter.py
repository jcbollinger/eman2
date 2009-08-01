#!/usr/bin/env python

#
# Author: Matthew Baker, 10/2005, modified 02/2006 by MFS  
# ported to EMAN2 by David Woolford October 6th 2008
# Copyright (c) 2000-2006 Baylor College of Medicine
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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
#
#


#N tomohunter.py
#F tomography hunter

import os
import sys
import string
import commands
import math
from EMAN2 import *
#import Numeric
from math import *
from sys import argv
from optparse import OptionParser

def check_options(options,filenames):
	'''
	Check the parser options
	Should probably be made into a class
	@return a list of error messages
	'''
	error_messages = []
	if not options.probe or not file_exists(options.probe):
		error_messages.append("You have to specify a valid probe")
	
	if len(filenames) < 1:
		error_messages.append("You must specify input files")
	else:
		all_images = True
		for f in filenames:
			if not file_exists(f): 
				error_messages.append("Error - %s does not exist" %f)
				all_images = False
				
		if all_images:
			nx,ny,nz = gimme_image_dimensions3D(filenames[0])
			for i in range(1,len(filenames)):
				x,y,z = gimme_image_dimensions3D(filenames[i])
				if x != nx or y != ny or z != nz:
					error_messages.append("File %s does not have the same dimensions as file %s" %(filenames[i],filenames[0]))
									
	if options.nsoln <= 0:
		error_messages.append("Error - nsoln must be greater than 0. Suggest using 10")
	
	
	if options.align == None:
		error_messages.append("Error - you have to supply the align option")
	else:
		error = check_eman2_type_string(options.align,Aligners,"Aligners")
		if error != None:
			error_messages.append(error)
			
	if options.ralign != None:
		error = check_eman2_type_string(options.ralign,Aligners,"Aligners")
		if error != None:
			error_messages.append(error)
	
	return error_messages

def gen_average(options,args,logid=None):
	
	project_list = "global.tpr_ptcls_ali_dict"
	db = db_open_dict("bdb:project",ro=True)
	db_map = db.get(project_list,dfl={})
	
	probes = []
	probes_data = {}
	
	average = None
	
	prog = 0
	total_prog = len(args)
	if logid: E2progress(logid,0.0)
	
	for i,arg in enumerate(args):
		image = EMData(arg,0)
		if not options.aliset:
			t = db_map[arg][options.probe][0]
		else:
			t = get_ali_data(arg,options.probe,options.aliset)
			if t == None:
				raise RuntimeError("An error occured trying to retrieve the alignment data using the given ali set")
			
		image.process_inplace("math.transform",{"transform":t})
		if average == None: average = image
		else: average = average + image
		if logid: E2progress(logid,(i+1)/float(total_prog))
		
	average.mult(1.0/len(args))
		
	average.write_image(options.avgout,0)
	
	if options.dbls:
		pdb = db_open_dict("bdb:project")
		db = pdb.get(options.dbls,dfl={})
		if isinstance(db,list): # this is for back compatibility - it used to be a list, now it's a dict
			d = {}
			for name in db:
				s = {}
				s["Original Data"] = name
				d[name] = s
			db = d
		s = {}
		s["Original Data"] = options.avgout
		db[options.avgout] = s
		pdb[options.dbls] = db

def get_ali_data(filename,probe,aliset):
	from emtprworkflow import EMProbeAliTools
	from emsprworkflow import EMPartSetOptions
	
	#EMProjectListCleanup.clean_up_filt_particles(self.project_list)
	db = db_open_dict("bdb:project",ro=True)
	db_map = db.get("global.tpr_ptcls_ali_dict")
	if db_map == None:
		return None # calling function will barf
	
	ptcl_opts = EMPartSetOptions("global.tpr_ptcls_dict")
	particles_map, particles_name_map, choices, name_map = ptcl_opts.get_particle_options()
	tls = EMProbeAliTools()
	probe_set_map,probe_and_ali,probe_name_map = tls.accrue_data()
	
	base_set = probe_set_map[get_file_tag(probe)][aliset]
	ptcl_base_set = [name_map[name] for name in base_set]
	
	base_name = name_map[filename]
	
	for i in xrange(0,len(base_set)):
		if base_name == ptcl_base_set[i]:
			dct = db_map[base_set[i]]
			if dct.has_key(probe):
				alis = dct[probe]
				return alis[0]
			
	return None
	
def main():
	progname = os.path.basename(sys.argv[0])
	usage = """%prog <images to be aligned> [options]"""
	
	parser = OptionParser(usage=usage,version=EMANVERSION)

	parser.add_option("--probe",type="string",help="The probe. This is the model that the input images will be aligned to", default=None)
	parser.add_option("--thresh",type="float",help="Threshold", default=0.0)
	parser.add_option("--nsoln",type="int",help="The number of solutions to report", default=10)
	parser.add_option("--n",type="int",help="0 or 1, multiplication by the reciprocal of the boxsize", default=1)
	parser.add_option("--dbls",type="string",help="data base list storage, used by the workflow. You can ignore this argument.",default=None)
	parser.add_option("--aliset",type="string",help="Supplied with avgout. Used to choose different alignment parameters from the local database. Used by workflow.", default=None)
	parser.add_option("--avgout",type="string",help="If specified will produce an averaged output, only works if you've previously run alignments", default=None)
	parser.add_option("--align",type="string",help="The aligner and its parameters. e.g. --align=rot.3d.grid:ralt=180:dalt=10:dphi=10:rphi=180:search=5", default=None)
	parser.add_option("--ralign",type="string",help="This is the second stage aligner used to refine the first alignment. This is usually the \'refine\' aligner.", default=None)
	
	if EMUtil.cuda_available():
		parser.add_option("--cuda",action="store_true",help="GPU acceleration using CUDA. Experimental", default=False)
   
	(options, args) = parser.parse_args()
	
	error_messages = check_options(options,args)
	if len(error_messages) != 0:
		msg = "\n"
		for error in error_messages:
			msg += error +"\n"
		parser.error(msg)
		exit(1)
		
	logid=E2init(sys.argv)
	
	if options.avgout: # unfortunately this functionality is part of this script.
		gen_average(options,args,logid)
		exit(0)
	
	prog = 0
	total_prog = len(args)
	E2progress(logid,0.0)
	
	ali = parsemodopt(options.align)
	rali = None
	if options.ralign != None:
		rali = parsemodopt(options.ralign)
	
	ali[1]["threshold"] = options.thresh # this one is used universally
	
	probeMRC=EMData(options.probe,0)
	print_info(probeMRC,"Probe Information")
	if using_cuda(options):
		probeMRC.set_gpu_rw_current()
		probeMRC.cuda_lock()
	
	for arg in args:
		targetMRC =EMData(arg,0)
		print_info(targetMRC,"Target Information")

		if using_cuda(options):
			targetMRC.set_gpu_rw_current()
			targetMRC.cuda_lock() # locking it prevents if from being overwritten
			
		solns = probeMRC.xform_align_nbest(ali[0],targetMRC,ali[1],options.nsoln)
		
		
		if rali:
			refine_parms=rali[1]
			for s in solns:
				refine_parms["xform.align3d"] = s["xform.align3d"]
				aligned = probeMRC.align(rali[0],targetMRC,refine_parms)
				s["xform.align3d"] = aligned.get_attr("xform.align3d")
				
		out=file("log-s3-%s_%s.txt"%(get_file_tag(arg),get_file_tag(options.probe)),"w")
		peak=0
		
		if using_cuda(options):
			targetMRC.cuda_unlock() # locking it prevents if from being overwritten
		
		db = None
		pdb = None
		if options.dbls:
			pdb = db_open_dict("bdb:project")
			db = pdb.get(options.dbls,dfl={})
			if db == None: db = {}
			results = []
	
		for d in solns:
			t = d["xform.align3d"]
			# inverting because the probe was aligned to the target
			t = t.inverse()
			params = t.get_params("eman")
			ALT=params["alt"]
			AZ=params["az"]
			PHI=params["phi"]
			COEFF=str(d["score"])
			LOC=str( ( (params["tx"]),(params["tx"]),(params["tx"] ) ) )
			line="Peak %d rot=( %f, %f, %f ) trans= %s coeff= %s\n"%(peak,ALT,AZ,PHI,LOC,COEFF)
			out.write(line)
			peak=peak+1
			
			if options.dbls:
				t = Transform({"type":"eman","alt":ALT,"phi":PHI,"az":AZ})
				t.set_trans(params["tx"]),(params["tx"]),(params["tx"])
				results.append(t)
		
		if options.dbls:
			if db.has_key(arg):d = db[arg]
			else:d = {}
			d[options.probe] = results
			db[arg] = d
			pdb[options.dbls] = db
			
		prog += 1.0
		E2progress(logid,progress/total_prog)
			
		out.close()
		
	if using_cuda(options):
		probeMRC.cuda_unlock()
	E2progress(logid,1.0) # just make sure of it
		
	
	E2end(logid)
	
def using_cuda(options):
	return EMUtil.cuda_available() and options.cuda
	
def print_info(image,first_line="Information"):
	
	print first_line
	print "   mean:	   %f"%(image.get_attr("mean"))
	print "   sigma:	  %f"%(image.get_attr("sigma"))
	
	
def tomoccf(targetMRC,probeMRC):
	ccf=targetMRC.calc_ccf(probeMRC)
	# removed a toCorner...this puts the phaseorigin at the left corner, but we can work around this by
	# by using EMData.calc_max_location_wrap (below)
	return (ccf)

def ccfFFT(currentCCF, thresh, box):
	tempCCF = currentCCF.do_fft()
	#tempCCF.ri2ap()
	tempCCF.process_inplace("threshold.binary.fourier",{"value":thresh}) 
	mapSum=tempCCF["mean"]*box*box*box
	return(mapSum)

def updateCCF(bestCCF,bestALT,bestAZ,bestPHI,bestX,bestY,bestZ,altrot,azrot,phirot,currentCCF,scalar,n,searchx,searchy,searchz):
	best = currentCCF.calc_max_location_wrap(searchx,searchy,searchz)
	xbest = best[0]
	ybest = best[1]
	zbest = best[2]
	bestValue = currentCCF.get_value_at_wrap(xbest,ybest,zbest)/scalar
	inlist=0
	while inlist < n:
		if  bestValue > bestCCF.get(inlist):
			swlist=n-1
			while swlist >inlist:
				#print swlist
				bestCCF.set(swlist,bestCCF.get(swlist-1,))
				bestALT.set(swlist,bestALT.get(swlist-1))
				bestAZ.set(swlist,bestAZ.get(swlist-1))
				bestPHI.set(swlist,bestPHI.get(swlist-1))
				bestX.set(swlist,bestX.get(swlist-1))
				bestY.set(swlist,bestY.get(swlist-1))
				bestZ.set(swlist,bestZ.get(swlist-1))
				swlist=swlist-1
			bestCCF.set(inlist,bestValue)
			bestALT.set(inlist,altrot)
			bestAZ.set(inlist,azrot)
			bestPHI.set(inlist,phirot)
			bestX.set(inlist,xbest)
			bestY.set(inlist,ybest)
			bestZ.set(inlist,zbest)
			break
		inlist=inlist+1
	
	#bestCCF.update() uneccessary?
	#bestALT.update()
	#bestAZ.update()
	#bestPHI.update()
	return(bestCCF)


def check(options,args):
	#should write a function to check the inputs, such as positive range, delta less than range etc
	#should also check that the file names exist
	
	error = False
	if options.nsoln <= 0:
		error = True
		print "Error - nsoln must be greater than 0. Suggest using 10"
	
	if options.ralt <= 0:
		error = True
		print "Error - ralt must be greater than 0"
	
	# etc....
	
	return error

# If executed as a program
if __name__ == '__main__':
	main() 

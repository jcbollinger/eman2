#!/usr/bin/env python

#
# Author: Jesus Galaz-Montoya 10/2017, 
# Last modification: October/2017
#
# Copyright (c) 2011 Baylor College of Medicine
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

from past.utils import old_div
from EMAN2 import *
from EMAN2_utils import *
import math
import numpy
f#rom copy import deepcopy
import os
import sys
#import random
#from random import choice
#from pprint import pprint
from EMAN2jsondb import JSTask,jsonclasses
#import datetime
import gc 	#this will be used to free-up unused memory


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """prog <output> [options]

	This program produces iterative class-averages akin to those generated by e2classaverage, 
	but for stacks of 3-D Volumes.
	Normal usage is to provide a stack of particle volumes and a classification matrix file 
	(if you have more than one class) defining class membership. 
	Members of each class are then iteratively aligned to each other (within the class) and 
	averaged together. 
	It is also possible to use this program on all of the volumes in a single stack without
	providing a classification matrix.

	Specify preprocessing options through --lowpass, --highpass, --mask, --normproc, --thresh, 
	--preprocess and --shrink. These take EMAN2 processors (to see a list, type e2help.py processors at
	the command line).
	
	Notice that the alignment is broken down into two step: 1) Coarse alignment and 2) Fine 
	alignment. This is done for speed optimization. By default, the particles are preprocessed
	THE SAME was for Coarse and Fine alignment, unless you supply --notprocfinelinecoarse.
	In this case, the particles will be preprocessed with default parameters for fine alignment.
	To specify or inactivate any preprocessing for fine alignment, do so through fine 
	alignment parameters:
	--lowpassfine, --highpassfine, --preprocessfine and --shrinkfine.
	
	"""
			
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
	
	parser.add_header(name="caheader", help="""Options below this label are specific to sptclassaverage""", title="### sptclassaverage options ###", row=3, col=0, rowspan=1, colspan=3, mode='alignment,breaksym')
	
	
	'''
	REQUIRED PARAMETERS
	'''
	parser.add_argument("--input", type=str, default='',help="""Default=None. The name of the input volume stack. MUST be HDF since volume stack support is required.""", guitype='filebox', browser='EMSubTomosTable(withmodal=True,multiselect=False)', row=0, col=0, rowspan=1, colspan=3, mode='alignment,breaksym')

	
	'''
	STANDARD PARAMETERS
	'''
	parser.add_argument("--apix",type=float,default=0.0,help="""Default=0.0 (not used). Use this apix value where relevant instead of whatever is in the header of the reference and the particles.""")

	parser.add_argument("--align",type=str,default="rotate_translate_3d_tree",help="""Default is rotate_translate_3d_tree. See e2help.py aligners to see the list of parameters the aligner takes (for example, if there's symmetry, supply --align rotate_translate_3d_tree:sym=icos). This is the aligner used to align particles to the previous class average. Specify 'None' (with capital N) to disable.""", returnNone=True,guitype='comboparambox', choicelist='re_filter_list(dump_aligners_list(),\'3d\')', row=12, col=0, rowspan=1, colspan=3, nosharedb=True, mode="alignment,breaksym['rotate_symmetry_3d']")
	
	parser.add_argument("--aligncmp",type=str,default="ccc.tomo.thresh",help="""Default=ccc.tomo.thresh. The comparator used for the --align aligner. Do not specify unless you need to use another specific aligner.""",guitype='comboparambox',choicelist='re_filter_list(dump_cmps_list(),\'tomo\')', row=13, col=0, rowspan=1, colspan=3,mode="alignment,breaksym")
	
	parser.add_argument("--clip",type=int,default=0,help="""Default=0 (which means it's not used). Boxsize to clip particles as part of preprocessing to speed up alignment. For example, the boxsize of the particles might be 100 pixels, but the particles are only 50 pixels in diameter. Aliasing effects are not always as deleterious for all specimens, and sometimes 2x padding isn't necessary; still, there are some benefits from 'oversampling' the data during averaging; so you might still want an average of size 2x, but perhaps particles in a box of 1.5x are sufficiently good for alignment. In this case, you would supply --clip=75""")

	parser.add_argument("--iter", type=int, default=1, help="""Default=1. The number of iterations to perform.""", guitype='intbox', row=5, col=0, rowspan=1, colspan=1, nosharedb=True, mode='alignment,breaksym')
	
	parser.add_argument("--path",type=str,default='spt_refprep',help="""Default=spt. Directory to store results in. The default is a numbered series of directories containing the prefix 'spt'; for example, spt_02 will be the directory by default if 'spt_01' already exists.""")
		
	parser.add_argument("--npeakstorefine", type=int, help="""Default=1. The number of best coarse alignments to refine in search of the best final alignment.""", default=1, guitype='intbox', row=9, col=0, rowspan=1, colspan=1, nosharedb=True, mode='alignment,breaksym[1]')
			
	parser.add_argument("--preavgproc1",type=str,default='',help="""Default=None. A processor (see 'e2help.py processors -v 10' at the command line) to be applied to the raw particle after alignment but before averaging (for example, a threshold to exclude extreme values, or a highphass filter if you have phaseplate data.)""")
	
	parser.add_argument("--preavgproc2",type=str,default='',help="""Default=None. A processor (see 'e2help.py processors -v 10' at the command line) to be applied to the raw particle after alignment but before averaging (for example, a threshold to exclude extreme values, or a highphass filter if you have phaseplate data.)""")

	parser.add_argument("--parallel",default="thread:1",help="""default=thread:1. Parallelism. See http://blake.bcm.edu/emanwiki/EMAN2/Parallel""", guitype='strbox', row=19, col=0, rowspan=1, colspan=3, mode='alignment,breaksym')
	
	parser.add_argument("--ppid", type=int, help="""Default=-1. Set the PID of the parent process, used for cross platform PPID""",default=-1)
			
	parser.add_argument("--resume",type=str,default='',help="""(Not working currently). sptali_ir.json file that contains alignment information for the particles in the set. If the information is incomplete (i.e., there are less elements in the file than particles in the stack), on the first iteration the program will complete the file by working ONLY on particle indexes that are missing. For subsequent iterations, all the particles will be used.""")
															
	parser.add_argument("--plots", action='store_true', default=False,help="""Default=False. Turn this option on to generate a plot of the ccc scores during each iteration in.png format (otherwise only .txt files will be saved). This option will also produce a plot of mean ccc score across iterations. Running on a cluster or via ssh remotely might not support plotting.""")

	parser.add_argument("--savepreproc",action="store_true",  default=False,help="""Default=False. Will save stacks of preprocessed particles (one for coarse alignment and one for fine alignment if preprocessing options are different).""")
	
	parser.add_argument("--subset",type=int,default=0,help="""Default=0 (not used). Refine only this substet of particles from the stack provided through --input""")
	
	parser.add_argument("--savesteps",action="store_true", default=False, help="""Default=False. If set, this will save the average after each iteration to class_#.hdf. Each class in a separate file. Appends to existing files.""", guitype='boolbox', row=4, col=0, rowspan=1, colspan=1, mode='alignment,breaksym')
	
	parser.add_argument("--saveali",action="store_true", default=False, help="""Default=False. If set, this will save the aligned particle volumes in class_ptcl.hdf. Overwrites existing file.""", guitype='boolbox', row=4, col=1, rowspan=1, colspan=1, mode='alignment,breaksym')
	
	parser.add_argument("--saveallalign",action="store_true", default=False, help="""Default=False. If set, this will save an aligned stack of particles for each iteration""", guitype='boolbox', row=4, col=2, rowspan=1, colspan=1, mode='alignment,breaksym')
	
	#parser.add_argument("--saveallpeaks",action="store_true", default=False, help="""Default=False. If set, this will save the alignment information and score for all examined peaks --npeakstorefine during coarse alignment.""")
	
	parser.add_argument("--sym", type=str,dest = "sym", default='', help = """Default=None (equivalent to c1). Symmetry to impose -choices are: c<n>, d<n>, h<n>, tet, oct, icos""", guitype='symbox', row=9, col=1, rowspan=1, colspan=2, mode='alignment,breaksym')
	
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="""Default=0. Verbose level [0-9], higher number means higher level of verboseness; 10-11 will trigger many messages that might make little sense since this level of verboseness corresponds to 'debugging mode'""")

	parser.add_argument("--weighbytiltaxis",type=str,default='',help="""Default=None. A,B, where A is an integer number and B a decimal. A represents the location of the tilt axis in the tomogram in pixels (eg.g, for a 4096x4096xZ tomogram, this value should be 2048), and B is the weight of the particles furthest from the tiltaxis. For example, --weighbytiltaxis=2048,0.5 means that praticles at the tilt axis (with an x coordinate of 2048) will have a weight of 1.0 during averaging, while the distance in the x coordinates of particles not-on the tilt axis will be used to weigh their contribution to the average, with particles at the edge(0+radius or 4096-radius) weighing 0.5, as specified by the value provided for B.""")

	parser.add_argument("--weighbyscore",action='store_true',default=False,help="""Default=False. This option will weigh the contribution of each subtomogram to the average by score/bestscore.""")

	
	'''
	REFPREP SPECIFIC PARAMETERS
	'''
	parser.add_argument("--hacref",type=int,default=0,help="""Default=0 (not used by default). Size of the SUBSET of particles to use to build an initial reference by calling e2spt_hac.py which does Hierarchical Ascendant Classification (HAC) or 'all vs all' alignments.""") 
		
	parser.add_argument("--ssaref",type=int,default=0,help="""Default=0 (not used by default). Size of the SUBSET of particles to use to build an initial reference by calling e2symsearch3d.py, which does self-symmetry alignments. You must provide --sym different than c1 for this to make any sense.""")
		
	parser.add_argument("--btref",type=int,default=0,help="""Default=0 (internally turned on and set to 64). Size of the SUBSET of particles to use to build an initial reference by calling e2spt_binarytree.py. By default, the largest power of two smaller than the number of particles in --input will be used. For example, if you supply a stack with 150 subtomograms, the program will automatically select 128 as the limit to use because it's the largest power of 2 that is smaller than 150. But if you provide, say --btref=100, then the number of particles used will be 64, because it's the largest power of 2 that is still smaller than 100.""")
	
	parser.add_argument("--keep",type=float,default=1.0,help="""Default=1.0 (all particles kept). The fraction of particles to keep in each class.""", guitype='floatbox', row=6, col=0, rowspan=1, colspan=1, mode='alignment,breaksym')
	
	parser.add_argument("--keepsig", action="store_true", default=False,help="""Default=False. Causes the keep argument to be interpreted in standard deviations.""", guitype='boolbox', row=6, col=1, rowspan=1, colspan=1, mode='alignment,breaksym')

	(options, args) = parser.parse_args()
	
	options = checkinput( options )
	
	nptcl = EMUtil.get_image_count(options.input)
	
	if nptcl < 2:
		print("\nERROR: you need more than two particles in --input to particles")
	
	if not options.input:
		parser.print_help()
		exit(0)
	
	if options.subset:
		#print "there is subset!", options.subset
		
		if options.subset < nptcl:
			if options.verbose:
				print("\taking a subset s={} smaller than the number of particles in --input n={}".format(options.subset, nptcl))
			
	options = makepath(options,'spt_refprep')
		
	if options.subset < 4:
		print("ERROR: You need at least 4 particles in --input for goldstandard refinement if --ref is not provided and --goldstandardoff not provided.")
		sys.exit()
	
	if options.hacref and options.subset < options.hacref * 2:
		print("""WARNING: --subset=%d wasn't large enough to accommodate gold standard
		refinement with two independent halves using the specified number of particles
		for initial model generation --hacref=%d. Therefore, --hacref will be reset
		to --subset/2.""" %( options.subset, options.hacref ))				
		options.hacref = old_div(options.subset, 2)			

	elif options.ssaref and options.subset < options.ssaref * 2:		
		print("""WARNING: --subset=%d wasn't large enough to accommodate gold standard
		refinement with two independent halves using the specified number of particles
		for initial model generation --ssaref=%d. Therefore, --ssaref will be reset
		to --subset/2.""" %( options.subset, options.ssaref ))
		options.ssaref = old_div(options.subset, 2)	

	elif options.btref and options.subset < options.btref * 2:			
		print("""WARNING: --subset=%d wasn't large enough to accommodate gold standard
		refinement with two independent halves using the specified number of particles
		for initial model generation --btref=%d. Therefore, --btref has been reset
		to --subset/2.""" %( options.subset, options.btref ))
		options.btref = old_div(options.subset, 2)
		
	refsdict = sptRefGen( options, ptclnumsdict, cmdwp )
		
	return
	

	
'''
Function to generate the reference either by reading from disk or bootstrapping
'''
def sptRefGen( options, ptclnumsdict, cmdwp, refinemulti=0, method='',subset4ref=0):
	
	import glob, shutil
	
	refsdict = {}
	elements = cmdwp.split(' ')
	
	#print "elements are", elements
	#print "ptclnumsdict received in sptRefGen is", ptclnumsdict
	#print "RECEIVED CMDWP", cmdwp
	#print 'Therefore elemnts are', elements
	
	#current = os.getcwd()
	
	for klassnum in ptclnumsdict:
		
		klassidref = '_even'
		
		if klassnum == 1:
			klassidref = '_odd'
		
		try:
			if options.goldstandardoff:
				klassidref = ''
		except:
			if refinemulti:
				klassidref = ''
				
		
		if refinemulti:
			zfillfactor = len(str( len( ptclnumsdict )))
			
			#if ptclnumsdict[klassnum]:
			#else:
			
			klassidref = '_' + str( klassnum ).zfill( zfillfactor )
			if len( ptclnumsdict ) < 2:
				klassidref = '_' + str( refinemulti ).zfill( zfillfactor )	
		
		if options.ref: 
			func_refrandphase(options)
			
		
		elif not options.ref:
			ptclnums = ptclnumsdict[ klassnum ]
			#print "Therefore for class", klassnum
			#print "ptclnums len and themsvels are", len(ptclnums), ptclnums
		
			if ptclnums:
				ptclnums.sort()		
			
			try:
				if options.hacref:
					method = 'hac'
			except:
				pass
			
			try:
				if options.btref:
					method = 'bt'
			except:
				pass
			
			try:
				if options.ssaref:
					method = 'ssa'
			except:
				pass
				
			if not method and not options.ref:
				method = 'bt'
				print("\n\n\nbt by default!!!!")
				
			#elif options.hacref:
			if method == 'hac':
				pass
		
			if method == 'ssa':
				pass
								
			if method == 'bt':
				pass
				
	refnames={}
	

	return refsdict
	

def func_refrandphase(options,klassidref,klassnum):	
	
	ref = EMData(options.ref,0)

	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - READ reference and its types and min, max, sigma, mean stats are", options.ref, type(ref), ref['minimum'],ref['maximum'],ref['sigma'],ref['mean'])

	if not ref['maximum'] and not ref['minimum']:
		print("(e2spt_classaverage)(sptRefGen) - ERROR: Empty/blank reference file. Exiting.")
		sys.exit()
	
	if options.apix:
		ref['apix_x'] = options.apix
		ref['apix_y'] = options.apix
		ref['apix_z'] = options.apix
	
	if int(options.refrandphase) > 0:
		filterfreq =  old_div(1.0,float( options.refrandphase ))
		ref.process_inplace("filter.lowpass.randomphase",{"cutoff_freq":filterfreq,"apix":ref['apix_x']})
				
		refrandphfile = options.path + '/' + os.path.basename( options.ref ).replace('.hdf','_randPH' + klassidref +'.hdf')
		
		if 'final_avg' in refrandphfile:								#you don't want any confusion between final averages produces in other runs of the program and references
			refrandphfile = refrandphfile.replace('final_avg','ref')

		ref['origin_x'] = 0
		ref['origin_y'] = 0
		ref['origin_z'] = 0
		ref.write_image( refrandphfile, klassnum )

	if float(ref['apix_x']) <= 1.0:
		print("\n(e2spt_classaverage)(sptRefGen) - WARNING: apix <= 1.0. This is most likely wrong. You might want to fix the reference's apix value by providing --apix or by running it through e2procheader.py")
	
	#refsdict.update({ klassnum : ref })
			
	return refrandphfile


def func_hacref():
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Generating initial reference using hierarchical ascendant classification through e2spt_hac.py")

	subsetForHacRef = 'spthacrefsubset'+ klassidref + '.hdf'
	
	try:
		os.remove(subsetForHacRef)
	except:
		pass
					
	i = 0
	nptclsforref = 10
	try:
		if options.hacref:
			nptclsforref = options.hacref								
	except:
		if subset4ref:
			nptclsforref=subset4ref

	if nptclsforref >= len(ptclnums):
		nptclsforref =  len(ptclnums)

	print("Hacreflimit is", nptclsforref)
	if nptclsforref < 3:
		print("""ERROR: You cannot build a HAC reference with less than 3 particles.
		Either provide a larger --hacref number, a larger --subset number, or provide
		--goldstandardoff""")
	
		sys.exit()

	i = 0
	while i < nptclsforref :
		a = EMData( options.input, ptclnums[i] )
		a.write_image( subsetForHacRef, i )
		i+=1

	niterhac = nptclsforref - 1

	hacelements = []
	for ele in elements:
		if 'saveallpeaks' not in ele and 'raw' not in ele and 'btref' not in ele and 'hacref' not in ele and 'ssaref' not in ele and 'subset4ref' not in ele and 'refgenmethod' not in ele and 'nref' not in ele and 'output' not in ele and 'fsc' not in ele and 'subset' not in ele and 'input' not in ele and '--ref' not in ele and 'path' not in ele and 'keep' not in ele and 'iter' not in ele and 'subset' not in ele and 'goldstandardoff' not in ele and 'saveallalign' not in ele and 'savepreproc' not in ele:
			hacelements.append(ele)

	cmdhac = ' '.join(hacelements)
	cmdhac = cmdhac.replace('e2spt_classaverage','e2spt_hac')

	if refinemulti:
		cmdhac = cmdhac.replace('e2spt_refinemulti','e2spt_hac')
	

	hacrefsubdir = 'spthacref' + klassidref
	
	
	try:
		files=glob.glob(hacrefsubdir+'*')		
		print("files are", files)
		for path in files: 
			shutil.rmtree(path)
	except:
		pass
		
	cmdhac+=' --path=' + hacrefsubdir
	cmdhac+=' --input='+subsetForHacRef
	
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Command to generate hacref is", cmdhac)
	
	runcmd( options, cmdhac )

	try:
		print("\nmoving hacrefsubdir %s into path %s" %( hacrefsubdir, options.path ))
		os.rename( hacrefsubdir, options.path + '/' + hacrefsubdir )
	except:
		print("\nfirst try moving hacrefsubdir %s into path %s failed" %( hacrefsubdir, options.path ))
		hacsubdirstem = '_'.join( hacrefsubdir.split('_')[:-1])

		try:
			hacsubdircount = str( int(hacrefsubdir.split('_')[-1])+1)
		except:
			hacsubdircount = '01'
			hacsubdirstem = hacrefsubdir
		

		newhacrefsubdir = hacsubdirstem + '_' + hacsubdircount #if the subdirectory exists, add one to the tag count at the end of the subdirectory name
		try: 
			print("\nmoving hacrefsubdir %s into path %s" %( hacrefsubdir, options.path ))
			os.rename( newhacrefsubdir, options.path + '/' + hacrefsubdir )
		except:
			print("\nsecond and final try moving hacrefsubdir %s into path %s failed" %( newhacrefsubdir, options.path ))
			sys.exit(1)

	
	
	try:
		os.rename( subsetForHacRef, options.path + '/' + subsetForHacRef )
	except:
		newsubsetcount = '_'.join( subsetForHacRef.split('_')[:-1]) + '_' + str( int(subsetForHacRef.split('_')[-1])+1 )	#if the subdirectory exists, add one to the tag count at the end of the subdirectory name
		os.rename( subsetForHacRef, options.path + '/' + newsubsetcount )
	
	
	findir = os.listdir(options.path)
	if subsetForHacRef in findir:
		currentdir = os.getcwd()
		findircurrent = os.listdir(currentdir)
		if subsetForHacRef in findircurrent:
			newsubsetcount = '_'.join( subsetForHacRef.split('_')[:-1]) + '_' + str( int(subsetForHacRef.split('_')[-1].split('.hdf')[0]) +1 ) +'.hdf'	#if the subdirectory exists, add one to the tag count at the end of the subdirectory name
			print("\ntrying to move new subsetForHacRef into path", newsubsetcount, options.path)
			os.rename( subsetForHacRef, options.path + '/' + newsubsetcount )
		else:
			print("\nWARNING subsetForHacRef does not exist in current directory", subsetForHacRef)
			
	else:
		os.rename( subsetForHacRef, options.path + '/' + subsetForHacRef )
		print("\nmoving subsetForHacRef into path", subsetForHacRef, options.path)
	
	
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Command to generate hacref is", cmdhac)
	
	ref = EMData( options.path + '/' + hacrefsubdir +'/final_avg.hdf', 0 )

	refsdict.update({ klassnum : ref })

	return
	

def func_btref():
	nptclsforref = 64
				
	#try:
	#	if options.btref:
	#		nptclsforref = options.btref		
	#except:
	#	if subset4ref:
	#		nptclsforref = subset4ref

	#if nptclsforref >= len(ptclnums):
	#	nptclsforref =  len(ptclnums)
	
	
	
	#from e2spt_binarytree import binaryTreeRef
	print("\ninput is", options.input)
	print("with nimgs", EMUtil.get_image_count( options.input ))
	print("--goldstandardoff is", options.goldstandardoff)
	print("len ptclnums is", len(ptclnums))
	
	print("log 2 of that is") 
	print(log( len(ptclnums), 2 ))

	niter = int(floor(log( len(ptclnums), 2 )))
	print("and niter is", niter)
	nseed = 2**niter
	print("therefore nseed=2**niter is", nseed)
	
	
	#try:
	#	if options.btref:
	#		niter = int(floor(log( options.btref, 2 )))
	#		nseed=2**niter			
	#except:
	#	if subset4ref:
	#		niter = int(floor(log( subset4ref, 2 )))
	#		nseed=2**niter	
	
	
	#if not options.goldstandardoff:
	#	nseed /= 2
	
		
	subsetForBTRef = 'sptbtrefsubset'+ klassidref + '.hdf'
	
	try:
		os.remove( subsetForBTRef )
	except:
		pass
					
	i = 0
	
	
	#print "ptclnums are", ptclnums
	#print "with len", len(ptclnums)
	
	while i < nseed :
		print("i is", i)
		a = EMData( options.input, ptclnums[i] )
		a.write_image( subsetForBTRef, i )
		print("writing image %d to file %s, which will contain the subset of particles used for BTA reference building" %(i,subsetForBTRef))
		i+=1

	btelements = []
	#print "elements are", elements
	for ele in elements:
		if 'saveallpeaks' not in ele and 'raw' not in ele and 'btref' not in ele and 'hacref' not in ele and 'ssaref' not in ele and 'subset4ref' not in ele and 'refgenmethod' not in ele and 'nref' not in ele and 'output' not in ele and 'fsc' not in ele and 'subset' not in ele and 'input' not in ele and '--ref' not in ele and 'path' not in ele and 'keep' not in ele and 'iter' not in ele and 'goldstandardoff' not in ele and 'saveallalign' not in ele and 'savepreproc' not in ele:
			#print "added ele", ele
			btelements.append(ele)
		else:
			pass
			#print "skipped ele", ele

	cmdbt = ' '.join(btelements)
	cmdbt = cmdbt.replace('e2spt_classaverage','e2spt_binarytree')

	#print "wildcard is!", wildcard
	#print "BEFORE replacement", cmdbt

	if refinemulti:
		cmdbt = cmdbt.replace('e2spt_refinemulti','e2spt_binarytree')


	btrefsubdir = 'sptbtref' + klassidref		
	
	try:
		files=glob.glob(btrefsubdir+'*')
		for path in files:
			shutil.rmtree(path)
	except:
		pass
		
			
	cmdbt+=' --path=' + btrefsubdir
	#cmdbt+=' --iter=' + str( niter )
	cmdbt+=' --input=' + subsetForBTRef
	#cmdbt+= ' --nopreprocprefft'
	
	runcmd( options, cmdbt )
	
	#cmdbt+= ' && mv ' + btrefsubdir + ' ' + options.path + '/' + ' && mv ' + subsetForBTRef + ' ' + options.path

	
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Command to generate btref is", cmdbt)

	try:
		print("\nmoving btrefsubdir into path", btrefsubdir, options.path)
		os.rename( btrefsubdir, options.path + '/' + btrefsubdir )
	except:
		print("\nfirst try moving btrefsubdir %s into path %s failed" %( btrefsubdir, options.path ))
		
		btsubdirstem = '_'.join( btrefsubdir.split('_')[:-1])

		try:
			btsubdircount = str( int(btrefsubdir.split('_')[-1])+1)
		except:
			btsubdircount = '01'
			btsubdirstem = btrefsubdir
		

		newbtrefsubdir = btsubdirstem + '_' + btsubdircount #if the subdirectory exists, add one to the tag count at the end of the subdirectory name
		try: 
			os.rename( newbtrefsubdir, options.path + '/' + btrefsubdir )
		except:
			print("\nsecond and final try moving btrefsubdir %s into path %s failed" %( newbtrefsubdir, options.path ))
			sys.exit(1)
			
					
	
	#cmdbt3 = 'mv ' + subsetForBTRef + ' ' + options.path
	#runcmd( options, cmdbt3 )
	
	findir = os.listdir(options.path)
	if subsetForBTRef in findir:
		print("tried moving subsetForBTRef into path but failed", subsetForBTRef, options.path)
		newsubsetcount = '_'.join( subsetForBTRef.split('_')[:-1]) + '_' + str( int(subsetForBTRef.split('_')[-1].split('.hdf')[0]) +1 ) +'.hdf'	#if the subset exists, add one to the tag count at the end of the subdirectory name
		os.rename( subsetForBTRef, options.path + '/' + newsubsetcount )
	else:
		os.rename( subsetForBTRef, options.path + '/' + subsetForBTRef )
		print("\nmoving subsetForBTRef into path", subsetForBTRef, options.path)
					
	

	#if os.getcwd() not in options.path:
	#	options.path = os.getcwd() + '/' + ptions.path

	print("\ncmdbt is", cmdbt)

	#print "\nfindir are"
	#findir=os.listdir(current)
	#for f in findir:
	#	print f

	print("The BT reference to load is in file",  options.path+ '/' +btrefsubdir +'/final_avg.hdf')
	ref = EMData( options.path + '/' + btrefsubdir +'/final_avg.hdf', 0 )

	refsdict.update({ klassnum : ref })



	return
	

def func_ssaref():
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Generating initial reference using self symmetry alignment through e2symsearch3d.py")

	if options.sym == 'c1' or options.sym == 'C1':
		print("""\n(e2spt_classaverage)(sptRefGen) - ERROR: You must provide at least c2 or higher symmetry to use
		--ssaref""")

	subsetForSsaRef = 'sptssarefsubset'+ klassidref + '.hdf'
	
	try:
		os.remove( subsetForSsaRef )
	except:
		pass
					
	nptclsforref = 10
	try:
		if options.ssaref:
			nptclsforref=options.ssaref	
	except:
		if subset4ref:
			nptclsforref=subset4ref

	if nptclsforref >= len(ptclnums):
		nptclsforref =  len(ptclnums)

	i = 0
	while i < nptclsforref :
		a = EMData( options.input, ptclnums[i] )
		a.write_image( subsetForSsaRef, i )
		i+=1

	ssarefsubdir = 'sptssaref' + klassidref
	
	try:
		files=glob.glob(ssarefsubdir+'*')
		for path in files:
			shutil.rmtree(path)
	except:
		pass
		
			
	ssaelements = []
	print("\nelements are", elements)
	for ele in elements:
		if 'saveallpeaks' not in ele and 'fine' not in ele and 'raw' not in ele and 'btref' not in ele and 'hacref' not in ele and 'ssaref' not in ele and 'subset4ref' not in ele and 'refgenmethod' not in ele and 'nref' not in ele and 'sfine' not in ele and 'procfine' not in ele and 'fsc' not in ele and 'output' not in ele and 'path' not in ele and 'goldstandardoff' not in ele and 'saveallalign' not in ele and 'savepreproc' not in ele and 'align' not in ele and 'iter' not in ele and 'npeakstorefine' not in ele and 'precision'not in ele and '--radius' not in ele and 'randphase' not in ele and 'search' not in ele and '--save' not in ele and '--ref' not in ele and 'input' not in ele and 'output' not in ele and 'subset' not in ele:
			ssaelements.append(ele)
			print("appending element",ele)

	cmdssa = ' '.join(ssaelements)
	print("before replacing program name, cmdssa is", cmdssa)

	cmdssa = cmdssa.replace('e2spt_classaverage','e2symsearch3d')
	if refinemulti:
		print("should replace refinemulti")
		cmdssa = cmdssa.replace('e2spt_refinemulti','e2symsearch3d')

	cmdssa += ' --input=' + subsetForSsaRef 
	cmdssa += ' --path=' + ssarefsubdir
	cmdssa += ' --symmetrize'
	cmdssa += ' --average'
	
	print("\ncmdssa is", cmdssa)
	
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Command to generate ssaref is", cmdssa)

	runcmd( options, cmdssa )
	
	ssarefname = 'final_avg.hdf'
		
	try:
		print("\nmoving ssarefsubdir %s into path %s" %( ssarefsubdir, options.path ))
		os.rename( ssarefsubdir, options.path + '/' + ssarefsubdir )
	except:
		print("\nfirst try moving ssarefsubdir %s into path %s failed" %( ssarefsubdir, options.path ))
		hacsubdirstem = '_'.join( ssarefsubdir.split('_')[:-1])

		try:
			ssasubdircount = str( int(ssarefsubdir.split('_')[-1])+1)
		except:
			ssasubdircount = '01'
			ssasubdirstem = ssarefsubdir
		
		newssarefsubdir = ssasubdirstem + '_' + ssasubdircount #if the subdirectory exists, add one to the tag count at the end of the subdirectory name
		try: 
			print("\nmoving ssarefsubdir %s into path %s" %( ssarefsubdir, options.path ))
			os.rename( newssarefsubdir, options.path + '/' + ssarefsubdir )
		except:
			print("\nsecond and final try moving ssarefsubdir %s into path %s failed" %( newssarefsubdir, options.path ))
			sys.exit(1)




	findir = os.listdir(options.path)
	if subsetForSsaRef in findir:
		print("tried moving subsetForSsaRef into path but failed", subsetForSsaRef, options.path)
		newsubsetcount = '_'.join( subsetForSsaRef.split('_')[:-1]) + '_' + str( int(subsetForSsaRef.split('_')[-1].split('.hdf')[0]) +1 ) +'.hdf'	#if the subdirectory exists, add one to the tag count at the end of the subdirectory name
		os.rename( subsetForSsaRef, options.path + '/' + newsubsetcount )
	else:
		os.rename( subsetForSsaRef, options.path + '/' + subsetForSsaRef )
		print("\nmoving subsetForSsaRef into path", subsetForSsaRef, options.path)
	
	
	if options.verbose:
		print("\n(e2spt_classaverage)(sptRefGen) - Command to generate ssaref is", cmdssa)

	#p=subprocess.Popen( cmdssa, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	#text=p.communicate()	
	#p.stdout.close()

	ref = EMData( options.path + '/' + ssarefsubdir +'/' + ssarefname, 0 )

	refsdict.update({ klassnum : ref })

	#elif not options.hacref and not options.ssaref:
	return
	
			
if '__main__' == __name__:
	main()


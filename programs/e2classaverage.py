#!/usr/bin/env python

#
# Author: Steven Ludtke, 2007 (sludtke@bcm.edu)
# Copyright (c) 2000-2007 Baylor College of Medicine
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

from EMAN2 import *
from optparse import OptionParser
from math import *
from copy import deepcopy
import os
import sys

READ_HEADER_ONLY = True

def main():
	progname = os.path.basename(sys.argv[0])
	usage = """%prog <input particles> <class mx> <output> [options]

	Produces class averages.
	Can perform iterative alignment.
	Provides bootstrapping functionality.
	"""
		
	parser = OptionParser(usage=usage,version=EMANVERSION)

	parser.add_option("--iter", type="int", help="The number of iterations to perform. Default is 0.", default=0)
	parser.add_option("--ref", type="string", help="Reference image. If specified, the metadata in this image is used to assign euler angles to the generated classes. This is typically the projections that were used for the classification.", default=None)
	parser.add_option("--align",type="string",help="This is the aligner used to align particles to the previous class average. Default is None.", default=None)
	parser.add_option("--aligncmp",type="string",help="The comparitor used for the --align aligner. Default is dot.",default="dot:normalize=1")
	parser.add_option("--ralign",type="string",help="This is the second stage aligner used to refine the first alignment. This is usually the \'refine\' aligner.", default=None)
	parser.add_option("--raligncmp",type="string",help="The comparitor used by the second stage aligner.",default="dot:normalize=1")
	parser.add_option("--averager",type="string",help="The type of averager used to produce the class average.",default="image")
	parser.add_option("--cmp",type="string",help="The comparitor used to generate quality scores for the purpose of particle exclusion in classes, strongly linked to the keep argument.", default="dot:normalize=1")
	parser.add_option("--keep",type="float",help="The fraction of particles to keep in each class.",default=1.0)
	parser.add_option("--keepsig", action="store_true", help="Causes the keep argument to be interpreted in standard deviations.",default=False)
	parser.add_option("--verbose","-v",action="store_true",help="Print useful information while the program is running. Default is off.",default=False)
	parser.add_option("--force", "-f",dest="force",default=False, action="store_true",help="Force overwrite the output file if it exists.")
	parser.add_option("--debug","-d",action="store_true",help="Print debugging infromation while the program is running. Default is off.",default=False)
	parser.add_option("--nofilecheck",action="store_true",help="Turns file checking off in the check functionality - used by e2refine.py.",default=False)
	parser.add_option("--check","-c",action="store_true",help="Performs a command line argument check only.",default=False)
	parser.add_option("--lowmem","-L",action="store_true",help="Causes images to be read from disk as they are needed, as opposed to having them all read from disk and stored in memory for the duration of the program. Saves on memory but causes more disk accesses.",default=False)
	parser.add_option("--bootstrap",action="store_true",help="Bootstraps iterative alignment by using the first particle in each class to seed the iterative alignment. Only works if the number of iterations is greater than 0.")
	parser.add_option("--resultmx",type="string",help="Specify an output image to store the result matrix. This contains 5 images where row is particle number. Rows in the first image contain the class numbers and in the second image consist of 1s or 0s indicating whether or not the particle was included in the class. The corresponding rows in the third, fourth and fifth images are the refined x, y and angle (respectively) used in the final alignment, these are updated and accurate, even if the particle was excluded from the class.", default=None)

	(options, args) = parser.parse_args()
	
	if len(args)<3 : parser.error("Input, classification matix, and output files required")
	
	if (options.check): options.verbose = True # turn verbose on if the user is only checking...
		
	if (options.nofilecheck == False):
		options.classifyfile=args[1]
		if ( options.ref ):
			options.ref=options.ref
		options.datafile = args[0]
		options.outfile = args[2]
	
	error = check(options,True)
	
	if (options.verbose):
		if (error):
			print "e2classaverage.py command line arguments test.... FAILED"
		else:
			print "e2classaverage.py command line arguments test.... PASSED"
	
	# returning a different error code is currently important to e2refine.py - returning 0 tells e2refine.py that it has enough
	# information to execute this script
	if error : exit(1)
	if options.check: exit(0)
	
	logger=E2init(sys.argv)
	
	# just remove the file - if the user didn't specify force then the error should have been found in the check function
	if os.path.exists(options.outfile):
		if options.force:
			remove_file(options.outfile)
	
	(num_classes, num_part ) = gimme_image_dimensions2D(args[1]);
	
	if ( not options.lowmem ) :
		images = EMData.read_images(args[0])
	
	
	if (options.verbose):
		print "Classifications per particle %d, particles %d" %(num_classes, num_part)
	
	# classes contains the classifications - row is particle number, column data contains class numbers (could be greater than 1)
	classes = EMData()
	classes.read_image(args[1], 0)
	class_max = int(classes.get_attr("maximum"))
	class_min = int(classes.get_attr("minimum"))
	
	# double check that the argument reference image makes sense
	if (options.ref):
		if not os.path.exists(options.ref):
			parser.error("File %s does not exist" %options.ref)
			
		num_ref= EMUtil.get_image_count(options.ref)
		if ( class_max > num_ref ):
			print "Error, the classification matrix refers to a class number (%d) that is beyond the number of images (%d) in the reference image (%s)." %(class_max,num_ref,options.ref)
			exit(1)
	
	# double check that the number of particles in the particle image matches the rows in the classification matrix (image)
	num_part_check =  EMUtil.get_image_count(args[0])
	if ( num_part != num_part_check ):
		print "Error, the number of rows (%d) in the classification matrix (image) does not match the number of particles (%d) in the input image." %(num_part,num_part_check)
		exit(1)
	
	# weights contains the weighting of the classification scheme stored in the EMData object "classes" - above
	# row is particle number, column data contains weights - rows should add to 1, but this is not checked.
	weights = EMData()
	weights.read_image(args[1], 1)
	# dx contains the x translation of the alignment
	# row is particle number, column is class idx (to class number stored in classes)
	dx = EMData()
	dx.read_image(args[1],2)
	# dy contains the y translation of the alignment
	# row is particle number, column is class idx (to class number stored in classes)
	dy = EMData()
	dy.read_image(args[1],3)
	# da contains is the azimuthal rotation of the alignment
	# row is particle number, column is class idx (to class number stored in classes)
	da = EMData()
	da.read_image(args[1],4)
	
	# empty space for flipping data
	dflip = EMData(da.get_xsize(),da.get_ysize())
	dflip.to_zero()
	
	if (options.iter > 0 or options.bootstrap):
		setAlignOpts(options)
	
	# do one class at a time
	for cl in range(class_min,class_max+1):
		if (options.verbose):
			ndata = []
		
		if (options.iter > 0 or options.verbose): ccache = [] # class cache
		
		averager_parms=parsemodopt(options.averager)
		
		if ( not options.bootstrap ):
			# generate the first class average by applying the transformations, adding and finally normalizing...
			averager=Averagers.get(averager_parms[0], averager_parms[1])
			# do the initial average, based on the program inputs
			weightsum = 0 # used to normalize the average
			np = 0 # number of particles in the average
			for p in range(0,num_part):
				for c in range(0,num_classes):
					if classes.get(c,p) == cl:
						# cache the hit if necessary
						if (options.iter > 0 or options.verbose): ccache.append((p,c))
						
						# Position the image correctly
						print da.get(c,p)
						t3d = Transform3D(EULER_EMAN,-da.get(c,p),0,0)
						t3d.set_posttrans(dx.get(c,p),dy.get(c,p))
						if (options.lowmem):
							image = EMData()
							image.read_image(args[0],p)
						else:
							image = images[p].copy()
						image.rotate_translate(t3d)
						
						np += 1
						weight = weights(c,p)
						weightsum += weight
						image.mult(weight)
						
						# Add the image to the averager
						averager.add_image(image)
						
			
			
			if np == 0 or weightsum == 0:
				if options.verbose:
					print "Class",cl,"...no particles"
				# FIXME
				# write blank image? Write meta data?
				continue
				
			average = averager.finish()
			average.mult(float(np)) # Undo the division of np by the averager - this was incorrect because the particles were weighted.
			average.mult(1.0/weightsum) # Do the correct division
			average.process_inplace("xform.centeracf")
		else:
			# generate a bootstrapped initial average. Do this 'inductively' by aligning the 2nd image to the first, then averaging.
			# Then align the 3rd image to the average, and average again etc... until all the particles have been aligned and contribute
			# to the (running) average.
			average = None
			np = 0
			for p in range(0,num_part):
				for c in range(0,num_classes):
					if classes.get(c,p) == cl:
						# cache the hit if necessary
						if (options.iter > 0 or options.verbose): ccache.append((p,c))
						
						if (average == None):
							if (options.lowmem):
								average = EMData()
								average.read_image(args[0],p)
							else:
								average = images[p].copy()
								#average.process_inplace("xform.centerofmass")
								#average.process_inplace("mask.sharp",{"outer_radius":average.get_xsize()/2})
							np = 1
						else:
							if (options.lowmem): 
								image = EMData()
								image.read_image(args[0],p)
							else: image = images[p]
					
							ta = align(image,average,options)
							
							#ta.process_inplace("mask.sharp",{"outer_radius":ta.get_xsize()/2})
							
							np += 1
							#frac = 1.0/float(np)
							#omfrac = 1.0 - frac
							#ta.mult(frac) # be careful about the weighting
							#average.mult(omfrac) # be carefult about the weighting
							average.add(ta) # now add the image
							#if ta.get_attr('align.flip') != 0:
								#image.process_inplace("xform.flip", {"axis":"x"});
							#average.process_inplace("xform.centerofmass")
			
			average/=np
			average.process_inplace("xform.centeracf")
			#average.write_image("avg.img",-1)
			#average.process_inplace("mask.sharp",{"outer_radius":average.get_xsize()/2})
			#average.process_inplace("normalize.edgemean")
					
		if np == 0:
			if options.verbose:
				print "Class",cl,"...no particles"
			# FIXME
			# write blank image? Write meta data?
			continue
		
	
		if (options.iter > 0):
			options.cull = True
			if ( options.keepsig == False and options.keep == 1.0 ) : options.cull = False
		
		for it in range(0,options.iter):
			itfrac=it/float(options.iter)
			# do alignment
			for d in ccache:
				p = d[0]
				c = d[1]
				if (options.lowmem): 
					image = EMData()
					image.read_image(args[0],p)
				else: image = images[p]
					
				ta = align(image,average,options)
				#ta.write_image("ta"+str(cl)+".img",-1)
				
				# store the refined translational and rotational values
				dx.set(c,p, ta.get_attr_default("align.dx",0))
				dy.set(c,p, ta.get_attr_default("align.dy",0))
				da.set(c,p, ta.get_attr_default("align.az",0))
				try: dflip.set(c,p, ta.get_attr_default("align.flip",0))
				except:pass
				# store the quality score on top of the weights, seeing as the starting weights are no longer required
				
				try:
					if ta.get_attr('align.flip') != 0:
						image.process_inplace("xform.flip", {"axis":"x"});
				except:pass
				if (options.cull): # but only if we need to
					weights.set(c,p, ta.cmp(options.cmp[0],average,options.cmp[1]))
			
			# get the culling threshold
			if options.cull:
				qual_scores = []
				for d in ccache:
					p = d[0]
					c = d[1]
					qual_scores.append(weights.get(c,p))
				
				if ( options.keepsig ):
					a = Util.get_stats_cstyle(qual_scores)
					mean = a["mean"]
					std_dev = a["std_dev"]
					cullthresh = mean + (5.0*(1.0-itfrac)+itfrac*options.keepsig)*std_dev
				else:
					b = deepcopy(qual_scores)
					b.sort()
					# The ceil reflects a conservative policy. If the user specified keep=0.93
					# and there were 10 particles, then they would all be kept. If floor were
					# used instead of ceil, the last particle would be thrown away (in the
					# class average)
					idx = int(ceil(((1.0-itfrac)+itfrac*options.keep)*len(b))-1)
					cullthresh = b[idx]
			
			#finally average
			averager=Averagers.get(averager_parms[0], averager_parms[1]) # need an empty averager
			np = 0 # number of particles in the average
			for d in ccache:
				p = d[0]
				c = d[1]
				if (options.cull ):
					if ( weights.get(c,p) > cullthresh ) :
						weights.set(c,p,0)
						continue
					else: weights.set(c,p,1)
				else: weights.set(c,p,1)
				
				t3d = Transform3D(EULER_EMAN,da.get(c,p),0,0)
				t3d.set_posttrans(dx.get(c,p),dy.get(c,p))
				
				if (options.lowmem):
					image = EMData()
					image.read_image(args[0],p)
				else:
					image = images[p].copy()
				
				try:
					if dflip.get(c,p) != 0:
						image.process_inplace("xform.flip", {"axis":"x"});
				except:pass
					
				image.rotate_translate(t3d)
				#image.process_inplace("mask.sharp",{"outer_radius":image.get_xsize()/2})
				np += 1
				averager.add_image(image)
				
				
			
			if options.verbose:
				ndata.append(np)
		
			if np == 0:
				if (options.verbose):
					print "Class",cl,"...no particles on iteration",it
				# FIXME
				# write blank image? Write meta data?
				continue
		
			average = averager.finish()
			#average.write_image("avg.img",-1)
			average.process_inplace("xform.centerofmass")
			
		# extract euler data from the ref image, if it was specified
		if ( options.ref  ):
			e = EMData()
			e.read_image(options.ref, cl, READ_HEADER_ONLY)
			average.set_attr("euler_alt", e.get_attr("euler_alt"))
			average.set_attr("euler_az",e.get_attr("euler_az"))
			average.set_attr("euler_phi",e.get_attr("euler_phi"))
			if options.verbose:
				edata = []
				edata.append(e.get_attr("euler_alt"))
				edata.append(e.get_attr("euler_az"))
				edata.append(e.get_attr("euler_phi"))
		
		# now write to disk
		average.set_attr("ptcl_repr",np)
		average.write_image(args[2],-1)
			
		if options.verbose:
			sys.stdout.write( "Class %d: particles..%d" %(cl,len(ccache)) )
			for t in range(0,options.iter):
				sys.stdout.write("..%d" %ndata[t] )
			if ( options.ref  ):
				sys.stdout.write(" : Eulers..")
				for t in edata:
					sys.stdout.write(" %f" %t)
			sys.stdout.write("\n")
		
	if (options.resultmx != None ):
		if os.path.exists(options.resultmx):
			remove_file(options.resultmx) #oooh not sure about this!
		
		# note the order is important!
		classes.write_image(options.resultmx,-1)
		weights.write_image(options.resultmx,-1)
		dx.write_image(options.resultmx,-1)
		dy.write_image(options.resultmx,-1)
		da.write_image(options.resultmx,-1)
		
	
	E2end(logger)

def setAlignOpts(options):
	'''
	Call this before calling align
	'''
	options.align=parsemodopt(options.align)
	options.alicmp=parsemodopt(options.aligncmp)
	# note the parsing of the options.ralign parameters is left for later
	options.alircmp=parsemodopt(options.raligncmp)
	options.cmp=parsemodopt(options.cmp)

def align(this,to,options):
	# Align the particle to the average
	ta=this.align(options.align[0],to,options.align[1],options.alicmp[0],options.alicmp[1])
	
	if ( options.ralign != None ): # potentially employ refine alignment
		
		refineparms=parsemodopt(options.ralign)
		# this parameters I think west best with the refine aligner, but they're not well tested
		# i.e. I need to do some rigorous testing before I can claim anything
		#refineparms[1]["az"] = ta.get_attr_default("align.az",0)-1
		#refineparms[1]["dx"] = ta.get_attr_default("align.dx",0)-1
		#refineparms[1]["dy"] = ta.get_attr_default("align.dy",0)-1
		#refineparms[1]["mode"] = 0
		#refineparms[1]["stepx"] = 2
		#refineparms[1]["stepy"] = 2
		#refineparms[1]["stepaz"] = 5
		
		refineparms[1]["az"] = ta.get_attr_default("align.az",0)
		refineparms[1]["dx"] = ta.get_attr_default("align.dx",0)
		refineparms[1]["dy"] = ta.get_attr_default("align.dy",0)
		
		ta = this.align(refineparms[0],to,refineparms[1],options.alircmp[0],options.alircmp[1])
	
	return ta
		
def check(options, verbose=False):

	error = False
	if ( options.nofilecheck == False ):
		
		if os.path.exists(options.outfile):
			if not options.force:
				error = True
				if (verbose):
					print "Error: output file %s exists, force not specified, will not overwrite, exiting" %options.outfile
		
		if not os.path.exists(options.classifyfile):
			error = True
			if (verbose):
				print "Error: the file expected to contain the classification matrix (%s) was not found, cannot run e2classaverage.py" %(options.classifyfile)
		
		if not os.path.exists(options.datafile):
			error = True
			if (verbose):
				print "Error: the file containing the particle data (%s) does not exist" %options.datafile
		
		if os.path.exists(options.classifyfile) and os.path.exists(options.datafile):
			(xsize, ysize ) = gimme_image_dimensions2D(options.classifyfile);
			numimg = EMUtil.get_image_count(options.datafile)
			if ( numimg != ysize ):
				error = True
				if (verbose):
					print "Error - the number of rows (%d) in the classification matrix image %s does not match the number of images (%d) in %s" %(ysize, options.classifyfile,numimg,options.datafile)
				
			
		if options.ref != None and not os.path.exists(options.ref):
			print "Error: the file expected to contain the reference images (%s) does not exist" %(options.ref)
			error = True
			if os.path.exists(options.ref) and os.path.exists(options.datafile):
				(xsize, ysize ) = gimme_image_dimensions2D(options.datafile);
				(pxsize, pysize ) = gimme_image_dimensions2D(options.ref);
				if ( xsize != pxsize ):
					error = True
					if (verbose):
						print "Error - the dimensions of the reference and particle images do not match"

	
	#if ( options.keep and options.keepsig ):
		#error = True
		#if ( verbose ):
			#print "Error: --keep and --keepsig are mutually exclusive"
	
	if ( options.keep > 1 or options.keep <= 0) and not options.keepsig :
		error = True
		if (verbose):
			print "The --keep option is a percentage expressed as a fraction - it must be between 0 and 1"
	
	if ( options.iter < 0 ):
		error = True
		if (verbose):
			print "Error, --iter must be greater than or equal to 0 - you specified %d" %(options.iter)
		
	if ( check_eman2_type(options.averager,Averagers,"Averager") == False ):
		error = True
	
	if ( options.iter > 0 ):
		
		if ( check_eman2_type(options.cmp,Cmps,"Comparitor") == False ):
			error = True
			
		if (options.align == None):
			print "If --classiter is greater than zero, the -align argument must be specified"
			error = True
			
		if ( check_eman2_type(options.align,Aligners,"Aligner") == False ):
			error = True

		if ( check_eman2_type(options.aligncmp,Cmps,"Comparitor") == False ):
			error = True
		
		if ( options.ralign != None ):
			if ( check_eman2_type(options.ralign,Aligners,"Aligner") == False ):
				error = True
				
			if ( check_eman2_type(options.raligncmp,Cmps,"Comparitor") == False ):
				error = True

	return error
	
if __name__ == "__main__":
    main()

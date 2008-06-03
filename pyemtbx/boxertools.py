#!/usr/bin/env python
#
# Author:  David Woolford 06/02/2008 (woolford@bcm.edu)
# Copyright (c) 2000-2008 Baylor College of Medicine
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
from time import time
from math import ceil
from copy import copy

class EMProjectDB:
	"""
	It's implemented as a singleton
	"""
	class __impl:
		""" Implementation of the singleton interface """

		def __init__(self):
			self.projectdb = shelve.open('.eman2projectdb','c',-1,True)

		#def __del__(self):
			#print "closing projectdb"
			#self.projectdb.close()
	# storage for the instance reference
	__instance = None


	def __init__(self):
		""" Create singleton instance """
		# Check whether we already have an instance
		if EMProjectDB.__instance is None:
			# Create and remember instance
			EMProjectDB.__instance = EMProjectDB.__impl()
	
	def __getattr__(self, attr):
		""" Delegate access to implementation """
		return getattr(self.__instance.projectdb, attr)

	def __setattr__(self, attr, value):
		""" Delegate access to implementation """
		return setattr(self.__instance.projectdb, attr, value)
	
	def close(self):
		self.__instance.projectdb.close()



class Box:
	def become(self,trimbox):
		'''
		This is like a copy constructor
		'''
		self.xcorner = trimbox.xcorner			# the xcorner - bottom left
		self.ycorner = trimbox.ycorner			# the ycorner - bottom left
		self.xsize = trimbox.xsize				# the xsize of the box
		self.ysize = trimbox.ysize				# the ysize of the box
		self.isref = trimbox.isref				# a flag that can be used to tell if the box is being used as a reference
		self.changed = trimbox.changed			# a flag signalling the box has changed and display needs updatin
		self.isanchor = trimbox.isanchor		# a flag signalling the box has changed and display needs updatin
		self.TS = trimbox.TS
		self.imagename = trimbox.imagename
		
	def __init__(self,xcorner=-1,ycorner=-1,xsize=-1,ysize=-1,isref=0,correlationscore=0,imagename=None):
		self.xcorner = xcorner			# the xcorner - bottom left
		self.ycorner = ycorner			# the ycorner - bottom left
		self.xsize = xsize				# the xsize of the box
		self.ysize = ysize				# the ysize of the box
		self.isref = isref				# a flag that can be used to tell if the box is being used as a reference
		self.correlationscore = correlationscore	# the correlation score
		
		self.optprofile = None			# a correlation worst-case profile, used for selective auto boxing
		self.changed = False			# a flag signalling the box has changed and display needs updating
		self.corx = -1			# stores the x coordinate of the correlation peak
		self.cory = -1			# stores the y coordinate of the correlation peak
		self.shape = None		# stores the shape used by the image2d widget
		self.image = None 		# stores the image itself, an emdata object
		self.r = 0.4			# RGB red
		self.g = 0.9			# RGB green
		self.b = 0.4			# RGB blue
		self.rorig = 0.4			# RGB red
		self.gorig = 0.9			# RGB green
		self.borig = 0.4			# RGB blue
		self.footprint = None	# stores the image footprint as an emdata object
		self.group = None		# stores a group, typically an int
		self.footprintshrink = 1
		self.isanchor = True		# A flag used by AutoBoxer routines that - if set to true the box will not be included in the generation the template) - This is specific to the SwarmPS autoboxer
		self.boxingobj = None
		self.shape = None
		self.TS = None

		self.flcf = None
		self.imagename = imagename
		
	
	def setImageName(self,imagename):
		self.imagename = imagename
		
	def getImageName(self):
		return self.imagename
	
	def updateBoxImage(self,image,norm=True):
		#print "getting region",self.xcorner,self.ycorner,self.xsize,self.ysize
		self.image = image.get_clip(Region(self.xcorner,self.ycorner,self.xsize,self.ysize))
		if norm:
			self.image.process_inplace("normalize.edgemean")
		
		# make sure there are no out of date footprints hanging around
		self.footprint = None
		
			
	def getBoxImage(self,image=None,norm=True,force=False):
		if self.image == None or force:
			if image == None:
				print 'error, need to specify the image argument when first calling getBoxImage'
			self.updateBoxImage(image,norm)
		return self.image
	
	def getSmallBoxImage(self,flattenradius,shrink):
		'''
		gets a shrunken version of the box by asking the database if the shrunken (entire) image
		exists and then clipping out from it
		'''
		image = self.getSmallImage(flattenradius,shrink)
		if image == None:
			return None
		else:
			return image.get_clip(Region(int(self.xcorner/shrink),int(self.ycorner/shrink),int(self.xsize/shrink),int(self.ysize/shrink)))
		
	def getSmallImage(self,flattenradius,shrink):
		
		doit = True
		try:
			cfimage = EMData(self.imagename+".cf.hdf")
		except:
			doit = False
			
		if doit:
			fr = cfimage.get_attr("flatten_radius")
			shr = cfimage.get_attr("shrink")
			if fr == flattenradius and shr == shrink:
				return cfimage
	
		
		cfimage = CoarsenedFlattenedImage(self.imagename)
		cfimage.updateImage(self.boxingobj.image,flattenradius,shrink)
		return cfimage.getImage()
		
	def getFLCFImage(self,flattenradius,shrink,template):
		cfimage = self.getSmallImage(flattenradius,shrink)
		foundflcf = False
		try:
			flcfimage = EMData(self.imagename+".flcf.hdf")
			foundflcf = True
		except:
			pass
	
		if foundflcf:
			if flcfimage.get_attr("template_time_stamp") == template.getTemplateTS() and flcfimage.get_attr("data_image_time_stamp") == cfimage.get_attr("creation_time_stamp"):
				self.correlation = flcfimage
				return flcfimage
			
		flcfimage = FLCFImage(self.imagename)
		flcfimage.updateImage(cfimage,template)

		return flcfimage.getImage()

	
	def getFootPrint(self,shrink=1):
		if self.footprint == None or shrink != self.footprintshrink:
			self.footprintshrink = shrink
			if self.image == None:
				print "error, you can not make a footprint if there is no image"
				exit(1)
			if shrink == 1:
				self.footprint = self.image.make_footprint()
			else :
				self.footprint = self.image.process("math.meanshrink",{"n":shrink}).make_footprint()
				
		return self.footprint
			
	def updateParams(self,autoBoxer,center=False):
		'''
		Updates internally stored parameters, currently works only for SwarmAutoBoxer, but
		have attempted to lay basic framework if in future we use a different autoBoxer which
		requires its own parameters
		'''
		correlation = self.getFLCFImage(autoBoxer.getTemplateRadius(),autoBoxer.getBestShrink(),autoBoxer.getTemplateObject())
		if correlation == None:
			print 'error, can not update the parameters of a Box because the Boxable has no correlation image'
			return 0
		
		if isinstance(autoBoxer,SwarmAutoBoxer):
			shrink = autoBoxer.getBestShrink()
			invshrink = 1/shrink
	
			# the central coordinates of the box in terms of the shrunken correlation image
			x = (self.xcorner+self.xsize/2.0)*invshrink
			y = (self.ycorner+self.ysize/2.0)*invshrink
			
			#the search radius is used in correlation space - it limits the radial distance
			# up to which 'profile' data can be accrued
			# it is currently half the boxsize in terms of the correlation image's dimensions
			searchradius = autoBoxer.getSearchRadius()
		
			peak_location = BoxingTools.find_radial_max(correlation,int(x),int(y),searchradius )
			peak_location2 = BoxingTools.find_radial_max(correlation,peak_location[0],peak_location[1],searchradius )
			if (peak_location != peak_location2):
				# this represents a troubling condition
				# setting box.correlationscore is the flag that other functions can act on in order to exclude
				# this box from consideration
				self.correlationscore = None
				print "Error, peak location unrefined"
				return 0
		
			# store the peak location
			self.corx = peak_location[0]
			self.cory = peak_location[1]
		
			# store the correlation value at the correlation max
			self.correlationscore = correlation.get(self.corx,self.cory)
		
			# store the profile
			self.optprofile = BoxingTools.get_min_delta_profile(correlation,self.corx,self.cory, searchradius )
			
			# center on the correlation peak
			if (center):
				self.xcorner = self.corx*shrink-self.xsize/2.0
				self.ycorner = self.cory*shrink-self.ysize/2.0
				self.changed = True
			
			return 1
			
		else:
			print 'error, the autoBoxer you are using is not currently known by the Box class'
			return 0
		
class TrimBox():
	'''
	A trimmed down version of a box
	'''
	def __init__(self,box):
		self.xcorner = box.xcorner			# the xcorner - bottom left
		self.ycorner = box.ycorner			# the ycorner - bottom left
		self.xsize = box.xsize				# the xsize of the box
		self.ysize = box.ysize				# the ysize of the box
		self.isref = box.isref				# a flag that can be used to tell if the box is being used as a reference
		self.changed = box.changed			# a flag signalling the box has changed and display needs updatin
		self.isanchor = box.isanchor		# a flag signalling the box has changed and display needs updatin
		self.TS = box.TS					# a time stamp flag
		self.imagename = box.imagename
		
class Reference(Box):
	'''

	A reference is a box, but with extra capabilities. It knows the BoxingObject from which it originated.
	It can therefore tell ...
	AM NOT SURE IF THIS IS NECESSARY YET
	'''
	def __init__(self):
		Box.__init__(self)
		self.boxingobj = None


class FLCFImage:
	def __init__(self,imagename):
		self.templateTS = -1 	# this is the time stamp of the template used to generate this correlation map
		self.imageTS = -1		# this is the time stamp of the image used to generate the correlation map
		self.flcfimage = None	# this is the flcf image
		self.imagename=imagename # we must store this
		
	def paramsMatch(self,cfimage,template):
		if cfimage.getTS() != self.imageTS or template.getTemplateTS() != self.templateTS:
			return False
		else: return True
		
	def getTemplateTS(self):
		return self.templateTS
	def updateImage(self,cfimage,template):
		if not isinstance(cfimage,EMData):
			print "you can't call genFLCF on the cfimage is not an EMData"
			return 0
			
		if not isinstance(template,SwarmTemplate) and not isinstance(template,TrimSwarmTemplate):
			print "you can't call genFLCF on an object that is not a SwarmTemplate"
			return 0
			
		if template.getTemplateTS() != self.templateTS or cfimage.get_attr("creation_time_stamp") != self.imageTS or self.templateTS == -1 or self.imageTS == -1:
			self.templateTS = template.getTemplateTS()
			self.imageTS =cfimage.get_attr("creation_time_stamp")
			smallimage = cfimage
			
			self.flcfimage = smallimage.calc_flcf( template.getTemplate() )
			self.flcfimage.process_inplace("xform.phaseorigin.tocenter")
			self.flcfimage.set_attr("template_time_stamp",template.getTemplateTS())
			self.flcfimage.set_attr("data_image_time_stamp",cfimage.get_attr("creation_time_stamp"))
			self.flcfimage.write_image(self.imagename+".flcf.hdf")

			
			return 1
		
		return 0
		
	def getImage(self):
		return self.flcfimage
	
	
	
class CoarsenedFlattenedImage:
	def __init__(self,imagename):
		self.flattenradius = -1		# the r value used to run the flatten image processor
		self.shrink = -1			# the value used to shrink the image
		self.smallimage = None		# a small copy of an image which has had its background flattened
		self.TS = -1
		self.imagename = imagename
		
	def getTS(self):
		return	self.smallimage.get_attr("creation_time_stamp")
		
	def updateImage(self,image,flattenradius,shrink,forceupdate=False):
		'''
		Updates the image using the function arguments
		If they match current parameters than nothing happens - the correct image is already cached
		'''
		
		if self.smallimage == None or flattenradius != self.flattenradius or self.shrink != shrink or forceupdate:
			
			if (self.shrink != shrink or forceupdate or self.smallimage == None):
				self.shrink = shrink
				if self.shrink != 1:
					self.smallimage = image.process("math.meanshrink",{"n":self.shrink})
				else: self.smallimage = image.copy()
			
				# if we redid the shrink we definitely have to redo the background flattening
				self.flattenradius = flattenradius
				self.smallimage.process_inplace("filter.flattenbackground",{"radius":self.flattenradius})
			
			elif flattenradius != self.flattenradius:
				# if we make it here all that has changed is the flattenradius parameter
				self.flattenradius = flattenradius
				self.smallimage.process_inplace("filter.flattenbackground",{"radius":self.flattenradius})
				
			# store the time stamp for persistence
			self.TS = time()
			
			# now put the results in the database
			
			self.smallimage.set_attr("flatten_radius",self.flattenradius)
			self.smallimage.set_attr("shrink",self.shrink)
			self.smallimage.set_attr("creation_time_stamp",self.TS)
			self.smallimage.write_image(self.imagename+".cf.hdf")
				
		#else:
			#print "doing nothing to currently stored small image in CoarsenedFlattenedImage"
			
	def getImage(self):
		'''
		Should only be called if you know the stored image is up to date
		'''
		return self.smallimage
	
	def reset(self):
		'''
		Sets member variables to their default starting values
		This will force an update next time updateImage is called.
		'''
		self.flattenradius = -1		# the r value used to run the flatten image processor
		self.shrink = -1			# the value used to shrink the image
		self.smallimage = None		# a small copy of an image which has had its background flattened
		
	def paramsMatch(self,flattenradius,shrink):
		if flattenradius != self.flattenradius or self.shrink != shrink:
			return False
		else:
			return True


class Boxable:
	def __init__(self,image,imagename,parent=None,autoBoxer=None):
		self.image = image			# the image containing the boxes
		self.parent = parent		# keep track of the parent in case we ever need it
		self.boxes = []				# a list of boxes
		self.refboxes = []			# a list of boxes
		self.boxsize = -1			#  the boxsize
		self.cfimage = CoarsenedFlattenedImage(imagename)	# a small copy of the image which has had its background flattened
		self.imagename = imagename
		
		self.fpshrink = -1
		self.exclusionimage = None
		self.template = None
		self.correlation = None
		self.refcache = []
		self.allowcorrelationupdate = False	# a temporary flag that can be used by externally objects, for instance a reference box, which is forcing an update for example
		self.templateTS = -1 # a template time stamp, used to avoid unecessarily regenerating the template in self.autoBox
		self.autoBoxerTS = -1 # and autoBoxer time stamp, used to avoid unecessary autoboxing, and to force autoboxing when appropriate
		
		self.autoBoxer = autoBoxer
		
		try: self.boxesReady(True)
		except: pass # this probably means there is a projectdb but it doesn't store any autoboxing results from this image
			
	def getImageName(self):
		return self.imagename
	
	def setAutoBoxer(self,autoBoxer):
		self.autoBoxer = autoBoxer
		
	def getCorrelation(self):
		return self.correlation
	
	def extendBoxes(self,boxes):
		self.boxes.extend(boxes)
		
	def boxesReady(self,forcereadall=False):
		projectdb = EMProjectDB()
		
		trimboxes = projectdb[self.imagename+"_boxes"]
		for trimbox in trimboxes:
			if trimbox.changed or forcereadall:
				box = Box()
				
				# had to do conversion stuff so pickle would work
				box.become(trimbox)
				box.setImageName(self.imagename)
				if forcereadall:
					box.changed = True
				if box.isref and not forcereadall:
					continue;
				elif box.isref:
					box.rorig = 0			# RGB red
					box.gorig = 0			# RGB green
					box.borig = 0			# RGB blue
					box.r = 0
					box.g = 0
					box.b = 0
				
				box.boxingobject=self
				self.boxes.append(box)

	def addbox(self,box):
		if not isinstance(box,Box):
			print "You can not add a box to this box set if it is not of type Box"
			return;
		
		box.isref = True # make sure it knows that it's a reference box
		box.TS = time()
		box.boxingobj = self
		
		box.rorig = 0			# RGB red
		box.gorig = 0			# RGB green
		box.borig = 0			# RGB blue
		box.r = 0
		box.g = 0
		box.b = 0
		
		#print "adding box",box.xcorner,box.ycorner,box.xsize,box.ysize
		self.boxes.append(box)
		self.refboxes.append(box)
	
	def delbox(self,i):
		tmp = self.boxes.pop(i)
		#yuck, this is horribly inefficient
		for j,box in enumerate(self.refboxes):
			if box.isref and box.TS == tmp.TS:
				self.refboxes.pop(j)
				return True
			
		return False
	
	def deletenonrefs(self):
		boxestodelete = []
		n = len(self.boxes)
		for m in range(n-1,-1,-1):
			box = self.boxes[m]
			if box.isref == False:
				self.delbox(m)
				boxestodelete.append(m)

				
		self.parent.deleteDisplayShapes(boxestodelete)
	
	def addnonrefs(self,boxes):
		'''
		Add boxes that are stored in eman1 format
		box[0] = xnorner, box[1] = ycorner, box[2] = xsize, box[3] = ysize
		'''
		for box in boxes:
			b = Box(box[0],box[1],box[2],box[3])
			b.setImageName(self.imagename)
			b.isref = False
			b.changed = True
			self.boxes.append(b)

	def numboxes(self):
		return len(self.boxes)
	
	def updateBoxSize(self,boxsize):
		'''
		Updates only the box size and corner coordinates
		Switches the changed flag to True to trigger redisplay (but the calling function
		is responsible for knowing and testing for this)
		'''
		# do nothing if it's the same size as what we already have
		if  boxsize == self.boxsize: return
		
		for box in self.boxes:
			if box.xsize != boxsize:
				box.xcorner -= (boxsize-box.xsize)/2
				box.xsize = boxsize
				box.changed = True
			if box.ysize != boxsize:
				box.ycorner -= (boxsize-box.ysize)/2
				box.ysize = boxsize
				box.changed = True
			
			box.image = None
			box.footprint = None

		self.fprink = -1
		self.flattenimager = -1
		self.boxsize = boxsize
		self.cfimage.reset()
		self.correlation = None
		
	def getfootprintshrink(self):
		if self.fpshrink == -1:
			shrink = 1
			tn = self.boxsize/2
			while ( tn >= 32 ):
				tn /= 2
				shrink *= 2
			self.fpshrink = shrink
		
		return self.fpshrink
		
	def getBestShrink(self):
		'''
		FIXME - there should probably be a more well established framework for doing this
		At the moment it is possible that the self.autBoxer is actually None, which isn't good.	
		'''
		if self.autoBoxer != None:
			return self.autoBoxer.getBestShrink()
		else:
			print 'warning, there is not autoboxer set, am not sure how to shrink, returning 1 as the shrink factor'
			return 1
		
	def updateCorrelation(self,template):
		'''
		A function that will update the correlation image if the correlationupdate flag is set to true
		Useful if a template has been updated somewhere, yet many references originate from this BoxingOject -
		All the references will call this function, but we only need to act the first time it happens
		
		Be warned - whoever opens the gate by setting self.allowcorrelationupdate to True should set it 
		to False once their done with the specialized operation
		
		'''
		if self.allowcorrelationupdate:
			self.templateTS = template.getTemplateTS() # Time Stamp, used for efficiency in autoBox to save an unecessary correlation update
			self.__genCorrelation(template)
			
			# I made a conscientious decision to leave the responsibility of turning this flag off
			# to that of the calling program/function. This uncommented line is left only for documentation purposes
			#self.allowcorrelationupdate = False


	def __genCorrelation(self,template):
		'''
		The force update flag is only meant to be used if the box size has changed - this changes 
		the shrink factor, and also affects the background flattening process.
		'''
		cfimage = self.getSmallImage()
		
		foundflcf = False
		try:
			flcfimage = EMData(self.imagename+".flcf.hdf")
			foundflcf = True
		except:
			pass
	
		if foundflcf:
			if flcfimage.get_attr("template_time_stamp") == template.getTemplateTS() and flcfimage.get_attr("data_image_time_stamp") == cfimage.get_attr("creation_time_stamp"):
				self.correlation = flcfimage
				return flcfimage
			#else: 
				#print flcfimage.get_attr("template_time_stamp"),template.getTemplateTS(),flcfimage.get_attr("data_image_time_stamp"),cfimage.get_attr("creation_time_stamp")
				#print 'bummer'
			
		flcfimage = FLCFImage(self.imagename)
		flcfimage.updateImage(cfimage,template)
		self.correlation = flcfimage.getImage()
		self.templateTS = flcfimage.getImage().get_attr("template_time_stamp")
		
		return flcfimage.getImage()
	
	def getCorrelationImage(self):
		return self.correlation
	
	
	def getSmallImage(self):
		
		doit = True
		try:
			cfimage = EMData(self.imagename+".cf.hdf")
		except:
			doit = False
			
		if doit:
			flatten_radius = cfimage.get_attr("flatten_radius")
			shrink = cfimage.get_attr("shrink")
			if flatten_radius == self.autoBoxer.getTemplateRadius() and shrink == self.autoBoxer.getBestShrink():
				return cfimage

		self.cfimage.updateImage(self.image,self.autoBoxer.getTemplateRadius(),self.autoBoxer.getBestShrink())
		return self.cfimage.getImage()
	
	def updateExcludedBoxes(self):
		'''
		
		'''
		lostboxes = []
		
		invshrink = 1.0/self.getBestShrink()
		exc = self.getExclusionImage()
		n = len(self.boxes)
		for i in range(n-1,-1,-1):
			box = self.boxes[i]
			x = int((box.xcorner+box.xsize/2.0)*invshrink)
			y = int((box.ycorner+box.ysize/2.0)*invshrink)
			
			if ( exc.get(x,y) != 0):
				lostboxes.append(i)
			
				self.boxes.pop(i)
	
		return lostboxes
	
	def addExclusionArea(self, type,x,y,radius):
		
		xx = int(x/self.getBestShrink())
		yy = int(y/self.getBestShrink())
		
		rr = int(radius/self.getBestShrink())
		rrs = rr**2
		#print xx,yy,rr
		
		# this does implicit initialization
		self.getExclusionImage()
		
		ny = self.getSmallImage().get_ysize()
		nx = self.getSmallImage().get_xsize()
		for j in range(-rr,rr):
			for i in range(-rr,rr):
				if (i**2 + j**2) > rrs: continue
				jj = j+yy
				ii = i+xx
				if jj >= ny or jj < 0:continue
				if ii >= nx or ii < 0:continue
				
				self.exclusionimage.set(ii,jj,0.1)
				
		# FIXME - is this the best place for this?
		self.parent.guiim.setOtherData(self.getExclusionImage(),self.getBestShrink(),True)
	
	def getExclusionImage(self,force=True):
		if self.exclusionimage == None and force:
				
			self.exclusionimage = EMData(self.getSmallImage().get_xsize(),self.getSmallImage().get_ysize())
			self.exclusionimage.to_zero()
		
		return self.exclusionimage
	
	#def classify(self):
		#v = []
		## accrue all params
		#for box in self.boxes:
		#self.accrueparams(self.boxes)
		
		#for box in self.boxes:
			#b = copy(box.optprofile[0:self.radius])
			#b.sort()
			##for a in b:
				##a = box[6]-a
			##print b
			#v.append(b)
			
		#cl = BoxingTools.classify(v,4)
		#self.parent.updateboxcolors(cl)
	
	def genRefImages(self):
		tmpimage = "tmpparticles.img"
		self.parent.writeBoxesTo(tmpimage)
		
		self.process = QtCore.QProcess()

		program = QtCore.QString("e2refine2d.py")
		args = QtCore.QStringList()
		args.append("--input="+tmpimage)
		args.append("--ncls=25")
		
		QtCore.QObject.connect(self.process, QtCore.SIGNAL("finished(int)"), self.process_finished)
		QtCore.QObject.connect(self.process, QtCore.SIGNAL("started()"), self.process_start)
		print self.process.start(program,args)

	def process_start(self):
		print "received process start signal"
		
	def boxsel(self,event,lc):
		#print "selected",lc[0]
		for box in self.boxes:
			if box.group == lc[0]:
				box.r = 1
				box.g = 1
				box.b = 1
				box.changed = True
			elif box.r == 1 and box.g == 1 and box.b == 1:
				box.r = box.rorig
				box.g = box.gorig
				box.b = box.borig
				box.changed = True
		self.imagemx2.setSelected(lc[0])
		self.parent.boxDisplayUpdate()
	def process_finished(self,int):
		try:
			from emimage import EMImage
		except:
			print "Cannot import EMAN image GUI objects (emimage,etc.)"
			sys.exit(1)
		
		e = EMData().read_images("classes.init.hdf")
		self.imagemx2p = EMImage(e)
		self.imagemx2 = self.imagemx2p.child
		self.imagemx2.setmmode("app")
		QtCore.QObject.connect(self.imagemx2,QtCore.SIGNAL("mousedown"),self.boxsel)
		self.imagemx2p.show()
		
		ef = []
		for image in e:
			image.process_inplace("normalize.edgemean")
			if self.getBestShrink() != 1:
				image = image.process("math.meanshrink",{"n":self.getfootprintshrink()})	
			ef.append(image.make_footprint())
		
		for box in self.boxes:
			best = -1
			group = -1
			for i,g in enumerate(ef): 
				s = box.getFootPrint(self.getfootprintshrink()).cmp("optvariance",g,{"matchfilt":1,"matchamp":1})
				# REMEMBER - cmp returns values that have potentially been negated - a smaller value is better
				if best == -1 or s < best:
					group = i
					best = s
			
			box.group = group
					
		
		#print scores
		
		print "received finish signal"

class AutoBoxer:
	'''
	Base class design for auto boxers
	'''
	def __init__(self,boxable=None):
		self.version = 1.0

	def getTemplate(self):
		'''This should return a single template which is an EMData object'''
		raise Exception
	
	def name(self):
		'''
		Every autoboxer should return a unique name
		'''
		raise Exception
	
	def addReference(self,box):
		'''
		add a reference box - the box should be in the format of a Box object, see above
		Returns 0 if there is a problem, returns 1 if it's all good
		Adds a reference to a list
		'''
		raise Exception
	
	def removeReference(self,box):
		'''
		Remove a reference box - the box should in the format of a Box object, see above
		Pops a reference from a list
		'''
		raise Exception
	
	def referenceMoved(self,ref):
		'''
		If a reference was moved interactively in the interface this is the function that should be called
		'''
		raise Exception

	def getTemplate(self):
		'''
		Return the template that is being used. Returns None if there is not template
		'''
		raise Exception

	def setBoxSize(self,boxsize):
		'''
		Hard set the boxsize. Note that nothing is done to the reference boxes. It is
		assumed whichever part of the program calls this function also updates the Box objects
		independently (which implicitly affects the boxes stored internally in the AutoBoxer
		class, because it only ever stores programmatic references)
		'''
		raise Exception
	
	def autoBox(self,correlation,boxes=[],exclusion=None):
		'''
		The main autoBox routine. The calling program should pass in its own correlation map (EMData), and optionally
		an exclusion map of ones and zeros (0 means include, non zero means exclude). Also a list of boxes that will
		not be removed prior to the autoBoxing (and hence probably have associated excluded regions in the exlcusion
		image, but this is beside the point), The model of use here is that
		the calling program should get the current template from the AutoBoxer to generate the correlation map. The
		calling program should be able to cache the correlation map, so should be able to detect if there's been
		a template update by asking for the current set of references (getReferences) and cross checking against a list of its own.
		@Returns a list of Boxes
		'''
		raise Exception

class TrimSwarmTemplate:
	'''
	used from writing a template to the database
	'''
	def __init__(self,swarmTemplate):
		self.refboxes = []		# this will eventually be a list of Box objects
		for ref in swarmTemplate.refboxes:
			self.refboxes.append(TrimBox(ref))
		self.template = swarmTemplate.template	# an EMData object that is the template
		self.templateTS = swarmTemplate.templateTS 	# a time stamp that records when the template was generate

class SwarmTemplate:
	def __init__(self,autoBoxer):
		self.refboxes = []		# this will eventually be a list of Box objects
		self.template = None	# an EMData object that is the template
		self.templateTS = -1 	# a time stamp that records when the template was generate
		self.autoBoxer = autoBoxer
	
	def become(self,trimSwarmTemplate):
		if not isinstance(trimSwarmTemplate,TrimSwarmTemplate):
			print "error, can't become anything other than a TrimSwarmTemplate"
			return 0
		else:
			
			self.template = trimSwarmTemplate.template	# an EMData object that is the template
			self.templateTS = trimSwarmTemplate.templateTS 	# a time stamp that records when the template was generate
			self.refboxes = []		# this will eventually be a list of Box objects
			for ref in trimSwarmTemplate.refboxes:
				b = Box()
				b.become(ref)
				self.refboxes.append(b)
			
	def getTemplate(self):
		return self.template
	
	def getTemplateTS(self):
		return self.template.get_attr("template_time_stamp")
	
	def getReferences(self):
		return self.refboxes
	
	def appendReference(self,ref):
		if isinstance(ref,Box):
			self.refboxes.append(ref)
		else:
			print "error, can't append that reference, it's not of type Box"
		
	def removeReference(self,box):
		'''
		Returns 1 if the reference was removed
		Returns 0 if it wasn't found
		'''
		if not isinstance(box,Box):
			print "error, can't remove a reference that isn't a box"
			return 0
		
		for j,tmp in enumerate(self.refboxes):
			if box.isref and box.TS == tmp.TS:
				tmp = self.refboxes.pop(j)
				return 1	
				
		return 0
	
	def genTemplate(self):
		'''
		Returns 0 if there are errors
		Return 1 if not
		'''
		# you can only generate a template if there are references
		if len(self.refboxes) <= 0: 
			print 'error, cant call private function genTemplate when there are no refboxes, this is an internal error'
			return 0
		
		images_copy = []
		for ref in self.refboxes:
			# some references can be excluded from the template generation procedure, this is flagged
			# by the isanchor flag
			if ref.isanchor == False:
				continue
			image = ref.getSmallBoxImage(self.autoBoxer.getTemplateRadius(),self.autoBoxer.getBestShrink())
			images_copy.append(image)
		if len(images_copy) == 0:
			print 'error, you have probably set references that all have the isanchor flag set to false, which exluded them all from the template making process'
			print 'can not proceed without references to create template'
			return 0
			
		ave = images_copy[0].copy()
		
		for i in range(1,len(images_copy)):
			#ta = images_copy[i].align("rotate_translate",ave,{},"dot",{"normalize":1})
			ave.add(images_copy[i])
		
		#ave.write_image("prealigned.hdf")
		ave.mult(1.0/len(images_copy))
		ave.process_inplace("math.radialaverage")
		ave.process_inplace("xform.centeracf")
		ave.process_inplace("mask.sharp",{'outer_radius':ave.get_xsize()/2})
		
		#for image in images_copy:
		#	image.write_image("aligned_refs.img",-1)
		#
		#ave.write_image("aligned_refs.img",-1)
		#
		#black = EMData(image.get_xsize(),image.get_ysize())
		#black.to_zero()
		#black.write_image("aligned_refs.img",-1)
		
		#ave.write_image("ave.hdf")
		shrink = self.autoBoxer.getBestShrink()
		# 4 is a magic number
		for n in range(0,4):
			t = []
			for idx,i in enumerate(images_copy):
				ta = i.align("translational",ave,{},"dot",{"normalize":1})
				#t.append(ta)
				
				# FIXME - make it so that a newly clipped portion of the original image
				# is used as the 'aligned' image, to avoid zeroing effects at the edges
				# The problem with this approach is one of persistence. If the box has no associated
				# boxingobj then it will fail. The box will have no boxing obj using the persistent database 
				# approach of e2boxer
				dx = ta.get_attr("align.dx")
				dy = ta.get_attr("align.dy")
				box = self.refboxes[idx]
				size = ta.get_xsize()
				image = box.getSmallImage(self.autoBoxer.getTemplateRadius(),self.autoBoxer.getBestShrink())
				a = image.get_clip(Region(int(box.xcorner/shrink-dx),int(box.ycorner/shrink-dy),size,size))
				a.process_inplace("normalize.edgemean")
				
				t.append(a)
				
		
			ave = t[0].copy()
			for i in range(1,len(images_copy)):
				ave.add(t[i])
				
			ave.mult(1.0/len(t))
			ave.process_inplace("math.radialaverage")
			ave.process_inplace("xform.centeracf")
			# edge normalize here SL before
			ave.process_inplace("mask.sharp",{'outer_radius':ave.get_xsize()/2})
			# or normalize and no mask
		
		#debug, un-comment to see the aligned refs and the final template
		#for image in t:
		#	image.write_image("aligned_refs.img",-1)
		
		#ave.write_image("aligned_refs.img",-1)
		
		#black = EMData(image.get_xsize(),image.get_ysize())
		#black.to_zero()
		#black.write_image("aligned_refs.img",-1)
		#END uncomment block
		self.template = ave
		
		self.templateTS = time()
		self.template.set_attr("template_time_stamp",self.templateTS)
		return 1
	
	

class TrimSwarmAutoBoxer():
	def __init__(self,swarmAutoBoxer):
			
		self.boxsize = swarmAutoBoxer.boxsize
		self.shrink = swarmAutoBoxer.shrink
		self.templatedimmin = swarmAutoBoxer.templatedimmin
		
		self.optthreshold = swarmAutoBoxer.optthreshold
		self.optprofile = copy(swarmAutoBoxer.optprofile)
		self.optprofileradius = swarmAutoBoxer.optprofileradius
		self.autoboxmethod = swarmAutoBoxer.autoboxmethod
		self.templateTS = swarmAutoBoxer.templateTS
		self.stateTS = swarmAutoBoxer.stateTS
		self.mode = swarmAutoBoxer.mode
		self.refupdate = swarmAutoBoxer.refupdate
		self.regressiveflag = swarmAutoBoxer.regressiveflag
		
		self.template = TrimSwarmTemplate(swarmAutoBoxer.template)
		
	
class SwarmAutoBoxer(AutoBoxer):
	'''
	This is an autoboxer that encapsulates the boxing approach first developed in SwarmPS
	'''
	THRESHOLD = "Threshold"
	SELECTIVE = "Selective"
	MORESELECTIVE = "More Selective"
	DYNAPIX = 1
	ANCHOREDDYNAPIX = 2
	USERDRIVEN = 3
	ANCHOREDUSERDRIVEN = 4
	COMMANDLINE = 5
	def __init__(self,boxable,parent=None):
		AutoBoxer.__init__(self,boxable)
		
		self.boxable = boxable
		
		self.refboxes = []		# this will eventually be a list of Box objects
		self.template = SwarmTemplate(self)	# an EMData object that is the template
		self.boxsize = -1		# stores the global boxsize, this is the value being used by boxer in the main interface
		self.shrink = -1
		
		# more privately stuff
		self.templatedimmin = 20  # the smallest amount the template can be shrunken to. Will attempt to get as close to as possible. This is an important part of speeding things up.
		self.optthreshold = -1	# the correlation threshold, used to as the basis of finding local maxima
		self.optprofile = []	# the optimum correlation profile used as the basis of auto selection
		self.optprofileradius = -1 # the optimum radius - used to choose which part of the optprofile is used as the basis of selection
		self.autoboxmethod = SwarmAutoBoxer.SELECTIVE	# the autobox method - see EMData::BoxingTools for more details
		self.__shrink = -1
		
		self.templateTS = -1 # a template time stamp to 
		self.stateTS = -1 # the time stamp that records when the current template and parameters are completely up to date
		
		
		self.mode = SwarmAutoBoxer.DYNAPIX
		self.refupdate = False # this is a flag used when self.mode is USERDRIVEN
		self.permissablemodes = [SwarmAutoBoxer.DYNAPIX,SwarmAutoBoxer.ANCHOREDDYNAPIX,SwarmAutoBoxer.USERDRIVEN,SwarmAutoBoxer.ANCHOREDUSERDRIVEN,SwarmAutoBoxer.COMMANDLINE]  # if another mode is added you would have to find all places where self.mode is used to make decisions and alter
		self.regressiveflag = False	# flags a force removal of non references in the Boxable in autoBox
		
		
		self.parent = parent
	def become(self,trimSwarmAutoBoxer):			
		self.boxsize = trimSwarmAutoBoxer.boxsize
		self.shrink = trimSwarmAutoBoxer.shrink
		self.templatedimmin = trimSwarmAutoBoxer.templatedimmin
		
		self.optthreshold = trimSwarmAutoBoxer.optthreshold
		self.optprofile = copy(trimSwarmAutoBoxer.optprofile)
		self.optprofileradius = trimSwarmAutoBoxer.optprofileradius
		self.autoboxmethod = trimSwarmAutoBoxer.autoboxmethod
		self.templateTS = trimSwarmAutoBoxer.templateTS
		self.stateTS = trimSwarmAutoBoxer.stateTS
		self.mode = trimSwarmAutoBoxer.mode
		self.refupdate = trimSwarmAutoBoxer.refupdate
		self.regressiveflag = trimSwarmAutoBoxer.regressiveflag
		self.template = SwarmTemplate(self)
		self.template.become(trimSwarmAutoBoxer.template)
		
	def getTemplate(self):
		return self.template
	
	def setBoxable(self,boxable):
		self.boxable = boxable
	
	def setModeExplicit(self,mode):
		if mode in self.permissablemodes:
			self.mode = mode
		else:
			print "error, that mode:", mode, "was not in the list of permissable modes"
			exit(1)
	def setMode(self,dynapix,anchortemplate):
		if dynapix:
			if anchortemplate: self.mode = SwarmAutoBoxer.ANCHOREDDYNAPIX
			else: self.mode = SwarmAutoBoxer.DYNAPIX
		else:
			if anchortemplate: self.mode = SwarmAutoBoxer.ANCHOREDUSERDRIVEN
			else: self.mode = SwarmAutoBoxer.USERDRIVEN

	def name(self):
		return 'swarmautoboxer'

	def addReference(self,box):
		'''
		 add a reference box - the box should be in the format of a Box, see above):
		'''
		if isinstance(box,Box):
			if box.xsize != box.ysize:
				print 'error, support for uneven box dimensions is not currently implemented'
				return 0
		
			# store the boxsize if we don't have one already
			if self.boxsize == -1:
				self.boxsize = box.xsize
			# do a sanity check, this shouldn't happen if the program is managing everything carefully
			elif self.boxsize != box.xsize:
				print 'error, the currently stored box size does not match the boxsize of the reference that was just added'
				return 0
			
			self.template.appendReference(box)
		
			if self.mode == SwarmAutoBoxer.DYNAPIX:
				if not box.isanchor:
					print 'the box flag is internally inconsistent when using pure dynapix'
					return 0
				self.__fullUpdate()
				self.autoBox(self.boxable)
			elif self.mode == SwarmAutoBoxer.ANCHOREDDYNAPIX:
				if box.isanchor:
					print 'the box flag is internally inconsistent when anchoring'
					return 0
				box.updateParams(self)
				self.__accrueOptParams()
				self.stateTS = time()
				self.autoBox(self.boxable)
			elif self.mode == SwarmAutoBoxer.USERDRIVEN:
				self.refupdate = True
				self.stateTS = -1
				self.templateTS = -1
			elif self.mode == SwarmAutoBoxer.ANCHOREDUSERDRIVEN:
				box.updateParams(self)
				self.__accrueOptParams()
				self.stateTS = time()
			else:
				print 'error, unknown mode in SwarmAutoBoxer'
				return 0
		
			return 1
	
			
		else:
			print "error, you cannot add a reference to the AutoBoxer if it is not in the format of a Box object"
			return 0
	
	def removeReference(self,box):
		if self.template.removeReference(box):
			if len(self.template.refboxes) == 0:
				self.__reset()
				return 1
			
			if self.mode == SwarmAutoBoxer.DYNAPIX or self.mode == SwarmAutoBoxer.ANCHOREDDYNAPIX:
				if box.isanchor:
					self.__fullUpdate()
					self.regressiveflag = True
					self.autoBox(self.boxable)
				else:
					self.__accrueOptParams()
					self.stateTS = time()
					self.regressiveflag = True
					self.autoBox(self.boxable)
				return 1
			elif self.mode == SwarmAutoBoxer.USERDRIVEN or self.mode == SwarmAutoBoxer.ANCHOREDUSERDRIVEN:
				if box.isanchor:
					self.refupdate = True
					self.stateTS = -1
					self.templateTS = -1
				else:
					box.updateParams(self)
					self.__accrueOptParams()
					self.stateTS = time()
					self.regressiveflag = True
					
				return 1
		return 0
	
	def getTemplateRadius(self):
		'''
		Returns what will be or is the template radius
		'''
		return int(self.boxsize/2/self.getBestShrink())
	
	def referenceMoved(self,box):
		'''
		If a reference was moved interactively in the interface this is the function that should be called
		'''
		if self.mode == SwarmAutoBoxer.DYNAPIX or self.mode == SwarmAutoBoxer.ANCHOREDDYNAPIX:
			if box.isanchor:
				self.__fullUpdate()
				self.regressiveflag = True
				self.autoBox(self.boxable)
			else:
				box.updateParams(self)
				self.__accrueOptParams()
				self.stateTS = time()
				self.regressiveflag = True
				self.autoBox(self.boxable)
			return 1
		elif self.mode == SwarmAutoBoxer.USERDRIVEN or self.mode == SwarmAutoBoxer.ANCHOREDUSERDRIVEN:
			if box.isanchor:
				self.refupdate = True
				self.stateTS = -1
				self.templateTS = -1
			else:
				box.updateParams(self)
				self.__accrueOptParams()
				self.stateTS = time()
				self.regressiveflag = True
			return 1
		else:
			print 'error, unknown mode in SwarmAutoBoxer'
			return 0
		
	def getTemplateObject(self):
		return self.template
		
	def getTemplate(self):
		if self.refupdate:
			self.__fullUpdate()
			self.refupdate = False
			
		if self.template == None:
			print 'error, you have either asked for the template without setting a reference, or you have added a reference and not set the refupdate flag'
			return None
		
		return self.template
		
	def setBoxSize(self,boxsize):
		if (boxsize < 6 ):
			print 'error, a hard limit of 6 for the box size is currently enforced. Email developers if this is a problem'
			return
		if self.boxsize == boxsize:	return
		
		self.boxsize = boxsize
		# make sure the shrink value is updated - use the force flag to do it
		self.getBestShrink(True)
		
		if self.mode == SwarmAutoBoxer.DYNAPIX or self.mode == SwarmAutoBoxer.ANCHOREDDYNAPIX:
			# update references
			self.__fullUpdate()
			self.autoBox(self.boxable)
		elif self.mode == SwarmAutoBoxer.USERDRIVEN or self.mode == SwarmAutoBoxer.ANCHOREDUSERDRIVEN :
			self.refupdate = True
			self.stateTS = -1
			self.templateTS = -1
		else:
			print 'error, unknown mode in SwarmAutoBoxer'
	
	def getSearchRadius(self):
		return int(0.75*(self.boxsize)/float(self.getBestShrink()))
	
	
	def getBestShrink(self,force=True):	
		if self.boxsize == -1:	
			print "error - the boxsize is currently -1 - I can't figure out the best value to shrink by"	
			return -1
			
		if self.shrink == -1 or force:	
			self.shrink = ceil(float(self.boxsize)/float(self.templatedimmin))	
			
		return self.shrink
		
	def autoBox(self,boxable):
		# If it's user driven then the user has selected a bunch of references and then hit 'autobox'.
		# In which case we do a complete reference update, which generates the template and the
		# best autoboxing parameters
		if len(self.getRefBoxes()) == 0:
			print 'error, cant get template if there are no references'
			return 0

		# ref update should only be toggled if we are in user driven mode
		if self.refupdate:
			self.__fullUpdate()
			self.refupdate = False

		templateTS = boxable.templateTS
		correlation = boxable.getCorrelationImage()
		
		if templateTS == -1 or correlation == None or self.template.getTemplateTS() != templateTS:
			if self.template != None:
				boxable.allowcorrelationupdate = True
				boxable.updateCorrelation(self.template)
				boxable.allowcorrelationupdate = False
				
				correlation = boxable.correlation
			else:
				print 'error, cant ask the autoBoxer for its template, it doesnt have one'
				return 0

		autoBoxerTS = boxable.autoBoxerTS
		# auto boxing will only ever occur if the time stamp of the AutoBoxer is not the
		# same as the time stamp cached by the Boxable. -1 means it's the first time.
		if autoBoxerTS == -1 or autoBoxerTS != self.stateTS:
			
			if self.mode == SwarmAutoBoxer.DYNAPIX or self.mode == SwarmAutoBoxer.USERDRIVEN or self.regressiveflag:
				# we must clear all non-refs if we're using dynapix
				boxable.deletenonrefs()
				self.regressiveflag = False
			
			exclusion = boxable.getExclusionImage().copy()
			self.__paintExcludedBoxAreas(exclusion,boxable.boxes)
			exclusion.write_image("exclusion.hdf")
		
			boxes = self.__autoBox(correlation,boxable,boxable.boxes,exclusion)
			print "autoboxed",len(boxes)
			boxable.autoBoxerTS = self.stateTS

			# This shouldn't happen in the Database instance
			if boxes != 0:
				projectdb = EMProjectDB()
				boxes.extend(boxable.boxes)
				trimboxes = []
				for  box in boxes: trimboxes.append(TrimBox(box))
				
				projectdb[boxable.getImageName()+"_boxes"] = trimboxes
				trimSelf = TrimSwarmAutoBoxer(self)

				projectdb[boxable.getImageName()+"_autoboxer"] = trimSelf
				projectdb["currentautoboxer"] = trimSelf
				
				boxable.boxesReady()
				#else: 
					#boxable.extendboxes(boxes)
				
			return 1

		else: print 'no auto boxing was necessary, up-2-date' # DEBUG
		
	def __reset(self):
		self.boxsize = -1
		self.stateTS = -1
		self.templateTS = -1

	def __autoBox(self,correlation,boxable,boxes=[],exclusion=None):
		'''
		Does the autoboxing. Returns a list of Boxes
		'''
		if not isinstance(correlation,EMData):
			print 'error, cannot autobox, the correlation argument is not an EMData object'
			return 0
			
			#print "using opt radius",self.radius, "which has value",tmp,"shrink was",self.shrink
		if self.autoboxmethod == SwarmAutoBoxer.THRESHOLD:
			mode = 0
		elif self.autoboxmethod == SwarmAutoBoxer.SELECTIVE:
			mode = 1
		elif self.autoboxmethod == SwarmAutoBoxer.MORESELECTIVE:
			mode = 2
		
		shrink = self.getBestShrink()
		# Warning, this search radius value should be the same as the one used by the BoxSets that contributed the reference boxes
		# to this AutoBoxer object. There should be one place/function in the code where both parties access this value
		searchradius = self.getSearchRadius()
		soln = BoxingTools.auto_correlation_pick(correlation,self.optthreshold,searchradius,self.optprofile,exclusion,self.optprofileradius,mode)

		# This is what should be written to the database
		boxes = []
		
		for b in soln:
			x = b[0]
			y = b[1]
			xx = int(x*shrink)
			yy = int(y*shrink)
			box = Box(xx-self.boxsize/2,yy-self.boxsize/2,self.boxsize,self.boxsize,0)
			box.setImageName(boxable.getImageName())
			box.correlationscore = correlation.get(x,y)
			box.corx = b[0]
			box.cory = b[1]
			box.changed = True
			boxes.append(box)
	
		return boxes
		
	
	
	def __fullUpdate(self):
		'''
		Forces a template update, then updates all correlation images
		that the references come from, then does a parameters update.
		This is like a completeness function - this needs to happen for internal
		consistency. It is motivated by the thought that if references
		come from many images, then the correlation images of each of the images
		needs to be updated in order for the correlation parameters to be generated
		consistently (from a correlation image generated by a universal template)
		'''
		
		if not self.template.genTemplate():
			print 'error, couldnt generate template'
			return 0
		
		# First tell all references' associated boxing objects to be open to the prospect 
		# if update their correlation images
		
		for ref in self.getRefBoxes():
			ref.updateParams(self)
	
		# parameters should be updated now
		# it's important that the BoxingObjext.updateCorrelation updated the parameters stored in the boxes
		self.__accrueOptParams()
		
		self.stateTS = time()

	
	def getRefBoxes(self):
		return self.template.getReferences()
	
	def __accrueOptParams(self):
		'''
		A function for accruing the parameters of the SwarmPSAutoBoxer autoboxing technique
		returns True if optimal parameters were accrued
		return False if not
		'''

		# To determine the threshold from what we've got, iterate through all of the reference
		# boxes and use the lowest correlation score as the correlation threshold
		#print 'current params are, using a total of',len(self.refboxes),'references'
		#print 'threshod:',self.optthreshold
		#print 'profile:',self.optprofile
		#print 'optrad:',self.optprofileradius
		
		found = False
		for i,box in enumerate(self.getRefBoxes()):
			if box.correlationscore == None:
				# this is an error which probably means that the box, as created by the user, has a strong correlation maximum next to it which is disrupting the auto parameters
				# this is mostly an error for dwoolfords attention
				# for the time being just ignoring it  probably suffices
				# FIXME
				print "continuing on faulty"
				continue
			if found == False:
				self.optthreshold = box.correlationscore
				found = True
			else:	
				if box.correlationscore < self.optthreshold: self.optthreshold = box.correlationscore

		# catch the circumstance where for some strange reason things just didn't work
		# probably the user has some strange data and the rotational template isn't responding normally. 
		# correlation peaks aren't where the user thinks they are.
		if not found:
			print 'error, there were no parameter data that I could inspect. I cant make the optimal parameters'
			return False
		
		# Iterate through the reference boxes and accrue what you can think of
		# as the worst case scenario, in terms of correlation profiles
		found = False
		for i,box in enumerate(self.getRefBoxes()):
			if box.correlationscore == None:
				##print "continuing on faulty" - this was already printed above
				continue
			if found == False:
				self.optprofile = copy(box.optprofile)
				n = len(self.optprofile)
				found = True
			else:
				profile = box.optprofile
				for j in range(0,n):
					if profile[j] < self.optprofile[j]: self.optprofile[j] = profile[j]
		
	
		# determine the point in the profile where the drop in correlation score is the greatest, store it in radius
		self.optprofileradius = -1
		tmp = self.optprofile[0]
		for i in range(1,len(self.optprofile)):
			# the tmp > 0 is a
			if self.optprofile[i] > tmp and tmp > 0:
				tmp = self.optprofile[i]
				self.optprofileradius = i
		
		try:
			self.parent.optparamsupdate(self.optthreshold,self.optprofile,self.optprofileradius)
		except: pass
		#print 'NOW THEY ARE'
		#print 'threshod:',self.optthreshold
		#print 'profile:',self.optprofile
		#print 'optrad:',self.optprofileradius
		return True
	
	def __paintExcludedBoxAreas(self,exclusionimage,boxes):
	
		searchradius = self.getSearchRadius()

		for box in boxes:
			# xx and yy are the centers of the image, but in real image coordinates
			xx = box.xcorner + box.xsize/2
			yy = box.ycorner + box.ysize/2
			# shrink them to the small correlation image coordinates
			xx /= self.getBestShrink()
			yy /= self.getBestShrink()
			# Set a positive circle into the exclusionimage
			BoxingTools.set_radial_non_zero(exclusionimage,int(xx),int(yy),searchradius)
			
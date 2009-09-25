#!/usr/bin/env python

#
# Author: David Woolford 11/25/08 (woolford@bcm.edu
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

from emapplication import EMStandAloneApplication
from emimage3dsym import EM3DSymViewerModule,EMSymInspector
from emglobjects import EMImage3DGUIModule
from PyQt4 import QtGui,QtCore
from PyQt4.QtCore import Qt
from OpenGL import GL,GLU,GLUT
from OpenGL.GL import *
from OpenGL.GLU import *
import weakref
from optparse import OptionParser
from EMAN2 import *
from emapplication import get_application
from emimagemx import EMImageMXModule
import os
import sys
from libpyGLUtils2 import GLUtil
from emsprworkflow import error
from emanimationutil import OrientationListAnimation,Animator
from emimagemx import EMLightWeightParticleCache
from time import time

def get_eulers_from(filename):
	eulers = []
	n = EMUtil.get_image_count(filename)
	for i in range(n):
		h = get_header(filename,i)
		try: p = h["xform.projection"]
		except:
			print "image",i,"doesn't have the xform.projection attribute"
			return None
		
		eulers.append(p)
		
	return eulers

def main():
	progname = os.path.basename(sys.argv[0])
	usage = """%prog 
	
Asymmetric unit viewer for EMAN2. Allows the user to inspect the results of refinement in terms of the asymmetric unit.

Works if launched in a workflow directory.

"""

	parser = OptionParser(usage=usage,version=EMANVERSION)

	(options, args) = parser.parse_args()
	
	
	logid=E2init(sys.argv)
	
	em_app = EMStandAloneApplication()
	window = EMEulerExplorer(application=em_app)

	em_app.show()
	em_app.execute()
	
	E2end(logid)

class InputEventsHandler:
	'''
	Perhaps the final installation in terms of what I think is the best design for the mouse events
	handling class. Others exist in emimage2d, emimagemx, and e2boxer. Eventually they should all use the same approach, and 
	I vote for using this one
	'''
	def __init__(self,parent=None):
		if parent != None:
			self.parent = weakref.ref(parent)
		else:
			self.parent = None
		
	def mousePressEvent(self,event):
		pass
	
	def mouseReleaseEvent(self,event):
		pass
	
	def mouseMoveEvent(self,event):
		pass
	
	def mouseDoubleClickEvent(self,event):
		pass
	
	def keyPressEvent(self,event):
		pass

	def wheelEvent(self,event):
		pass
	
class InputEventsManager(InputEventsHandler):
	def __init__(self):
		InputEventsHandler.__init__(self)
		self.current_events_handler = None
		
	def mousePressEvent(self,event):
		if self.current_events_handler != None:
			self.current_events_handler.mousePressEvent(event)
	
	def mouseReleaseEvent(self,event):
		if self.current_events_handler != None:
			self.current_events_handler.mouseReleaseEvent(event)
	
	def mouseMoveEvent(self,event):
		if self.current_events_handler != None:
			self.current_events_handler.mouseMoveEvent(event)
	
	def mouseDoubleClickEvent(self,event):
		if self.current_events_handler != None:
			self.current_events_handler.mouseDoubleClickEvent(event)
	
	def keyPressEvent(self,event):
		if self.current_events_handler != None:
			self.current_events_handler.keyPressEvent(event)
	
	def wheelEvent(self,event):
		if self.current_events_handler != None:
			self.current_events_handler.wheelEvent(event)


class NavigationEvents(InputEventsHandler):
	def __init__(self,parent):
		InputEventsHandler.__init__(self,parent)
		
	def mousePressEvent(self,event):
		EMImage3DGUIModule.mousePressEvent(self.parent(),event)
	
	def mouseReleaseEvent(self,event):
		EMImage3DGUIModule.mouseReleaseEvent(self.parent(),event)
	
	def mouseMoveEvent(self,event):
		EMImage3DGUIModule.mouseMoveEvent(self.parent(),event)
	
	def mouseDoubleClickEvent(self,event):
		EMImage3DGUIModule.mouseDoubleClickEvent(self.parent(),event)
	
	def keyPressEvent(self,event):
		EMImage3DGUIModule.keyPressEvent(self.parent(),event)
		
	def wheelEvent(self,event):
		EMImage3DGUIModule.wheelEvent(self.parent(),event)
		


class ClassOrientationEvents(InputEventsHandler,QtCore.QObject): 
	def __init__(self,parent):
		InputEventsHandler.__init__(self,parent)
		QtCore.QObject.__init__(self)
#		self.old_intersection = -1
#		self.old_color = None
#		self.nc = None
#		self.intsct = None
		self.current_hit = None
		
	def mousePressEvent(self,event):
		self.current_hit = self.get_hit(event)
		if self.current_hit == None: EMImage3DGUIModule.mousePressEvent(self.parent(),event)
	
	def mouseReleaseEvent(self,event):
		
		if self.current_hit != None:
			self.parent().updateGL() # there needs to be a clear or something  in order for the picking to work. This is  bit of hack but our rendering function doesn't take long anyhow
			hit = self.get_hit(event)
			if hit == self.current_hit:
				self.emit(QtCore.SIGNAL("point_selected"),self.current_hit,event)
		else:
			EMImage3DGUIModule.mouseReleaseEvent(self.parent(),event)
				
		self.current_hit = None
		
	def mouseMoveEvent(self,event):
		if not self.current_hit:
			EMImage3DGUIModule.mouseMoveEvent(self.parent(),event)
		
	def get_hit(self,event):
		v = self.parent().vdtools.wview.tolist()
		self.parent().gl_widget.makeCurrent() # prevents a stack underflow
#		x = event.x()
#		y = v[-1]-event.y()
#		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT )
#		vals = self.parent().render(color_picking=True)
#		glFlush()
#		vv = glReadPixels(x,y,1,1,GL_RGB,GL_FLOAT)
#		reslt = Vec3f(float(vv[0][0][0]),float(vv[0][0][1]),float(vv[0][0][2]))
#		for i,val in enumerate(vals):
##			print val,reslt,(reslt-val).length(),vv[0][0]
#			if (reslt-val).length() < 0.01:
#				print i
##				print (reslt-val).length()
#				return i
#		print vv
#		
		# the problem with this approach is that depth testing is not part of picking
		sb = [0 for i in xrange(0,512)]
		glSelectBuffer(512)
		glRenderMode(GL_SELECT)
		glInitNames()
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluPickMatrix(event.x(),v[-1]-event.y(),5,5,v)
		self.parent().gl_context_parent.load_perspective()
		glMatrixMode(GL_MODELVIEW)
		glInitNames()
		self.parent().render()
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		glFlush()
		
		intersection = None
		hits = list(glRenderMode(GL_RENDER))
		for hit in hits:
			a,b,c=hit
			if len(c) > 0:
				intersection = c[0]-1
				break
			
		return intersection
	
	def mouseDoubleClickEvent(self,event):
		EMImage3DGUIModule.mouseDoubleClickEvent(self.parent(),event)
	
	def keyPressEvent(self,event):
		EMImage3DGUIModule.keyPressEvent(self.parent(),event)
		
	def wheelEvent(self,event):
		EMImage3DGUIModule.wheelEvent(self.parent(),event)
	
class EMEulerExplorer(InputEventsManager,EM3DSymViewerModule,Animator):
	def get_desktop_hint(self): return "image"
	def keyPressEvent(self,event):
		
		if event.key() == Qt.Key_F1:
			self.display_web_help("http://blake.bcm.edu/emanwiki/EMAN2/Programs/e2eulerxplor")
		else:
			EMImage3DGUIModule.keyPressEvent(self,event)
	
	def __init__(self,application=None,ensure_gl_context=True,application_control=True,auto=True,sparse_mode=False):
		self.init_lock = True # a lock indicated that we are still in the __init__ function
		self.au_data = None # This will be a dictionary, keys will be refinement directories, values will be something like available iterations for visual study
		if auto: # this a flag that tells the eulerxplorer to search for refinement data and automatically add elements to the inspector, if so
			self.gen_refinement_data()
		
		EM3DSymViewerModule.__init__(self,application,ensure_gl_context=ensure_gl_context,application_control=application_control)
		InputEventsManager.__init__(self)
		Animator.__init__(self)
		self.height_scale = 8.0 # This is a value used in EM3DSymViewerModule which scales the height of the displayed cylinders - I made it 8 because it seemed fine. The user can change it anyhow
		self.__init_events_handlers() # Makes a dictionary of event handlers
		self.projection_file = None  # This is a string - the name of the projection images file
		self.average_file = None # This is a string - the name of the class averages file
		self.proj_class_viewer = None # This will be an EMImageMXModule that shows the class and/or projection
		self.particle_viewer = None  # This will be an EMImageMXModule that shows the particles in a class
		self.clsdb = None # I think this will become redundant - it used to be the old database that stores which particles are in a class, but now that's stored in the header
		self.particle_file = None # This will be a string - the name of the file that has the particle files in it. This might be made redundant with the new approach
		self.alignment_file = None # This will be a string - the name of the file containing the alignment parameters - this is essential if you we want to show the aligned particles
		self.refine_dir = None # This will be a string - the name of the current refinement directory that is being studied
		self.dx = None # This is an EMData object storing the x shifts of the alignments for all particles. Generated by e2classaverage
		self.dy = None # This is an EMData object storing the y shifts of the alignments for all particles. Generated by e2classaverage
		self.da = None# This is an EMData object storing the angle of the alignments for all particles. Generated by e2classaverage
		self.dflip = None # This is an EMData object storing whether or not tthe alignment involved a flip, for all particles. Generated by e2classaverage
		self.classes = None # This is an EMData object storing which class(es) a particle belongs to. Generated by e2classaverage
		self.inclusions = None # This is and EMDAta storing a boolean that indicates the particle was actually included in the final average. Generated by e2classaverage 
		
		self.average = None # This the class average itself, an EMData object
		self.projection = None # This is the projection itelse, an EMData object
		self.class_idx = None # This is the idx of the current class being studied in the interface
			
		self.previous_len = -1 # To keep track of the number of class averages that were previously viewable. This helps to make sure we can switch to the same class average in the context of a different refinement iteration
		self.mirror_eulers = False
		if sparse_mode:
			self.mirror_eulers = True # If True the drawn Eulers are are also rendered on the opposite side of the sphere - see EM3DSymViewerModule.make_sym_dl_lis
				
		# Grab the symmetry from the workflow database if possible
		sym = "icos"
		if db_check_dict("bdb:emform.e2refine"):
			db = db_open_dict("bdb:emform.e2refine",ro=True)
			if db.has_key("symname") and db.has_key("symnumber"):
				sym = db["symname"] + db["symnumber"]
			#db_close_dict("bdb:emform.e2refine")
		
		# Have to tell the EMImage3DSymModule that there is a new sym
		self.set_symmetry(sym)
		
		# this object will have 
		if self.au_data != None:
			combo_entries = self.au_data.keys()
			combo_entries.sort()
			combo_entries.reverse()
			
			if len(combo_entries) > 0:
				au = combo_entries[0]
				cls = self.au_data[au][0][0]
				self.au_selected(au,cls)
				self.mirror_eulers = True 
						
		self.init_lock = False
		self.force_update=True # Force a display udpdate in EMImage3DSymModule
		
	def __del__(self):
		EM3DSymViewerModule.__del__(self) # this is here for documentation purposes - beware that the del function is important

	def initializeGL(self):
		glEnable(GL_NORMALIZE)
	
	def generate_current_display_list(self,force=False):
		'''
		Redefinition of EMImage3DSymModule.generate_current_display_list
		
		'''
		if self.init_lock: return 0
		if self.au_data == None or len(self.au_data) == 0:
			EM3DSymViewerModule.generate_current_display_list(self,force)
		
		self.init_basic_shapes()
		if self.nomirror == True : val = 0
		else: val = 1
		self.trace_great_arcs(self.sym_object.get_asym_unit_points(val))
		self.trace_great_triangles(val)
		
		self.eulers = self.specified_eulers
		if self.eulers == None:	return 0
		
#		if not self.colors_specified: self.point_colors = []
#		else: self.point_colors = self.specified_colors
#		self.points = []
#		for i in self.eulers:
#			p = i.transpose()*Vec3f(0,0,self.radius)
#			self.points.append(p)
#			if not self.colors_specified: self.point_colors.append((0.34615, 0.3143, 0.0903,1))
		
		self.make_sym_dl_list(self.eulers)
		return 1
	def get_data_dims(self):
		return (2*self.radius,2*self.radius,2*self.radius)
	
	def width(self): return 2*self.radius
	def height(self): return 2*self.radius
	def depth(self): return 2*self.radius
	
	def gen_refinement_data(self):
		dirs,files = get_files_and_directories()
		
		dirs.sort()
		for i in range(len(dirs)-1,-1,-1):
			if len(dirs[i]) != 9:
				dirs.pop(i)
			elif dirs[i][:7] != "refine_":
				dirs.pop(i)
			else:
				try: int(dirs[i][7:])
				except: dirs.pop(i)
		
		self.dirs = dirs
		
		self.au_data = {}
		for dir in self.dirs:
			d = self.check_refine_db_dir(dir)
			if len(d) != 0 and len(d[dir]) != 0: self.au_data.update(d)

	def check_refine_db_dir(self,dir,s1="classes",s2="class_indices",s3="cls_result",s4="threed",s5="projections"):
		names = [s1,s2,s3,s4,s5]
		data = {}
		data[dir] = []
		register_db_name = "bdb:"+dir+"#register"
		# needs to exist
		if not db_check_dict(register_db_name):return {}
		db = db_open_dict(register_db_name,ro=True)
		if not db.has_key("cmd_dict"): return {}
		# need to be able to get the input data
		cmd = db["cmd_dict"]
		if cmd == None or not cmd.has_key("input"): return {}
		
		for i in range(0,9):
			for j in range(0,9):
				last_bit = str(i)+str(j)
				fail = False
				r = []
				# cmd dictionary needs to be stored
				
				for name in names:
					db_name = "bdb:"+dir+"#"+name+"_"+last_bit
					if not db_check_dict(db_name):
						fail= True
						break
					else: r.append(db_name)
	
				if not fail:
					data[dir].append(r)
		return data
	
	def set_projection_file(self,projection_file): self.projection_file = projection_file
	def get_inspector(self):
		if not self.inspector : 
			if (self.au_data == None or len(self.au_data) == 0) and self.mirror_eulers == False: #self.mirror_eulers thing is a little bit of a hack, it's tied to the sparse_mode flag in the init function, which is used by euler_display in EMAN2.py
				self.inspector=EMAsymmetricUnitInspector(self,True,True)
			else: 
				self.inspector=EMAsymmetricUnitInspector(self)
			QtCore.QObject.connect(self.inspector,QtCore.SIGNAL("au_selected"),self.au_selected)
		return self.inspector

	
	def au_selected(self,refine_dir,cls):
		self.refine_dir = refine_dir
		get_application().setOverrideCursor(Qt.BusyCursor)
		data = []
		for d in self.au_data[refine_dir]:
			if d[0] == cls:
				data = d;
				break
			
		if len(data) == 0:
			
			error("error, no data for %s %s, returning" %(refine_dir,cls))
#			print "error, no data for",au,cls,"returning"
			self.events_handlers["inspect"].reset()
			get_application().setOverrideCursor(Qt.ArrowCursor)
			return
		
		register_db_name = "bdb:"+refine_dir+"#register"
		if not db_check_dict(register_db_name):
			error("The %s database does not exist" %register_db_name )
			self.events_handlers["inspect"].reset()
			get_application().setOverrideCursor(Qt.ArrowCursor)
			return
		
		db = db_open_dict(register_db_name)
		if not db.has_key("cmd_dict"):
			error("The %s database must have the cmd entry" %register_db_name )
			self.events_handlers["inspect"].reset()
			get_application().setOverrideCursor(Qt.ArrowCursor)
			return
		
		cmd = db["cmd_dict"]
		
		if not cmd.has_key("input"):
			error("The %s database must have the cmd entry" %register_db_name )
			self.events_handlers["inspect"].reset()
			get_application().setOverrideCursor(Qt.ArrowCursor)
			return
		
		self.particle_file  = cmd["input"]
		
		self.average_file = cls
		self.projection_file = data[4]
		self.alignment_file = data[2]
		self.clsdb = data[1]
		
		self.dx = None
		self.dy = None
		self.da = None
		self.dflip = None
		self.classes = None
		
		eulers = get_eulers_from(self.average_file)
		#s = Symmetries.get("d7")
		#eulers = s.gen_orientations("rand",{"n":EMUtil.get_image_count(self.average_file)})

		self.specify_eulers(eulers)
		#from emimagemx import EMDataListCache
		#a = EMData.read_images(self.average_file)
		#a = [test_image() for i in range(EMUtil.get_image_count(self.average_file))]
		#print len(a),len(eulers)
		#b = [a[i].set_attr("xform.projection",eulers[i]) for i in range(len(eulers))]
		#b = [a[i].set_attr("ptcl_repr",1) for i in range(len(eulers))]
		
		self.set_emdata_list_as_data(EMLightWeightParticleCache.from_file(self.average_file),"ptcl_repr")
		#self.set_emdata_list_as_data(EMDataListCache(self.average_file),"ptcl_repr")
#		self.set_emdata_list_as_data(a,"ptcl_repr")
		self.force_update = True
		self.au_point_selected(self.class_idx,None)
		# if we have the same number of Eulers we can update everything
#		if self.previous_len == len(eulers) : self.events_handlers["inspect"].repeat_event()
#		else:self.events_handlers["inspect"].reset()
		self.previous_len = len(eulers)
		if not self.init_lock:self.updateGL()
		get_application().setOverrideCursor(Qt.ArrowCursor)
		
	def __get_file_headers(self,filename):
		headers = []
		n = EMUtil.get_image_count(filename)
		for i in range(n):
			e = EMData()
			e.read_image(filename,i,True)
			headers.append(e)
		return headers
	
	def __init_events_handlers(self):
		self.events_mode = "navigate"
		self.events_handlers = {}
		self.events_handlers["navigate"] = NavigationEvents(self)
		self.events_handlers["inspect"] = ClassOrientationEvents(self)
		QtCore.QObject.connect(self.events_handlers["inspect"],QtCore.SIGNAL("point_selected"), self.au_point_selected)
		self.current_events_handler = self.events_handlers["inspect"]
	
	def au_point_selected(self,i,event=None):
		if i == None: 
			if event != None and event.modifiers()&Qt.ShiftModifier:
				if self.special_euler != None:
					self.special_euler = None
					if not self.init_lock:self.regen_dl()
			return
#		self.arc_anim_points = None
		self.projection = None
		if self.euler_data:
#			db = db_open_dict(self.average_file)
#			a = db.get(i)
#			print a["nx"]
#			print self.average_file,i
#			self.average = EMData(self.average_file,i)
#			self.average["nx"]
			self.average = self.euler_data[i]#
			self.projection = EMData(self.projection_file,self.average.get_attr("projection_image_idx"))
			self.average.process_inplace("normalize.toimage",{"to":self.projection})
			try:
				self.class_idx = self.average.get_attr("projection_image_idx")
			except:
				self.class_idx = -1
		else: return
		
		#if self.projection  == None and self.average == None: return
		first = False
		if self.proj_class_viewer == None:
			first = True
			self.proj_class_viewer = EMImageMXModule(data=None,application=get_application())
			QtCore.QObject.connect(self.proj_class_viewer.emitter(),QtCore.SIGNAL("module_closed"),self.on_mx_view_closed)
			self.proj_class_viewer.set_mouse_mode("App" )
			QtCore.QObject.connect(self.proj_class_viewer.emitter(),QtCore.SIGNAL("mx_image_selected"), self.mx_image_selected)
			get_application().show_specific(self.proj_class_viewer)
			
		disp = []
		if self.projection != None: disp.append(self.projection)
		if self.average != None and self.projection!=None: 
			# ok, this really should be put into its own processor
			#dataf = self.projection.do_fft()
			#apix=self.projection["apix_x"]
			#curve = dataf.calc_radial_dist(dataf["ny"], 0, 0.5,True)
			#curve=[i/(dataf["nx"]*dataf["ny"])**2 for i in curve]
			#xcurve=[i/(apix*2.0*dataf["ny"]) for i in range(len(curve))]
			#xyd=XYData()
			#xyd.set_xy_list(xcurve,curve)
			#filt=self.average.process("filter.setstrucfac",{"apix":apix,"strucfac":xyd})
			#filt.process_inplace("normalize.toimage",{"to":self.average})
			filt=self.average.process("filter.matchto",{"to":self.projection})

			disp.append(filt)
			
		if self.average!=None:
			disp.append(self.average)

		self.proj_class_viewer.set_data(disp)
		
		self.proj_class_viewer.updateGL()
		if self.particle_viewer != None:
			self.mx_image_selected(None,None)
		if first: self.proj_class_viewer.optimally_resize()
		
		if i != self.special_euler:
			self.special_euler = i
			self.force_update = True
		
		if not self.init_lock: self.updateGL()
		
		
	def on_mx_view_closed(self):
		self.proj_class_viewer = None
		
	def on_particle_mx_view_closed(self):
		self.particle_viewer = None
	
	def animation_done_event(self,animation):
		pass
	
	def alignment_time_animation(self,transforms):
		if len(transforms) < 2: return
		animation = OrientationListAnimation(self,transforms,self.radius)
		self.register_animatable(animation)
			
	def particle_selected(self,event,lc):
		if lc != None:
			d = lc[3]
			ptcl_idx = d["Img #"]
			data = self.au_data[self.refine_dir]
			prj = []
			cls_result = []
			for l in data:
				for s in l:
					stag = get_file_tag(s)
					
					if len(stag) > 11 and stag[:11] == "projections":
						prj.append(s)
					elif len(stag) > 10 and stag[:10] == "cls_result":
						cls_result.append(s)
						
			transforms = []
			if len(prj) != len(cls_result): RunTimeError("The number of cls_result files does not match the number of projection files?")
			
			e = EMData()
			for i,cr in enumerate(cls_result):
				r = Region(0,ptcl_idx,1,1)
				e.read_image(cr,0,False,r)
				p = int(e.get(0))
				e.read_image(prj[i],p,True)
				transforms.append(e["xform.projection"])
				
			self.alignment_time_animation(transforms)

	def mx_image_selected(self,event,lc):
#		self.arc_anim_points = None
		get_application().setOverrideCursor(Qt.BusyCursor)
		if lc != None: self.sel = lc[0]
		
		if self.average != None:
			included = []
			if self.average.has_attr("class_ptcl_idxs"):
				included = self.average["class_ptcl_idxs"]
			excluded = []
			if self.average.has_attr("exc_class_ptcl_idxs"):
				excluded = self.average["exc_class_ptcl_idxs"]
				
			all = included + excluded
			#all.sort()
			
			bdata = []
			data = []
			idx_included = []
			running_idx = 0
			from emimagemx import ApplyAttribute
			for val in included:
				bdata.append([self.particle_file,val,[ApplyAttribute("Img #",val)]])
				idx_included.append(running_idx)
				running_idx += 1
			
			idx_excluded = []
			for val in excluded:
				bdata.append([self.particle_file,val,[ApplyAttribute("Img #",val)]])
				idx_excluded.append(running_idx)
				running_idx += 1
	
			data = EMLightWeightParticleCache(bdata)
			
			first = False
			if self.particle_viewer == None:
				first = True
				self.particle_viewer = EMImageMXModule(data=None,application=get_application())
				self.particle_viewer.set_mouse_mode("App" )
				QtCore.QObject.connect(self.particle_viewer.emitter(),QtCore.SIGNAL("module_closed"),self.on_particle_mx_view_closed)
				QtCore.QObject.connect(self.particle_viewer.emitter(),QtCore.SIGNAL("mx_image_selected"), self.particle_selected)
				get_application().show_specific(self.particle_viewer)
		
			
			self.check_images_in_memory()
			
			if self.sel== 0 or self.alignment_file == None:
				self.particle_viewer.set_data(data)
			else:
				
				for i,[name,idx,f] in enumerate(bdata):
					index = -1
					if self.classes.get_xsize() == 1:
						index = 0 # just assume it's the first one - this is potentially fatal assumption, but in obscure situations only
					else:
						for j in range(self.classes.get_xsize()):
							if int(self.classes.get(j,idx)) == self.class_idx:
								index = j
								break
					if index == -1:
						print "couldn't find"
						get_application().setOverrideCursor(Qt.ArrowCursor)
						return
				
					x = self.dx.get(index,idx)
					y = self.dy.get(index,idx)
					a = self.da.get(index,idx)
					m = self.dflip.get(index,idx)
					
					t = Transform({"type":"2d","alpha":a,"mirror":int(m)})
					t.set_trans(x,y)
					from emimagemx import ApplyTransform
					f.append(ApplyTransform(t))
					#data[i].transform(t)
				self.particle_viewer.set_data(data)

			
			if first:
				self.particle_viewer.updateGL()
				self.particle_viewer.optimally_resize()
			
			self.particle_viewer.clear_sets(False)
			self.particle_viewer.enable_set("Excluded",idx_excluded,True,False)
			self.particle_viewer.enable_set("Included",idx_included,False,False)
			self.particle_viewer.updateGL()
			
			get_application().setOverrideCursor(Qt.ArrowCursor)
			
			self.updateGL()
			
	def check_images_in_memory(self):
		if self.alignment_file != None:
			if self.dx == None:
				self.dx = EMData(self.alignment_file,2)
			if self.dy == None:
				self.dy = EMData(self.alignment_file,3)
			if self.da == None:
				self.da = EMData(self.alignment_file,4)
			if self.dflip == None:
				self.dflip  = EMData(self.alignment_file,5)
			if self.classes == None:
				self.classes  = EMData(self.alignment_file,0)
			if self.inclusions == None:
				self.inclusions  = EMData(self.alignment_file,1)
			
		
	def set_events_mode(self,mode):
		if not self.events_handlers.has_key(mode):
			print "error, unknown events mode", mode
			return
		
		else:
			self.current_events_handler = self.events_handlers[mode]
			
	def closeEvent(self,event):
		if self.inspector !=None: self.inspector.close()
		if self.proj_class_viewer !=None: self.proj_class_viewer.closeEvent(None)
		if self.particle_viewer != None: self.particle_viewer.closeEvent(None)
		get_application().close_specific(self)
		self.emit(QtCore.SIGNAL("module_closed")) # this signal is


def set_included_1(e):
	e.set_attr("included",1)
	e.mxset = [1]

def set_included_0(e):
	e.set_attr("included",0)
	e.mxset = [0]
	


class EMAsymmetricUnitInspector(EMSymInspector):
	def get_desktop_hint(self):
		return "inspector"
	def __init__(self,target,enable_trace=False,enable_og=False) :
		EMSymInspector.__init__(self,target,enable_trace=enable_trace,enable_og=enable_og)
		
		if target.au_data != None and len(target.au_data) > 0:
			self.add_au_table()
		
	def add_au_table(self):
		
		self.au_tab= QtGui.QWidget()
		self.au_tab.vbl = QtGui.QVBoxLayout(self.au_tab)
		
		self.au_data = self.target().au_data
		combo_entries = self.au_data.keys()
		combo_entries.sort()
		combo_entries.reverse()
		self.combo = QtGui.QComboBox(self)
		for e in combo_entries:
			self.combo.addItem(e)
			
		self.connect(self.combo,QtCore.SIGNAL("currentIndexChanged(QString&)"),self.on_combo_change)
		self.connect(self.combo,QtCore.SIGNAL("currentIndexChanged(const QString&)"),self.on_combo_change)
			
		self.au_tab.vbl.addWidget(self.combo)
		self.refine_dir = combo_entries[0]
		
		self.list_widget = QtGui.QListWidget(None)
		
		self.list_widget.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
		self.list_widget.setMouseTracking(True)
		QtCore.QObject.connect(self.list_widget,QtCore.SIGNAL("itemClicked(QListWidgetItem *)"),self.list_widget_item_clicked)
	
		self.update_classes_list(first_time=True)
		self.au_tab.vbl.addWidget(self.list_widget)
		self.tabwidget.insertTab(0,self.au_tab,"Refinement")
		self.tabwidget.setCurrentIndex(0)
	
	def on_combo_change(self,s):
		self.refine_dir = str(s)
		self.update_classes_list()
	
	def update_classes_list(self,first_time=False):
		selected_items = self.list_widget.selectedItems() # need to preserve the selection
		
		s_text = None
		if len(selected_items) == 1 :
			s_text = str(selected_items[0].text())
			if len(s_text) > 4: s_text = s_text[-4:] 
			
		self.list_widget.clear()
		for i,vals in enumerate(self.au_data[self.refine_dir]):
			choice = vals[0]
			
			a = QtGui.QListWidgetItem(str(choice),self.list_widget)
			if first_time and i == 0:
				self.list_widget.setItemSelected(a,True)
			elif len(choice) > 4 and (choice[-4:] == s_text):
				self.list_widget.setItemSelected(a,True)
				
		selected_items = self.list_widget.selectedItems() # need to preserve the selection
		if len(selected_items) == 1:
			self.emit(QtCore.SIGNAL("au_selected"),self.refine_dir,str(selected_items[0].text()))
	
	def list_widget_item_clicked(self,item):
		self.emit(QtCore.SIGNAL("au_selected"),self.refine_dir,str(item.text()))
	
	def on_mouse_mode_clicked(self,bool):
		for button in self.mouse_mode_buttons:
			if button.isChecked():
				self.target().set_events_mode(str(button.text()))
				
	

if __name__ == '__main__':
	main()

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
from EMAN2 import Util, E2init, E2end,EMANVERSION,is_2d_image_mx, EMUtil, db_open_dict, EMData, Transform, db_check_dict, db_close_dict, get_files_and_directories,get_file_tag,gimme_image_dimensions3D,Region, Vec3f, parsesym
from emapplication import get_application
from emimagemx import EMImageMXModule
import os
import sys
from libpyGLUtils2 import GLUtil
from emsprworkflow import error
from emanimationutil import OrientationListAnimation,Animator
from valslider import ValSlider

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


def get_ptcl_from(filename):
	ptcl = []
	n = EMUtil.get_image_count(filename)
	for i in range(n):
		h = get_header(filename,i)
		try: p = h["ptcl_repr"]
		except:
			print "image",i,"doesn't have the ptcl_repr attribute"
			return None
		ptcl.append(p)
#		
#	norm_ptcl = normalize_ptcl(ptcl)
	return ptcl

def check_projections_match_averages(projection_file, average_file):
	fine, message = is_2d_image_mx(projection_file)
	fine2, message2 = is_2d_image_mx(average_file)
	
	if not fine: print message # just print the messages first
	if not fine2: print message2
	
	if not fine or not fine2: return None
	
	if EMUtil.get_image_count(projection_file) != EMUtil.get_image_count(average_file):
		print "image count for projection and averages files don't match", EMUtil.get_image_count(projection_file), EMUtil.get_image_count(average_file)
		return None
	
	eulers = []
	ptcl = []
	for i in range(EMUtil.get_image_count(projection_file)):
		h1 = get_header(projection_file)
		p1 = h1["xform.projection"] # projection should definitely have this attribute
		eulers.append(p1)
		
		h2 = get_header(average_file)
		p2 = h2["ptcl_repr"] # projection should definitely have this attribute
		ptcl_append(p2)
		
	return eulers, ptcl

def normalize_ptcl(ptcl):
	mn = min(ptcl)
	mx = max(ptcl)
	diff = float(mx-mn)
	norm = [ (val-mn)/(diff) for val in ptcl ]
	return norm		
	
def get_header(filename,i):
	if filename[0:4] == "bdb:":
		db = db_open_dict(filename)
		return db.get_header(i)
	else:
		read_header_only = True
		e = EMData()
		e.read_image(filename,i,read_header_only)
		return e.get_attr_dict()

def get_normalize_colors(ptcls):
	mn = min(ptcls)
	mx = max(ptcls)
	df = float(mx-mn)
	colors = []
	for val in ptcls:
		if df != 0: val = (val-mn)/(df)
		else:
			val = 0.5
		if val < 0.5:
			frac = val/0.5
			colors.append((1.0,frac,frac,1.0))
		elif val > 0.5:
			frac = (-val+1.0)/0.5
			colors.append((frac,frac,1.0,1.0))
		else:
			colors.append((1.0,1.0,1.0,1.0))
			
	return colors


def get_normalize_colors_grey(ptcls):
	mn = min(ptcls)
	mx = max(ptcls)
	df = float(mx-mn)
	colors = []
	for val in ptcls:
		if df != 0: val = (val-mn)/(df)
		else:
			val = 0.5
		
		val = 0.2 + 0.8*val
		colors.append((val,val,val,1.0))
		
			
	return colors


def get_normalized_vector(ptcls):
	mn = min(ptcls)
	mx = max(ptcls)
	df = float(mx-mn)
	ret = []
	for val in ptcls:
		if df != 0: val = (val-mn)/(df)
		else:
			val = 0.5
		ret.append(val)
			
	return ret
	
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
	window = EMAsymmetricUnitViewer(application=em_app)

	em_app.show()
	em_app.execute()
	
	E2end(logid)

class InputEventsHandler:
	'''
	Perhaps the final installation in terms of what I think is the best design for the mouse events
	handling class. Others exist in emimage2d, emimagemx, and e2boxer. Eventually they should all use the same approach, and 
	I vote for using this one
	'''
	def __init__(self,parent):
		self.parent = weakref.ref(parent)
		
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
		InputEventsHandler.__init__(self,self)
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

class ClassOrientationEvents(NavigationEvents,QtCore.QObject): 
	def __init__(self,parent):
		NavigationEvents.__init__(self,parent)
		QtCore.QObject.__init__(self)
		self.old_intersection = -1
		self.old_color = None
		self.nc = None
		self.intsct = None
		
	def mouseReleaseEvent(self,event):
		m,p,v = self.parent().model_matrix.tolist(),self.parent().vdtools.wproj.tolist(),self.parent().vdtools.wview.tolist()
		
		model_matrix = []
		proj_matrix = []
		view_matrix = []
		
		
		for val in m:
			if isinstance(val,list): model_matrix.extend(val)
			else: modul_matrix.append(val)
			
		for val in p:
			if isinstance(val,list): proj_matrix.extend(val)
			else: proj_matrix.append(val)
			
		for val in v:
			if isinstance(val,list): view_matrix.extend(val)
			else: view_matrix.append(val)
			
		points = self.parent().points
		mouse_x = event.x()
		mouse_y = view_matrix[-1]-event.y()
		intersection = GLUtil.nearest_projected_points(model_matrix,proj_matrix,view_matrix,points,float(mouse_x),float(mouse_y),6.0)
		
		new_colors = {}
		
		if intersection >= 0:
			new_colors[intersection] = (1.0,1.0,0,1)
			
			if self.old_intersection >= 0:
				new_colors[self.old_intersection] = self.old_color
			
			self.old_intersection = intersection
			self.old_color = self.parent().point_colors[intersection]			
		
		else:
			if self.old_intersection >= 0:
				new_colors[self.old_intersection] = self.old_color
				self.old_intersection = -1
			else:
				NavigationEvents.mouseReleaseEvent(self,event)
				return
		
		if len(new_colors) > 0:
			self.nc = new_colors
			self.intsct = intersection
			self.parent().set_point_colors(new_colors)
			self.parent().updateGL()
			if intersection >= 0:self.emit(QtCore.SIGNAL("point_selected"),intersection)
	
		else:
			self.nc = None
			self.intsct = None
	def repeat_event(self):
		self.reset()
		if self.nc != None and self.intsct != None:
			self.parent().set_point_colors(self.nc)
			if self.intsct >= 0:self.emit(QtCore.SIGNAL("point_selected"),self.intsct)
			
	def reset(self):
		self.old_intersection = None
		self.old_color = None
	
class EMAsymmetricUnitViewer(InputEventsManager,EM3DSymViewerModule,Animator):
	def get_desktop_hint(self): return "image"
	def keyPressEvent(self,event):
		
		if event.key() == Qt.Key_F1:
			self.display_web_help("http://blake.bcm.edu/emanwiki/EMAN2/Programs/e2eulerxplor")
		else:
			EMImage3DGUIModule.keyPressEvent(self,event)
	def __init__(self,application,inspector_go=False,ensure_gl_context=True,application_control=True,auto=True):
		self.init_lock = True
		if auto:
			self.gen_refinement_data()
		self.euler_xplore_mode = False
		self.width_scale = 1.0
		self.height_scale = 1.0
		EM3DSymViewerModule.__init__(self,application,inspector_go=inspector_go,ensure_gl_context=ensure_gl_context,application_control=application_control)
		InputEventsManager.__init__(self)
		Animator.__init__(self)
		self.__init_events_handlers()
		self.projection_file = None
		self.average_file = None
		self.mx_viewer = None
		self.mx_particle_viewer = None
		self.clsdb = None
		self.particle_file = None
		self.alignment_file = None
		self.refine_dir = None
		self.dx = None
		self.dy = None
		self.da = None
		self.dflip = None
		self.classes = None
		self.inclusions = None
		self.ptcls = None
		
		self.average = None
		self.projection = None
		self.class_idx = None
		
		self.previous_len = -1
		
		sym = "icos"
		if db_check_dict("bdb:emform.e2refine"):
			db = db_open_dict("bdb:emform.e2refine",ro=True)
			if db.has_key("symname") and db.has_key("symnumber"):
				self.sym = db["symname"] + db["symnumber"]
			#db_close_dict("bdb:emform.e2refine")
		
		self.set_sym(sym)
		
		
		if hasattr(self,"au_data"):
			combo_entries = self.au_data.keys()
			combo_entries.sort()
			combo_entries.reverse()
		
			if len(combo_entries) > 0:
				au = combo_entries[0]
				cls = self.au_data[au][0][0]
				self.au_selected(au,cls)
				self.euler_xplore_mode = True
						
		self.init_lock = False
		self.regen_dl()
		
	def set_width_scale(self,val):
		self.width_scale = val
		self.regen_dl()
		
	def set_height_scale(self,val):
		self.height_scale = val
		self.regen_dl()
	
	def initializeGL(self):
		glEnable(GL_NORMALIZE)
	
	def generate_current_display_list(self,force=False):
		
		if self.init_lock: return 0
		if not self.euler_xplore_mode:
			EM3DSymViewerModule.generate_current_display_list(self,force)
		
		self.init_basic_shapes()
		if self.nomirror == True : val = 0
		else: val = 1
		self.trace_great_arcs(self.sym_object.get_asym_unit_points(val))
		self.trace_great_triangles(val)
		
		self.eulers = self.specified_eulers
		if self.eulers == None:
			return 0
		if not self.colors_specified: self.point_colors = []
		else: self.point_colors = self.specified_colors
		self.points = []
		for i in self.eulers:
			p = i.transpose()*Vec3f(0,0,self.radius)
			self.points.append(p)
			if not self.colors_specified: self.point_colors.append((0.34615, 0.3143, 0.0903,1))
		
		self.make_sym_dl_list(self.points,self.point_colors,self.eulers,self.ptcls,self.width_scale,self.height_scale)
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
		for i in range(0,9):
			for j in range(0,9):
				last_bit = str(i)+str(j)
				fail = False
				r = []
				
				register_db_name = "bdb:"+dir+"#register"
				
				# needs to exist
				if not db_check_dict(register_db_name):
					continue
				# cmd dictionary needs to be stored
				db = db_open_dict(register_db_name,ro=True)
				if not db.has_key("cmd_dict"):
					continue
		
				cmd = db["cmd_dict"]
				# need to be able to get the input data
				if not cmd.has_key("input"):
					continue
				
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
			if not self.euler_xplore_mode: 
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
		ptcls = get_ptcl_from(self.average_file)
		self.ptcls = get_normalized_vector(ptcls)
		self.specify_eulers(eulers)
		self.specify_colors(get_normalize_colors_grey(ptcls))
		self.force_update = True
		#self.generate_current_display_list(force=True)
		
		# if we have the same number of Eulers we can update everything
		if self.previous_len == len(eulers) : self.events_handlers["inspect"].repeat_event()
		else:self.events_handlers["inspect"].reset()
		self.previous_len = len(eulers)
		self.updateGL()
		get_application().setOverrideCursor(Qt.ArrowCursor)
	
	def __init_events_handlers(self):
		self.events_mode = "navigate"
		self.events_handlers = {}
		self.events_handlers["navigate"] = NavigationEvents(self)
		self.events_handlers["inspect"] = ClassOrientationEvents(self)
		QtCore.QObject.connect(self.events_handlers["inspect"],QtCore.SIGNAL("point_selected"), self.au_point_selected)
		self.current_events_handler = self.events_handlers["inspect"]
	
	def au_point_selected(self,i):
		self.arc_anim_points = None
		self.projection = None
		try:
			self.average = EMData(self.average_file,i)
			self.projection = EMData(self.projection_file,self.average.get_attr("projection_image_idx"))
			self.average.process_inplace("normalize.toimage.lsq",{"to":self.projection})
			try:
				self.class_idx = self.average.get_attr("projection_image_idx")
			except:
				self.class_idx = -1
		except: self.average = None
		
		if self.projection  == None and self.average == None: return
		
		first = False
		if self.mx_viewer == None:
			first = True
			self.mx_viewer = EMImageMXModule(data=None,application=get_application())
			QtCore.QObject.connect(self.mx_viewer.emitter(),QtCore.SIGNAL("module_closed"),self.on_mx_view_closed)
			self.mx_viewer.set_mouse_mode("app" )
			QtCore.QObject.connect(self.mx_viewer.emitter(),QtCore.SIGNAL("mx_image_selected"), self.mx_image_selected)
			get_application().show_specific(self.mx_viewer)
			
		
		disp = []
		if self.projection != None: disp.append(self.projection)
		if self.average != None: disp.append(self.average)

		self.mx_viewer.set_data(disp)
		
		self.mx_viewer.updateGL()
		
		if self.mx_particle_viewer != None:
			self.mx_image_selected(None,None)
			
		if first: self.mx_viewer.optimally_resize()
		
		self.updateGL()
			
	def on_mx_view_closed(self):
		self.mx_viewer = None
		
	def on_particle_mx_view_closed(self):
		self.mx_particle_viewer = None
	
	def animation_done_event(self,animation):
		pass
	
	
	def alignment_time_animation(self,transforms):
		if len(transforms) < 2: return
		animation = OrientationListAnimation(self,transforms,self.radius)
		self.register_animatable(animation)
			
	def particle_selected(self,event,lc):
		if lc != None:
			d = lc[3]
			ptcl_idx = d["ptcl_idx"]
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
		self.arc_anim_points = None
		get_application().setOverrideCursor(Qt.BusyCursor)
		if lc != None: self.sel = lc[0]
		if self.clsdb != None and self.particle_file != None and self.class_idx > -1:
			class_db = db_open_dict(self.clsdb)
			indices = class_db[str(self.class_idx)]
			mx = []
			if indices == None: return
			for val in indices:
				e = EMData(self.particle_file,val)
				e.set_attr("ptcl_idx",val)
				mx.append(e)
			
			first = False
			if self.mx_particle_viewer == None:
				first = True
				self.mx_particle_viewer = EMImageMXModule(data=None,application=get_application())
				self.mx_particle_viewer.set_mouse_mode("app" )
				QtCore.QObject.connect(self.mx_particle_viewer.emitter(),QtCore.SIGNAL("module_closed"),self.on_particle_mx_view_closed)
				QtCore.QObject.connect(self.mx_particle_viewer.emitter(),QtCore.SIGNAL("mx_image_selected"), self.particle_selected)
				get_application().show_specific(self.mx_particle_viewer)
			
			
			self.check_images_in_memory()
			incs = []
			excs = []
			for i,idx in enumerate(indices):
				index = -1
				for j in range(self.classes.get_xsize()):
					if int(self.classes.get(j,idx)) == self.class_idx:
						index = j
						break
				if index == -1:
					print "couldn't find"
					get_application().setOverrideCursor(Qt.ArrowCursor)
					return
				kept = self.inclusions.get(index,idx)
				if kept:
					incs.append(i)
				else: excs.append(i)
				mx[i]["included"] = kept
				mx[i].mxset = [kept]
			
			if self.sel== 0 or self.alignment_file == None:
				self.mx_particle_viewer.set_data(mx)
			else:
				
				for i,idx in enumerate(indices):
					index = -1
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
					mx[i].transform(t)
				self.mx_particle_viewer.set_data(mx)

			
			if first:
				self.mx_particle_viewer.updateGL()
				self.mx_particle_viewer.optimally_resize()
				
			self.mx_particle_viewer.enable_set(0,"Excluded",True,excs)
			self.mx_particle_viewer.enable_set(1,"Included",False,incs)
			self.mx_particle_viewer.updateGL()
			
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
		self.emit(QtCore.SIGNAL("module_closed")) # this signal is
		if self.inspector !=None: self.inspector.close()
		if self.mx_viewer !=None: self.mx_viewer.closeEvent(None)
		if self.mx_particle_viewer != None: self.mx_particle_viewer.closeEvent(None)
		get_application().close_specific(self)
		
	
#	def get_inspector(self):
#		pass

def get_alignment(dir_tag="00",iter="00",ptcl=0,post_align=False):
	'''
	Get alignment data.
	dir_tag corresponds to the refinement directory. If you specify "01" this will look in "refine_01"
	iter corresponds to the refinement iteration. For example if you specify "02", data from "projections_02", "classify_02" will be used, etc.
	ptcl is the particle number.
	post_align, if False, gets alignment parameters from before class averaging (classify_XX). If true it is after (cls_result_XX) 
	Return value is [Projection transfrom, Alignment transform, [projection,particle,aligned particle]]
	
	For example, 
	a = get_alignment("00","00",0,False)
	display(a[2])
	a[0].get_params("eman")
	a[1].get_params("2d")
	'''
	directory = "refine_"+str(dir_tag)
	if not os.path.isdir("refine_"+dir_tag):
		print "The directory",directory,"does not exist"
		return None
	
	if post_align:
		db_ali = "bdb:"+directory+"#cls_result_"+iter
	else:
		db_ali = "bdb:"+directory+"#classify_"+iter
	if not db_check_dict(db_ali):
		print  "Error, can't open database:", db_ali,".. please check your arguments"

	prj_file = "bdb:"+directory+"#projections_"+iter
	if not db_check_dict(prj_file):
		print  "Error, can't open database:", prj_file,".. please check your arguments"
	
	classes  = EMData(db_ali,0)
	kept = EMData(db_ali,1)
	dx = EMData(db_ali,2)
	dy = EMData(db_ali,3)
	da = EMData(db_ali,4)
	dflip  = EMData(db_ali,5)
	
	class_idx = int(classes.get(ptcl))
	
	projection = EMData(prj_file,class_idx)
	
	x = dx.get(ptcl)
	y = dy.get(ptcl)
	a = da.get(ptcl)
	m = dflip.get(ptcl)
	kept = kept.get(ptcl)
	
	#print "Class and ali parms are",class_idx,x,y,a,m
	
	t = Transform({"type":"2d","alpha":a,"mirror":int(m)})
	t.set_trans(x,y)
	
	ptcl_db = "bdb:"+directory+"#all"
	ptcl = EMData(ptcl_db,ptcl)
	ptcl_c = ptcl.copy()
	ptcl_c.transform(t)
	#print directory,db_ali,prj_file,ptcl_db
	return [projection["xform.projection"], t,[projection,ptcl,ptcl_c],kept]

class EMAsymmetricUnitInspector(EMSymInspector):
	def get_desktop_hint(self):
		return "inspector"
	def __init__(self,target,enable_trace=False,enable_og=False) :
		EMSymInspector.__init__(self,target,enable_trace=enable_trace,enable_og=enable_og)
		
		if hasattr(self.target(),"au_data") and len(self.target().au_data) > 0:
			self.init_au_table()
			self.init_scale_options()

	def init_scale_options(self):
		
		hbl = QtGui.QHBoxLayout()
		self.width_scale = ValSlider(self,(0.05,10.0),"Width scale:")
		self.width_scale.setValue(self.target().width_scale)
		self.height_scale = ValSlider(self,(0.05,10.0),"Height scale:")
		self.height_scale.setValue(self.target().height_scale)
		hbl.addWidget(self.width_scale)
		hbl.addWidget(self.height_scale)
		self.vbl2.addLayout(hbl)
		
		QtCore.QObject.connect(self.width_scale, QtCore.SIGNAL("valueChanged"), self.target().set_width_scale)
		QtCore.QObject.connect(self.height_scale, QtCore.SIGNAL("valueChanged"), self.target().set_height_scale)
		
		

	def init_au_table(self):
		self.au_data = self.target().au_data
		combo_entries = self.au_data.keys()
		combo_entries.sort()
		combo_entries.reverse()
		self.combo = QtGui.QComboBox(self)
		for e in combo_entries:
			self.combo.addItem(e)
			
		self.connect(self.combo,QtCore.SIGNAL("currentIndexChanged(QString&)"),self.on_combo_change)
		self.connect(self.combo,QtCore.SIGNAL("currentIndexChanged(const QString&)"),self.on_combo_change)
			
		self.vbl2.addWidget(self.combo)
		self.refine_dir = combo_entries[0]
		
		self.list_widget = QtGui.QListWidget(None)
		
		self.list_widget.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
		self.list_widget.setMouseTracking(True)
		QtCore.QObject.connect(self.list_widget,QtCore.SIGNAL("itemClicked(QListWidgetItem *)"),self.list_widget_item_clicked)
	
		self.update_classes_list(first_time=True)
		self.vbl2.addWidget(self.list_widget)
	
	def on_combo_change(self,s):
		self.refine_dir = str(s)
		print self.refine_dir
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
#!/usr/bin/env python
#
# Author: John Flanagan (jfflanag@bcm.edu)
# Copyright (c) 2000-2006 Baylor College of Medicine


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
from OpenGL.GL import *
from OpenGL import GLU
from PyQt4 import QtCore, QtGui, QtOpenGL 
from PyQt4.QtCore import Qt
from emapplication import EMGLWidget
from emitem3d import EMItem3D
from libpyGLUtils2 import GLUtil
from valslider import ValSlider, EMSpinWidget, EMQTColorWidget
import math
import weakref

# XPM format Cursors

visibleicon = [
    '16 12 3 1',
    'a c #0000ff',
    'b c #000000',
    'c c None',
    'cccccccccccccccc',
    'ccccccbbbbcccccc',
    'ccccbbbbbbbbcccc',
    'ccbbbccccccbbbcc',
    'cbbccccaaccccbbc',
    'cbccccaaaaccccbc',
    'cbccccaaaaccccbc',
    'cbbccccaaccccbbc',
    'ccbbbccccccbbbcc',
    'ccccbbbbbbbbcccc',
    'ccccccbbbbcccccc',
    'cccccccccccccccc'
]
    
invisibleicon = [
    '16 12 2 1',
    'b c #000000',
    'c c None',
    'cbbcccccccccbbcc',
    'ccbbcccccccbbccc',
    'cccbbcccccbbcccc',
    'ccccbbcccbbccccc',
    'cccccbbcbbcccccc',
    'ccccccbbbccccccc',
    'ccccccbbbccccccc',
    'cccccbbcbbcccccc',
    'ccccbbcccbbccccc',
    'cccbbcccccbbcccc',
    'ccbbcccccccbbccc',
    'cbbcccccccccbbcc'
]

zrotatecursor = [
    '15 14 2 1',
    'b c #00ff00',
    'c c None',
    'ccccccccccccccc',
    'ccccbbbbbbccbcc',
    'ccbbbbbbbbbbbbc',
    'cbbbcccccbbbbbc',
    'bbbcccccbbbbbbb',
    'bbcccccbbbbbbbc',
    'ccccccccccccccc',
    'ccccccccccccccc',
    'cbbbbbbbcccccbb',
    'bbbbbbbcccccbbb',
    'cbbbbbcccccbbbc',
    'cbbbbbbbbbbbbcc',
    'ccbccbbbbbbcccc',
    'ccccccccccccccc'
]

xyrotatecursor = [
    '14 13 2 1',
    'b c #00ff00',
    'c c None',
    'cccccccccccccc',
    'ccccbbbbbbcccc',
    'ccbbbbbbbbbbcc',
    'cbbbccccccbbbc',
    'bbbccccccccbbb',
    'bbbccccccccbbb',
    'bbbccccccccbbb',
    'bbbccccccccbbb',
    'bbbccccccccbbb',
    'cbbbccccccbbbc',
    'ccbbbbbbbbbbcc',
    'ccccbbbbbbcccc',
    'cccccccccccccc'
]

crosshairscursor = [
    '16 16 2 1',
    'b c #00ff00',
    'c c None',
    'cccccccbbcccccccc',
    'ccccccbbbbccccccc',
    'cccccbbbbbbcccccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'ccbccccbbccccbccc',
    'cbbccccbbccccbbcc',
    'bbbbbbbbbbbbbbbbb',
    'bbbbbbbbbbbbbbbbb',
    'cbbccccbbccccbbcc',
    'ccbccccbbccccbccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'cccccbbbbbbcccccc',
    'ccccccbbbbccccccc',
    'cccccccbbcccccccc'
]   
 
zhaircursor = [
    '16 16 2 1',
    'b c #00ff00',
    'c c None',
    'cccccccbbcccccccc',
    'ccccccbbbbccccccc',
    'cccccbbbbbbcccccc',
    'ccccbbcbbcbbccccc',
    'cccbbccbbccbbcccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'cccccccbbcccccccc',
    'cccbbccbbccbbcccc',
    'ccccbbcbbcbbccccc',
    'cccccbbbbbbcccccc',
    'ccccccbbbbccccccc',
    'cccccccbbcccccccc'
]   
scalecursor = [
    '16 16 2 1',
    'b c #00ff00',
    'c c None',
    'bbbbbbbbcccccccc',
    'bccccccccccccccc',
    'bccccccccccccccc',
    'bccccccccccccccc',
    'bccccccccccccccc',
    'bccccbbbbbbccccc',
    'bccccbccccbccccc',
    'bccccbccccbccccc',
    'cccccbccccbccccb',
    'cccccbccccbccccb',
    'cccccbbbbbbccccb',
    'cccccccccccbcccb',
    'ccccccccccccbccb',
    'cccccccccccccbcb',
    'ccccccccccccccbb',
    'ccccccccbbbbbbbb'
]   

selectorcursor = [
    '16 16 2 1',
    'b c #00ff00',
    'c c None',
    'cbbbbbbbbbcccccc',
    'bcccccccccbccccc',
    'cbbbbbbbcccbcccc',
    'cccbccccccccbccc',
    'ccccbbbbccccbccc',
    'cccbccccccccbccc',
    'ccccbbbbcccbcbcc',
    'cccbccccccbcccbc',
    'ccccbbbbbbcccccb',
    'cccccccbccbcccbc',
    'ccccccccbccccbcc',
    'cccccccccbccbccc',
    'ccccccccccbbcccc',
    'cccccccccccccccc',
    'cccccccccccccccc',
    'cccccccccccccccc'
]

class EMScene3D(EMItem3D, EMGLWidget):
	"""
	Widget for rendering 3D objects. Uses a scne graph for rendering
	"""
	def __init__(self, parentwidget=None, SGactivenodeset=set(), scalestep=0.5):
		"""
		@param parent: The parent of the widget
		@param SGnodelist: a list enumerating all the SGnodes
		@param SGactivenodeset: a set enumerating the list of active nodes
		@param scalestep: The step to increment the object scaling
		"""
		EMItem3D.__init__(self, parent=None, transform=Transform())
		EMGLWidget.__init__(self,parentwidget)
		QtOpenGL.QGLFormat().setDoubleBuffer(True)
		self.camera = EMCamera(1.0, 10000.0)	# Default near,far, and zclip values
		self.main_3d_inspector = None
		self.widget = None				# Get the inspector GUI
		#self.SGactivenodeset = SGactivenodeset			# A set of all active nodes (currently not used)
		self.scalestep = scalestep				# The scale factor stepsize
		self.toggle_render_selectedarea = False			# Don't render the selection box by default
		self.zrotatecursor = QtGui.QCursor(QtGui.QPixmap(zrotatecursor),-1,-1)
		self.xyrotatecursor = QtGui.QCursor(QtGui.QPixmap(xyrotatecursor),-1,-1)
		self.crosshaircursor = QtGui.QCursor(QtGui.QPixmap(crosshairscursor),-1,-1)
		self.scalecursor = QtGui.QCursor(QtGui.QPixmap(scalecursor),-1,-1)
		self.zhaircursor = QtGui.QCursor(QtGui.QPixmap(zhaircursor),-1,-1)
		self.selectorcursor = QtGui.QCursor(QtGui.QPixmap(selectorcursor),-1,-1)

	def initializeGL(self):
		glClearColor(0.0, 0.0, 0.0, 0.0)		# Default clear color is black
		glShadeModel(GL_SMOOTH)
		glEnable(GL_DEPTH_TEST)
		self.firstlight = EMLight(GL_LIGHT0)
		self.firstlight.enableLighting()
        
	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)		
		glColor3f(1.0, 1.0, 1.0)	# Default color is white
		#Call rendering
		self.renderSelectedArea() 	# Draw the selection box if needed
		self.render()			# SG nodes must have a render method
		glFlush()			# Finish rendering

	def resizeGL(self, width, height):
		self.camera.update(width, height)
	
	def getSceneGui(self):
		"""
		Return a Qt widget that controls the scene item
		"""	
		if not self.widget: self.widget = EMInspectorControlBasic("SG", self)
		return self.widget
		
	def renderNode(self):
		pass
	
	def setInspector(self, inspector):
		"""
		Set the main 3d inspector
		"""
		self.main_3d_inspector = inspector
		
	def pickItem(self):
		"""
		Pick an item on the screen using openGL's selection mechanism
		"""
		viewport = glGetIntegerv(GL_VIEWPORT)
		glSelectBuffer(1024)	# The buffer size for selection
		glRenderMode(GL_SELECT)
		glInitNames()
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		
		# Find the selection box. Go from Volume view coords to viewport coords
		x = self.sa_xi + self.camera.getWidth()/2
		y = self.camera.getHeight()/2 - self.sa_yi
		dx = 2*math.fabs(self.sa_xi - self.sa_xf) # The 2x is a hack.....
		dy = 2*math.fabs(self.sa_yi - self.sa_yf) # The 2x is a hack.....
		
		# Apply selection box
		GLU.gluPickMatrix(x, viewport[3] - y, dx, dy, viewport)
		self.camera.setProjectionMatrix()
		
		#drawstuff, but first we need to remove the influence of any previous xforms which ^$#*$ the selection
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		self.camera.setCameraPosition(sfactor=2) # Factor of two to compensate for the samera already being set
		self.render()
		glPopMatrix()
		
		# Return to default state
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode(GL_MODELVIEW)
		records = glRenderMode(GL_RENDER)
		
		# process records
		self.procesSelection(records)
	
	def selectArea(self, xi, xf, yi, yf):
		"""
		Set an area for selection. Need to switch bewteen viewport coords, where (0,0 is bottom left) to
		volume view coords where 0,0) is center of the screen.
		"""
		self.sa_xi = xi - self.camera.getWidth()/2
		self.sa_xf = xf - self.camera.getWidth()/2
		self.sa_yi = -yi + self.camera.getHeight()/2
		self.sa_yf = -yf + self.camera.getHeight()/2
		self.toggle_render_selectedarea = True
		
	def deselectArea(self):
		"""
		Turn off selectin box
		"""
		self.sa_xi = 0.0
		self.sa_xf = 0.0
		self.sa_yi = 0.0
		self.sa_yf = 0.0
		self.toggle_render_selectedarea = False
		
	def renderSelectedArea(self):
		"""
		Draw the selection box, box is always drawn orthographically
		"""
		if self.toggle_render_selectedarea: 
			glMatrixMode(GL_PROJECTION)
			glPushMatrix()
			glLoadIdentity()
			self.camera.setOrthoProjectionMatrix()
			glColor3f(0.0,1.0,0.0)
			glMaterialfv(GL_FRONT, GL_EMISSION, [0.0,1.0,0.0,1.0])
			glBegin(GL_LINE_LOOP)
			z = -self.camera.getZclip() - 1
			glVertex3f(self.sa_xi, self.sa_yi, z)
			glVertex3f(self.sa_xi, self.sa_yf, z)
			glVertex3f(self.sa_xf, self.sa_yf, z)
			glVertex3f(self.sa_xf, self.sa_yi, z)
			glEnd()
			glPopMatrix()
			glMatrixMode(GL_MODELVIEW)
			glMaterialfv(GL_FRONT, GL_EMISSION, [0.0,0.0,0.0,1.0])
		
	def procesSelection(self, records):
		"""
		Process the selection records
		"""
		# Remove old selection if not in append mode
		if not self.appendselection:
			for selected in self.getAllSelectedNodes():
				selected.is_selected = False
				# Inspector tree management
				if EMQTreeWidgetItem:
					selected.EMQTreeWidgetItem.setSelectionStateBox()
					self.main_3d_inspector.tree_widget.setCurrentItem(selected.EMQTreeWidgetItem)
		# Select the desired items	
		closestitem = None
		bestdistance = 1.0
		for record in records:
			selecteditem = EMItem3D.selection_idx_dict[record.names[len(record.names)-1]]()
			selecteditem.is_selected = True
			# Inspector tree management
			if EMQTreeWidgetItem:
				selecteditem.EMQTreeWidgetItem.setSelectionStateBox()
				self.main_3d_inspector.tree_widget.setCurrentItem(selecteditem.EMQTreeWidgetItem)
			try:
				self.main_3d_inspector.stacked_widget.setCurrentWidget(selecteditem.getSceneGui())
				self.main_3d_inspector.tree_widget.setCurrentItem(selecteditem.getSceneGui().treeitem)	# Hmmm... tak about tight coupling! It would be better to use getters
			except:
				pass
			#if record.near < bestdistance:
				#bestdistance = record.near
				#closestitem = record.names
			
	# Event subclassing
	def mousePressEvent(self, event):
		"""
		QT event handler. Records the coords when a mouse button is pressed and sets the cursor depending on what button(s) are pressed
		"""
		# The previous x,y records where the mouse was prior to mouse move, rest upon mouse move
		self.previous_x = event.x()
		self.previous_y = event.y()
		# The first x,y records where the mouse was first pressed
		self.first_x = self.previous_x
		self.first_y = self.previous_y
		if event.buttons()&Qt.LeftButton:
			if event.modifiers()&Qt.ControlModifier:
				self.setCursor(self.selectorcursor)
				self.appendselection = False
				if event.modifiers()&Qt.ShiftModifier:
					self.appendselection = True
			else:
				if  event.y() > 0.95*self.size().height():
					self.setCursor(self.zrotatecursor)
				else:
					self.setCursor(self.xyrotatecursor)
		if event.buttons()&Qt.MidButton:
			if event.modifiers()&Qt.ControlModifier:
				self.setCursor(self.zhaircursor)
			else:
				self.setCursor(self.crosshaircursor)
		if event.buttons()&Qt.RightButton:
			self.setCursor(self.scalecursor)		
			
	def mouseMoveEvent(self, event):
		"""
		Qt event handler. Scales the SG depending on what mouse button(s) are pressed when dragged
		"""
		dx = event.x() - self.previous_x
		dy = event.y() - self.previous_y
		if event.buttons()&Qt.LeftButton:
			if event.modifiers()&Qt.ControlModifier:
				self.setCursor(self.selectorcursor)
				self.selectArea(self.first_x, event.x(), self.first_y, event.y())
			else:
				magnitude = math.sqrt(dx*dx + dy*dy)
				#Check to see if the cursor is in the 'virtual slider pannel'
				if  event.y() > 0.95*self.size().height(): # The lowest 5% of the screen is reserved from the Z spin virtual slider
					self.setCursor(self.zrotatecursor)
					self.update_matrices([magnitude,0,0,-dx/magnitude], "rotate")
				else:
					self.setCursor(self.xyrotatecursor) 
					self.update_matrices([magnitude,-dy/magnitude,-dx/magnitude,0], "rotate")
			self.updateSG()
		if event.buttons()&Qt.MidButton:
			if event.modifiers()&Qt.ControlModifier:
				self.update_matrices([0,0,(dx+dy)], "translate")
			else:
				self.update_matrices([dx,-dy,0], "translate")
			self.updateSG()	
		if event.buttons()&Qt.RightButton:
			self.update_matrices([self.scalestep*0.1*(dx+dy)], "scale")
			self.setCursor(self.scalecursor)
			self.updateSG()
		self.previous_x =  event.x()
		self.previous_y =  event.y()
			
			
	def mouseReleaseEvent(self, event):
		"""
		Qt event handler. Returns the cursor to arrow unpn mouse button release
		"""
		self.setCursor(Qt.ArrowCursor)
		if self.toggle_render_selectedarea:
			self.pickItem()
			self.deselectArea()
			self.updateSG()
	
	def wheelEvent(self, event):
		"""
		QT event handler. Scales the SG unpon wheel movement
		"""
		if event.orientation() & Qt.Vertical:
			if event.delta() > 0:
				self.update_matrices([self.scalestep], "scale")
			else:
				self.update_matrices([-self.scalestep], "scale")
			self.updateSG()
			
	def mouseDoubleClickEvent(self,event):
		print "Mouse Double Click Event"
	
	def keyPressEvent(self,event):
		print "Mouse KeyPress Event"
	
	def updateSG(self):
		"""
		Update the SG
		"""
		QtOpenGL.QGLWidget.updateGL(self)
	
	# Maybe add methods to control the lights

class EMLight:
	def __init__(self, light):
		"""
		@type light: GL_LIGHTX, where 0 =< X <= 8
		@param light: an OpenGL light
		The light properties are set to reasnonale defaults.
		"""
		self.light = light
		self.setAmbient(0.1, 0.1, 0.1, 1.0)		# Default ambient color is light grey
		self.setDiffuse(1.0, 1.0, 1.0, 1.0)		# Default diffuse color is white
		self.setSpecualar(1.0, 1.0, 1.0, 1.0)		# Default specular color is white
		self.setPosition(0.0, 0.0, 1.0, 0.0)		# Defulat position is 0, 0, 1.0 and light is directional (w=0)
		if not glIsEnabled(GL_LIGHTING):
			glEnable(GL_LIGHTING)

	def setAmbient(self, r, g, b, a):
		"""
		@param r: the red component of the ambient light
		@param g: the green component of the ambient light
		@param b: the blue component of the ambient light
		@param a: the alpha component of the ambient light
		Set the ambient light color
		"""
		self.colorambient = [r, g, b, a]
		glLightfv(self.light, GL_AMBIENT, self.colorambient)

	def setDiffuse(self, r, g, b, a):
		"""
		@param r: the red component of the diffuse and specular light
		@param g: the green component of the diffuse and specular light
		@param b: the blue component of the diffuse and specular light
		@param a: the alpha component of the diffuse and specular light
		Set the diffuse light color
		"""
		self.colordiffuse = [r, g, b, a]
		glLightfv(self.light, GL_DIFFUSE, self.colordiffuse)
		
	def setSpecualar(self, r, g, b, a):
		"""
		@param r: the red component of the diffuse and specular light
		@param g: the green component of the diffuse and specular light
		@param b: the blue component of the diffuse and specular light
		@param a: the alpha component of the diffuse and specular light
		Set the specualr light color
		"""
		self.colorspecular = [r, g, b, a]
		glLightfv(self.light, GL_SPECULAR, self.colorspecular)

	def setPosition(self, x, y, z, w):
		"""
		@param x: The x component of the light position
		@param y: The y component of the light position
		@param z: The z component of the light position
		@param w: The w component of the light position
		Set the light position, in gomogenious corrds
		"""
		self.position = [x, y, z, w]
		glLightfv(self.light, GL_POSITION, self.position)

	def enableLighting(self):
		"""
		Enables this light
		"""
		if not glIsEnabled(self.light):
			glEnable(self.light)

	def disableLighting(self):
		"""
		Disables this light
		"""
		if glIsEnabled(self.light):
			glDisable(self.light)
class EMCamera:
	"""Implmentation of the camera"""
	def __init__(self, near, far, zclip=-1000.0, usingortho=True, fovy=60.0, boundingbox=50.0, screenfraction=0.5):
		"""
		@param fovy: The field of view angle
		@param near: The volume view near position
		@param far: The volume view far position
		@param zclip: The zclipping plane (basicaly how far back the camera is)
		@param usingortho: Use orthographic projection
		@param boundingbox: The dimension of the bounding for the object to be rendered
		@param screenfraction: The fraction of the screen height to occupy
		"""
		self.far = far
		self.near = near
		if usingortho:
			self.useOrtho(zclip)
		else:
			self.usePrespective(boundingbox, screenfraction, fovy)

	def update(self, width, height):
		"""
		@param width: The width of the window in pixels
		@param height: The height of the window in pixels
		updates the camera and viewport after windowresize
		"""
		self.width = width
		self.height = height
		if self.usingortho:
			glViewport(0,0,width,height)
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			self.setOrthoProjectionMatrix()
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
			glTranslate(0,0,self.zclip)
			self.setCameraPosition()
		else:
			# This may need some work to get it to behave
			glViewport(0,0,width,height)
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			self.setPerspectiveProjectionMatrix()
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
			glTranslate(0,0,self.perspective_z) #How much to set the camera back depends on how big the object is
			self.setCameraPosition()
	
	def setCameraPosition(self, sfactor=1):
		"""
		Set the default camera position
		"""
		glTranslate(0,0,sfactor*self.getZclip())
		
	def setProjectionMatrix(self):
		"""
		Set the projection matrix
		"""
		if self.usingortho:
			self.setOrthoProjectionMatrix()
		else:
			self.setPerspectiveProjectionMatrix()
			
	def setOrthoProjectionMatrix(self):
		"""
		Set the orthographic projection matrix. Volume view origin (0,0) is center of screen
		"""
		glOrtho(-self.width/2, self.width/2, -self.height/2, self.height/2, self.near, self.far)
		
	def setPerspectiveProjectionMatrix(self):
		"""
		Set the perspective projection matrix. Volume view origin (0,0) is center of screen
		"""
		GLU.gluPerspective(self.fovy, (float(self.width)/float(self.height)), self.near, self.far)
			
	def usePrespective(self, boundingbox, screenfraction, fovy=60.0):
		""" 
		@param boundingbox: The dimension of the bounding for the object to be rendered
		@param screenfraction: The fraction of the screen height to occupy
		Changes projection matrix to perspective
		"""
		self.fovy = fovy
		self.perspective_z = -(boundingbox/screenfraction)/(2*math.tan(math.radians(self.fovy/2)))  + boundingbox/2
		self.usingortho = False
		

	def useOrtho(self, zclip):
		"""
		Changes projection matrix to orthographic
		"""
		self.usingortho = True
		self.zclip = zclip

	def setClipFar(self, far):
		"""
		Set the far aspect of the viewing volume
		"""
		self.far = far

	def setClipNear(self, near):
		"""
		Set the near aspect of the viewing volume
		"""
		self.near = near

	def setFovy(self, fovy):
		"""
		Set the field of view angle aspect of the viewing volume
		"""
		self.fovy = fovy
		
	def getHeight(self):
		""" Get the viewport height """
		return self.height
	
	def getWidth(self):
		""" Get the viewport width"""
		return self.width
		
	def getZclip(self):
		""" Get the zclip """
		if self.usingortho:
			return self.zclip
		else:
			return self.perspective_z
	# Maybe other methods to control the camera

###################################### Inspector Code #########################################################################################

class EMInspector3D(QtGui.QWidget):
	def __init__(self, scenegraph):
		"""
		"""
		QtGui.QWidget.__init__(self)
		self.scenegraph = scenegraph
		self.mintreewidth = 200		# minimum width of the tree
		self.mincontrolwidth = 250
		
		vbox = QtGui.QVBoxLayout(self)
		
		self.inspectortab = QtGui.QTabWidget()
		self.inspectortab.addTab(self.getTreeWidget(), "Tree View")
		self.inspectortab.addTab(self.getLightsWidget(), "Lights")
		self.inspectortab.addTab(self.getCameraWidget(), "Camera")
		self.inspectortab.addTab(self.getUtilsWidget(), "Utils")

		vbox.addWidget(self.inspectortab)
		
		self.setLayout(vbox)
		self.updateGeometry()

	def getTreeWidget(self):
		"""
		This returns the treeview-control panel widget
		"""
		widget = QtGui.QWidget()
		hbox = QtGui.QHBoxLayout(widget)
		treesplitter = QtGui.QSplitter(QtCore.Qt.Vertical)
		treesplitter.setFrameShape(QtGui.QFrame.StyledPanel)
		treesplitter.setLayout(self._get_tree_layout(widget))
		treesplitter.setMinimumWidth(self.mintreewidth)
		hbox.addWidget(treesplitter)
		controlsplitter = QtGui.QSplitter(QtCore.Qt.Vertical)
		controlsplitter.setFrameShape(QtGui.QFrame.StyledPanel)
		controlsplitter.setLayout(self._get_controler_layout(widget))
		controlsplitter.setMinimumWidth(self.mincontrolwidth)
		hbox.addWidget(controlsplitter)
		widget.setLayout(hbox)
		
		return widget
		
	def _get_tree_layout(self, parent):
		"""
		Returns the tree layout
		"""
		tvbox = QtGui.QVBoxLayout()
		self.tree_widget = EMQTreeWidget(parent)
		self.tree_widget.setHeaderLabel("Choose a item")
		tvbox.addWidget(self.tree_widget)
		
		QtCore.QObject.connect(self.tree_widget, QtCore.SIGNAL("itemClicked(QTreeWidgetItem*,int)"), self._tree_widget_click)
		QtCore.QObject.connect(self.tree_widget, QtCore.SIGNAL("visibleItem(QTreeWidgetItem*)"), self._tree_widget_visible)
		
		return tvbox
	
	def addTreeNode(self, name, sgnode, parentnode=None):
		"""
		Add a node to the TreeWidget if not parent node, otherwise add a child to parent node
		We need to get a GUI for the treeitem. The treeitem and the GUI need know each other so they can talk
		The Treeitem also needs to know the SGnode, so it can talk to the SGnode.
		You can think of this as a three way conversation(the alterative it to use a meniator, but that is not worth it w/ only three players
		"""
		node = EMQTreeWidgetItem(QtCore.QStringList(name), sgnode)	# Make a QTreeItem widget, and let the TreeItem talk to the scenegraph node and its GUI
		sgnode.setEMQTreeWidgetItem(node)
		sgnodegui = sgnode.getSceneGui()				# Get the SG node GUI controls 
		sgnodegui.setInspector(self)					# Associate the SGGUI with the inspector
		self.stacked_widget.addWidget(sgnodegui)			# Add a widget to the stack
		# Set icon status
		node.setSelectionStateBox()
		# Set parent if one exists	
		if not parentnode:
			self.tree_widget.insertTopLevelItem(0, node)
		else:
			parentnode.addChild(node)
		return node
			
	def _tree_widget_click(self, item, col):
		self.stacked_widget.setCurrentWidget(item.sgnode.getSceneGui())
		item.setSelectionState(item.checkState(0))
		
	def _tree_widget_visible(self, item):
		item.toogleVisibleState()
		self.updateSceneGraph()
		
	def _get_controler_layout(self, parent):
		"""
		Returns the control layout
		"""
		cvbox = QtGui.QVBoxLayout()
		self.stacked_widget = QtGui.QStackedWidget()
		cvbox.addWidget(self.stacked_widget)
		
		return cvbox
		
	def getLightsWidget(self):
		"""
		Returns the lights control widget
		"""
		lwidget = QtGui.QWidget()
		
		return lwidget
		
	def getCameraWidget(self):
		"""
		Returns the camera control widget
		"""
		cwidget = QtGui.QWidget()
		
		return cwidget
	
	def getUtilsWidget(self):
		"""
		Retrusn the utilites widget
		"""
		uwidget = QtGui.QWidget()
		
		return uwidget
		
	def updateSceneGraph(self):
		""" 
		Updates SG, in the near future this will be improved to allow for slow operations
		"""
		self.scenegraph.updateSG()

class EMQTreeWidget(QtGui.QTreeWidget):
	def __init__(self, parent=None):
		QtGui.QTreeWidget.__init__(self, parent)
			
	def mousePressEvent(self, e):
		QtGui.QTreeWidget.mousePressEvent(self, e)
		if e.button()==Qt.RightButton:
			self.emit(QtCore.SIGNAL("visibleItem(QTreeWidgetItem*)"), self.currentItem())
		
		
	
class EMQTreeWidgetItem(QtGui.QTreeWidgetItem):
	"""
	Subclass of QTreeWidgetItem
	adds functionality
	"""
	def __init__(self, qstring, sgnode):
		QtGui.QTreeWidgetItem.__init__(self, qstring)
		self.sgnode = sgnode
		self.setCheckState(0, QtCore.Qt.Unchecked)
		self.visible = QtGui.QIcon(QtGui.QPixmap(visibleicon))
		self.invisible = QtGui.QIcon(QtGui.QPixmap(invisibleicon))
		self.getVisibleState()
	
	def setSelectionState(self, state):
		""" 
		Toogle selection state on and off
		"""
		if state == QtCore.Qt.Checked:
			self.sgnode.is_selected = True
		else:
			self.sgnode.is_selected = False
		self.setSelectionStateBox() # set state of TreeItemwidget
		
	def toogleVisibleState(self):
		self.sgnode.is_visible = not self.sgnode.is_visible
		self.getVisibleState()
		
	def getVisibleState(self):
		"""
		Toogle the visble state
		"""
		if self.sgnode.is_visible:
			self.setIcon(0, self.visible)
		else:
			self.setIcon(0, self.invisible)
	
	def setSelectionStateBox(self):
		"""
		Set the selection state icon
		"""
		if self.sgnode.is_selected:
			self.setCheckState(0, QtCore.Qt.Checked)
		else:
			self.setCheckState(0, QtCore.Qt.Unchecked)
		
class EMInspectorControlBasic(QtGui.QWidget):
	"""
	Class to make the EMItem GUI controls
	"""
	def __init__(self, name, sgnode):
		QtGui.QWidget.__init__(self)
		self.sgnode = sgnode
		self.name = name
		self.inspector = None
		self.transfromboxmaxheight = 400
		
		igvbox = QtGui.QVBoxLayout()
		self.addBasicControls(igvbox)
		self.addColorControls(igvbox)
		self.addControls(igvbox)
		self.setLayout(igvbox)
	
	def setInspector(self, inspector):
		self.inspector = inspector
		
	def addBasicControls(self, igvbox):
		# selection box and label
		font = QtGui.QFont()
		font.setBold(True)
		label = QtGui.QLabel(self.name,self)
		label.setFont(font)
		label.setAlignment(QtCore.Qt.AlignCenter)
		igvbox.addWidget(label)
		databox = QtGui.QHBoxLayout()
		if self.sgnode.boundingboxsize:
			databox.addWidget(QtGui.QLabel("Size: "+str(self.sgnode.boundingboxsize)+u'\u00B3',self))
		igvbox.addLayout(databox)
		# angluar controls
		xformframe = QtGui.QFrame()
		xformframe.setFrameShape(QtGui.QFrame.StyledPanel)
		xformbox = QtGui.QVBoxLayout()
		xformlabel = QtGui.QLabel("Transformation", xformframe)
		xformlabel.setFont(font)
		xformlabel.setAlignment(QtCore.Qt.AlignCenter)
		xformbox.addWidget(xformlabel)
		# Rotations
		self.rotcombobox = QtGui.QComboBox()
		xformbox.addWidget(self.rotcombobox)
		self.rotstackedwidget = QtGui.QStackedWidget()
		self.addRotationWidgets()
		xformbox.addWidget(self.rotstackedwidget)
		#translations
		textbox = QtGui.QHBoxLayout()
		txlabel = QtGui.QLabel("TX",xformframe)
		txlabel.setAlignment(QtCore.Qt.AlignCenter)
		textbox.addWidget(txlabel)
		tylabel = QtGui.QLabel("TY",xformframe)
		tylabel.setAlignment(QtCore.Qt.AlignCenter)
		textbox.addWidget(tylabel)
		xformbox.addLayout(textbox)
		box = QtGui.QHBoxLayout()
		self.tx = EMSpinWidget(0.0, 1.0)
		self.ty = EMSpinWidget(0.0, 1.0)
		box.addWidget(self.tx)
		box.addWidget(self.ty)
		xformbox.addLayout(box)
		zoombox = QtGui.QHBoxLayout()
		tzlabel = QtGui.QLabel("TZ",xformframe)
		tzlabel.setAlignment(QtCore.Qt.AlignCenter)
		zoombox.addWidget(tzlabel)
		zoomlabel = QtGui.QLabel("Zoom",xformframe)
		zoomlabel.setAlignment(QtCore.Qt.AlignCenter)
		zoombox.addWidget(zoomlabel)
		xformbox.addLayout(zoombox)
		zoomwidgetbox = QtGui.QHBoxLayout()
		self.tz = EMSpinWidget(0.0, 1.0)
		self.zoom = EMSpinWidget(0.0, 0.1)
		zoomwidgetbox.addWidget(self.tz)
		zoomwidgetbox.addWidget(self.zoom)
		xformbox.addLayout(zoomwidgetbox)
				
		xformframe.setMaximumHeight(self.transfromboxmaxheight)
		xformframe.setLayout(xformbox)
		igvbox.addWidget(xformframe)
		
		QtCore.QObject.connect(self.tx,QtCore.SIGNAL("valueChanged(int)"),self._on_translation)
		QtCore.QObject.connect(self.ty,QtCore.SIGNAL("valueChanged(int)"),self._on_translation)
		QtCore.QObject.connect(self.tz,QtCore.SIGNAL("valueChanged(int)"),self._on_translation)
		QtCore.QObject.connect(self.zoom,QtCore.SIGNAL("valueChanged(int)"),self._on_scale)
	
	def _on_translation(self, value):
		self.sgnode.transform.set_trans(self.tx.getValue(), self.ty.getValue(), self.tz.getValue())
		self.inspector.updateSceneGraph()
		
	def _on_scale(self, value):
		self.sgnode.transform.set_scale(self.zoom.getValue())
		self.inspector.updateSceneGraph()
		
	def addColorControls(self, igvbox):
		pass
	
	def addControls(self, igvbox):
		pass
	
	def updateInspector(self):
		# Translation update
		translation =  self.sgnode.transform.get_trans()
		self.tx.setValue(translation[0])
		self.ty.setValue(translation[1])
		self.tz.setValue(translation[2])
		# Rotation update
		rotation =  self.sgnode.transform.get_rotation(str(self.rotcombobox.currentText()))
		comboboxidx = self.rotcombobox.currentIndex()
		if comboboxidx == 0:
			self.emanazslider.setValue(rotation["az"], quiet=1)
			self.emanaltslider.setValue(rotation["alt"], quiet=1)
			self.emanphislider.setValue(rotation["phi"], quiet=1)
		if comboboxidx == 1:
			self.imagicgammaslider.setValue(rotation["gamma"], quiet=1)
			self.imagicbetaslider.setValue(rotation["beta"], quiet=1)
			self.imagicalphaslider.setValue(rotation["alpha"], quiet=1)
		if comboboxidx == 2:
			self.spiderpsislider.setValue(rotation["psi"], quiet=1)
			self.spiderthetaslider.setValue(rotation["theta"], quiet=1)
			self.spiderphislider.setValue(rotation["phi"], quiet=1)
		if comboboxidx == 3:
			self.mrcpsislider.setValue(rotation["phi"], quiet=1)
			self.mrcthetaslider.setValue(rotation["theta"], quiet=1)
			self.mrcomegaslider.setValue(rotation["omega"], quiet=1)
		if comboboxidx == 4:
			self.xyzzslider.setValue(rotation["ztilt"], quiet=1)
			self.xyzyslider.setValue(rotation["ytilt"], quiet=1)
			self.xyzxslider.setValue(rotation["xtilt"], quiet=1)
		if comboboxidx == 5:
			self.spinomegaslider .setValue(rotation["Omega"], quiet=1)
			self.spinn1slider.setValue(rotation["n1"], quiet=1)
			self.spinn2slider.setValue(rotation["n2"], quiet=1)
			self.spinn3slider.setValue(rotation["n3"], quiet=1)
		if comboboxidx == 6:
			self.spinomegaslider.setValue(rotation["q"], quiet=1)
			self.sgirotn1slider.setValue(rotation["n1"], quiet=1)
			self.sgirotn2slider.setValue(rotation["n2"], quiet=1)
			self.sgirotn3slider.setValue(rotation["n3"], quiet=1)
		if comboboxidx == 7:
			self.quaternione0slider.setValue(rotation["e0"], quiet=1)
			self.quaternione1slider.setValue(rotation["e1"], quiet=1)
			self.quaternione2slider.setValue(rotation["e2"], quiet=1)
			self.quaternione3slider.setValue(rotation["e3"], quiet=1)
		# Scaling update
		self.zoom.setValue(self.sgnode.transform.get_scale())
		
	def addRotationWidgets(self):
		EMANwidget = QtGui.QWidget()
		Imagicwidget = QtGui.QWidget()
		Spiderwidget = QtGui.QWidget()
		MRCwidget = QtGui.QWidget()
		XYZwidget = QtGui.QWidget()
		spinwidget = QtGui.QWidget()
		sgirotwidget = QtGui.QWidget()
		quaternionwidget = QtGui.QWidget()
		# EMAN
		emanbox = QtGui.QVBoxLayout()
		self.emanazslider = ValSlider(EMANwidget, (0.0, 360.0), "  Az")
		self.emanaltslider = ValSlider(EMANwidget, (0.0, 180.0), "Alt")
		self.emanphislider = ValSlider(EMANwidget, (0.0, 360.0), "Phi")
		emanbox.addWidget(self.emanazslider)
		emanbox.addWidget(self.emanaltslider)
		emanbox.addWidget(self.emanphislider)
		EMANwidget.setLayout(emanbox)
		# Imagic
		imagicbox = QtGui.QVBoxLayout()
		self.imagicgammaslider = ValSlider(Imagicwidget, (0.0, 360.0), "Gamma")
		self.imagicbetaslider = ValSlider(Imagicwidget, (0.0, 180.0), "     Beta")
		self.imagicalphaslider = ValSlider(Imagicwidget, (0.0, 360.0), "   Alpha")
		imagicbox.addWidget(self.imagicgammaslider)
		imagicbox.addWidget(self.imagicbetaslider)
		imagicbox.addWidget(self.imagicalphaslider)
		Imagicwidget.setLayout(imagicbox)
		# Spider
		spiderbox = QtGui.QVBoxLayout()
		self.spiderpsislider = ValSlider(Spiderwidget, (0.0, 360.0), "   Psi")
		self.spiderthetaslider = ValSlider(Spiderwidget, (0.0, 180.0), "Theta")
		self.spiderphislider = ValSlider(Spiderwidget, (0.0, 360.0), "   Phi")
		spiderbox.addWidget(self.spiderpsislider)
		spiderbox.addWidget(self.spiderthetaslider)
		spiderbox.addWidget(self.spiderphislider)
		Spiderwidget.setLayout(spiderbox)
		# MRC
		mrcbox = QtGui.QVBoxLayout()
		self.mrcpsislider = ValSlider(MRCwidget, (0.0, 360.0), "      Psi")
		self.mrcthetaslider = ValSlider(MRCwidget, (0.0, 180.0), "  Theta")
		self.mrcomegaslider = ValSlider(MRCwidget, (0.0, 360.0), "Omega")
		mrcbox.addWidget(self.mrcpsislider)
		mrcbox.addWidget(self.mrcthetaslider)
		mrcbox.addWidget(self.mrcomegaslider)
		MRCwidget.setLayout(mrcbox)
		# XYZ
		xyzbox = QtGui.QVBoxLayout()
		self.xyzzslider = ValSlider(XYZwidget, (0.0, 360.0), "Z")
		self.xyzyslider = ValSlider(XYZwidget, (0.0, 180.0), "Y")
		self.xyzxslider = ValSlider(XYZwidget, (0.0, 360.0), "X")
		xyzbox.addWidget(self.xyzzslider)
		xyzbox.addWidget(self.xyzyslider)
		xyzbox.addWidget(self.xyzxslider)
		XYZwidget.setLayout(xyzbox)
		# spin
		spinbox = QtGui.QVBoxLayout()
		self.spinomegaslider = ValSlider(spinwidget, (0.0, 360.0), "Omega")
		self.spinn1slider = ValSlider(spinwidget, (0.0, 1.0), "       N1")
		self.spinn2slider = ValSlider(spinwidget, (0.0, 1.0), "       N2")
		self.spinn3slider = ValSlider(spinwidget, (0.0, 1.0), "       N3")
		spinbox.addWidget(self.spinomegaslider)
		spinbox.addWidget(self.spinn1slider)
		spinbox.addWidget(self.spinn2slider)
		spinbox.addWidget(self.spinn3slider)
		spinwidget.setLayout(spinbox)
		# sgirot
		sgirotbox = QtGui.QVBoxLayout()
		self.sgirotqslider = ValSlider(sgirotwidget, (0.0, 360.0), " Q")
		self.sgirotn1slider = ValSlider(sgirotwidget, (0.0, 1.0), "N1")
		self.sgirotn2slider = ValSlider(sgirotwidget, (0.0, 1.0), "N2")
		self.sgirotn3slider = ValSlider(sgirotwidget, (0.0, 1.0), "N3")
		sgirotbox.addWidget(self.sgirotqslider)
		sgirotbox.addWidget(self.sgirotn1slider)
		sgirotbox.addWidget(self.sgirotn2slider)
		sgirotbox.addWidget(self.sgirotn3slider)
		sgirotwidget.setLayout(sgirotbox)
		# quaternion
		quaternionbox = QtGui.QVBoxLayout()
		self.quaternione0slider = ValSlider(quaternionwidget, (0.0, 1.0), "E0")
		self.quaternione1slider = ValSlider(quaternionwidget, (0.0, 1.0), "E1")
		self.quaternione2slider = ValSlider(quaternionwidget, (0.0, 1.0), "E2")
		self.quaternione3slider = ValSlider(quaternionwidget, (0.0, 1.0), "E3")
		quaternionbox.addWidget(self.quaternione0slider)
		quaternionbox.addWidget(self.quaternione1slider)
		quaternionbox.addWidget(self.quaternione2slider)
		quaternionbox.addWidget(self.quaternione3slider)
		quaternionwidget.setLayout(quaternionbox)		
		
		# Add widgets to the stack
		self.rotstackedwidget.addWidget(EMANwidget)
		self.rotstackedwidget.addWidget(Imagicwidget)
		self.rotstackedwidget.addWidget(Spiderwidget)
		self.rotstackedwidget.addWidget(MRCwidget)
		self.rotstackedwidget.addWidget(XYZwidget)
		self.rotstackedwidget.addWidget(spinwidget)
		self.rotstackedwidget.addWidget(sgirotwidget)
		self.rotstackedwidget.addWidget(quaternionwidget)
		# add choices to combobox
		self.rotcombobox.addItem("EMAN")
		self.rotcombobox.addItem("Imagic")
		self.rotcombobox.addItem("Spider")
		self.rotcombobox.addItem("MRC")
		self.rotcombobox.addItem("XYZ")
		self.rotcombobox.addItem("spin")
		self.rotcombobox.addItem("sgirot")
		self.rotcombobox.addItem("quaternion")
		
		# Signal for all sliders
		QtCore.QObject.connect(self.rotcombobox, QtCore.SIGNAL("activated(int)"), self._rotcombobox_changed)
		QtCore.QObject.connect(self.emanazslider,QtCore.SIGNAL("valueChanged"),self._on_EMAN_rotation)
		QtCore.QObject.connect(self.emanaltslider,QtCore.SIGNAL("valueChanged"),self._on_EMAN_rotation)
		QtCore.QObject.connect(self.emanphislider,QtCore.SIGNAL("valueChanged"),self._on_EMAN_rotation)
		QtCore.QObject.connect(self.imagicgammaslider,QtCore.SIGNAL("valueChanged"),self._on_Imagic_rotation)
		QtCore.QObject.connect(self.imagicbetaslider,QtCore.SIGNAL("valueChanged"),self._on_Imagic_rotation)
		QtCore.QObject.connect(self.imagicalphaslider,QtCore.SIGNAL("valueChanged"),self._on_Imagic_rotation)
		QtCore.QObject.connect(self.spiderpsislider,QtCore.SIGNAL("valueChanged"),self._on_Spider_rotation)
		QtCore.QObject.connect(self.spiderthetaslider,QtCore.SIGNAL("valueChanged"),self._on_Spider_rotation)
		QtCore.QObject.connect(self.spiderphislider,QtCore.SIGNAL("valueChanged"),self._on_Spider_rotation)
		QtCore.QObject.connect(self.mrcpsislider,QtCore.SIGNAL("valueChanged"),self._on_MRC_rotation)
		QtCore.QObject.connect(self.mrcthetaslider,QtCore.SIGNAL("valueChanged"),self._on_MRC_rotation)
		QtCore.QObject.connect(self.mrcomegaslider,QtCore.SIGNAL("valueChanged"),self._on_MRC_rotation)
		QtCore.QObject.connect(self.xyzzslider,QtCore.SIGNAL("valueChanged"),self._on_XYZ_rotation)
		QtCore.QObject.connect(self.xyzyslider,QtCore.SIGNAL("valueChanged"),self._on_XYZ_rotation)
		QtCore.QObject.connect(self.xyzxslider,QtCore.SIGNAL("valueChanged"),self._on_XYZ_rotation)
		QtCore.QObject.connect(self.spinomegaslider,QtCore.SIGNAL("valueChanged"),self._on_spin_rotation)
		QtCore.QObject.connect(self.spinn1slider,QtCore.SIGNAL("valueChanged"),self._on_spin_rotation)
		QtCore.QObject.connect(self.spinn2slider,QtCore.SIGNAL("valueChanged"),self._on_spin_rotation)
		QtCore.QObject.connect(self.spinn3slider,QtCore.SIGNAL("valueChanged"),self._on_spin_rotation)
		QtCore.QObject.connect(self.sgirotqslider,QtCore.SIGNAL("valueChanged"),self._on_sgirot_rotation)
		QtCore.QObject.connect(self.sgirotn1slider,QtCore.SIGNAL("valueChanged"),self._on_sgirot_rotation)
		QtCore.QObject.connect(self.sgirotn2slider,QtCore.SIGNAL("valueChanged"),self._on_sgirot_rotation)
		QtCore.QObject.connect(self.sgirotn3slider,QtCore.SIGNAL("valueChanged"),self._on_sgirot_rotation)
		QtCore.QObject.connect(self.quaternione0slider,QtCore.SIGNAL("valueChanged"),self._on_quaternion_rotation)
		QtCore.QObject.connect(self.quaternione1slider,QtCore.SIGNAL("valueChanged"),self._on_quaternion_rotation)
		QtCore.QObject.connect(self.quaternione2slider,QtCore.SIGNAL("valueChanged"),self._on_quaternion_rotation)
		QtCore.QObject.connect(self.quaternione3slider,QtCore.SIGNAL("valueChanged"),self._on_quaternion_rotation)
		
	def _rotcombobox_changed(self, idx):
		self.rotstackedwidget.setCurrentIndex(idx)
		self.updateInspector()
		
	def _on_EMAN_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"eman","az":self.emanazslider.getValue(),"alt":self.emanaltslider.getValue(),"phi":self.emanphislider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_Imagic_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"imagic","gamma":self.imagicgammaslider.getValue(),"beta":self.imagicbetaslider.getValue(),"alpha":self.imagicalphaslider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_Spider_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"spider","psi":self.spiderpsislider.getValue(),"theta":self.spiderthetaslider.getValue(),"phi":self.spiderphislider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_MRC_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"mrc","phi":self.mrcpsislider.getValue(),"theta":self.mrcthetaslider.getValue(),"omega":self.mrcomegaslider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_XYZ_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"xyz","ztilt":self.xyzzslider.getValue(),"ytilt":self.xyzyslider.getValue(),"xtilt":self.xyzxslider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_spin_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"spin","Omega":self.spinomegaslider.getValue(),"n1":self.spinn1slider.getValue(),"n2":self.spinn2slider.getValue(),"n3":self.spinn3slider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_sgirot_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"sgirot","q":self.sgirotqslider.getValue(),"n1":self.sgirotn1slider.getValue(),"n2":self.sgirotn2slider.getValue(),"n3":self.sgirotn3slider.getValue()})
		self.inspector.updateSceneGraph()
		
	def _on_quaternion_rotation(self, value):
		self.sgnode.transform.set_rotation({"type":"quaternion","e0":self.quaternione0slider.getValue(),"e1":self.quaternione1slider.getValue(),"e2":self.quaternione2slider.getValue(),"e3":self.quaternione3slider.getValue()})
		self.inspector.updateSceneGraph()
		
class EMInspectorControlShape(EMInspectorControlBasic):
	"""
	Class to make EMItem GUI SHAPE
	"""
	def __init__(self, name, sgnode):
		EMInspectorControlBasic.__init__(self, name, sgnode)
		
	def addControls(self, igvbox):
		pass
		
	def addColorControls(self, box):
		colorframe = QtGui.QFrame()
		colorframe.setFrameShape(QtGui.QFrame.StyledPanel)
		colorvbox = QtGui.QVBoxLayout()
		lfont = QtGui.QFont()
		lfont.setBold(True)
		colorlabel = QtGui.QLabel("Color",colorframe)
		colorlabel.setFont(lfont)
		colorlabel.setAlignment(QtCore.Qt.AlignCenter)

		# These boxes are a pain maybe I should use a Grid?
		cdialoghbox = QtGui.QHBoxLayout()
		cabox = QtGui.QHBoxLayout()
		self.ambcolorbox = EMQTColorWidget(parent=colorframe)
		self.ambcolorbox.setColor(QtGui.QColor(255*self.sgnode.ambient[0],255*self.sgnode.ambient[1],255*self.sgnode.ambient[2]))
		cabox.addWidget(self.ambcolorbox)
		cabox.setAlignment(QtCore.Qt.AlignCenter)
		cdbox = QtGui.QHBoxLayout()
		self.diffusecolorbox = EMQTColorWidget(parent=colorframe)
		self.diffusecolorbox.setColor(QtGui.QColor(255*self.sgnode.diffuse[0],255*self.sgnode.diffuse[1],255*self.sgnode.diffuse[2]))
		cdbox.addWidget(self.diffusecolorbox)
		cdbox.setAlignment(QtCore.Qt.AlignCenter)
		csbox = QtGui.QHBoxLayout()
		self.specularcolorbox = EMQTColorWidget(parent=colorframe)
		self.specularcolorbox.setColor(QtGui.QColor(255*self.sgnode.specular[0],255*self.sgnode.specular[1],255*self.sgnode.specular[2]))
		csbox.addWidget(self.specularcolorbox)
		csbox.setAlignment(QtCore.Qt.AlignCenter)
		cdialoghbox.addLayout(cabox)
		cdialoghbox.addLayout(cdbox)
		cdialoghbox.addLayout(csbox)
		
		colorhbox = QtGui.QHBoxLayout()
		self.ambient = QtGui.QLabel("Ambient", colorframe)
		self.ambient.setAlignment(QtCore.Qt.AlignCenter)
		self.diffuse = QtGui.QLabel("Diffuse", colorframe)
		self.diffuse.setAlignment(QtCore.Qt.AlignCenter)
		self.specular = QtGui.QLabel("Specular", colorframe)
		self.specular.setAlignment(QtCore.Qt.AlignCenter)
		colorhbox.addWidget(self.ambient)
		colorhbox.addWidget(self.diffuse)
		colorhbox.addWidget(self.specular)
		
		self.shininess = ValSlider(colorframe, (0.0, 50.0), "Shininess")
		self.shininess.setValue(self.sgnode.shininess)
		
		colorvbox.addWidget(colorlabel)
		colorvbox.addLayout(cdialoghbox)
		colorvbox.addLayout(colorhbox)
		colorvbox.addWidget(self.shininess)
		colorframe.setLayout(colorvbox)
		box.addWidget(colorframe)
		
		QtCore.QObject.connect(self.ambcolorbox,QtCore.SIGNAL("newcolor(QColor)"),self._on_ambient_color)
		QtCore.QObject.connect(self.diffusecolorbox,QtCore.SIGNAL("newcolor(QColor)"),self._on_diffuse_color)
		QtCore.QObject.connect(self.specularcolorbox,QtCore.SIGNAL("newcolor(QColor)"),self._on_specular_color)
		QtCore.QObject.connect(self.shininess,QtCore.SIGNAL("valueChanged"),self._on_shininess)
		
	def _on_ambient_color(self, color):
		rgb = color.getRgb()
		self.sgnode.setAmbientColor((float(rgb[0])/255.0),(float(rgb[1])/255.0),(float(rgb[2])/255.0))
		self.inspector.updateSceneGraph()
		
	def _on_diffuse_color(self, color):
		rgb = color.getRgb()
		self.sgnode.setDiffuseColor((float(rgb[0])/255.0),(float(rgb[1])/255.0),(float(rgb[2])/255.0))
		self.inspector.updateSceneGraph()
		
	def _on_specular_color(self, color):
		rgb = color.getRgb()
		self.sgnode.setSpecularColor((float(rgb[0])/255.0),(float(rgb[1])/255.0),(float(rgb[2])/255.0))
		self.inspector.updateSceneGraph()
		
	def _on_shininess(self, shininess):
		self.sgnode.setShininess(shininess)
		self.inspector.updateSceneGraph()
		
###################################### TEST CODE, THIS WILL NOT APPEAR IN THE WIDGET3D MODULE ##################################################
		
# All object that are rendered inherit from abstractSGnode and implement the render method
# In this example I use a cube, but any object can be drawn and so long as the object class inherits from abstractSGnode
class glCube(EMItem3D):
	def __init__(self, size):
		EMItem3D.__init__(self, parent=None, transform=Transform())
		# size
		self.boundingboxsize = size
		self.xi = -size/2
		self.yi = -size/2
		self.zi = -size/2
		self.xf = size/2
		self.yf = size/2
		self.zf = size/2
		
		# color
		self.diffuse = [1.0,1.0,1.0,1.0]
		self.specular = [1.0,1.0,1.0,1.0]
		self.ambient = [1.0, 1.0, 1.0, 1.0]
		self.shininess = 25.0
		
		# GUI contols
		self.widget = None
	
	def setAmbientColor(self, red, green, blue, alpha=1.0):
		self.ambient = [red, green, blue, alpha]

	def setDiffuseColor(self, red, green, blue, alpha=1.0):
		self.diffuse = [red, green, blue, alpha]
		
	def setSpecularColor(self, red, green, blue, alpha=1.0):
		self.specular = [red, green, blue, alpha]
	
	def setShininess(self, shininess):
		self.shininess = shininess
		
	def getSceneGui(self):
		"""
		Return a Qt widget that controls the scene item
		"""
		if not self.widget: self.widget = EMInspectorControlShape("CUBE", self)
		return self.widget
		
	def renderNode(self):
			
		# Material properties of the box
		glMaterialfv(GL_FRONT, GL_DIFFUSE, self.diffuse)
		glMaterialfv(GL_FRONT, GL_SPECULAR, self.specular)
		glMaterialf(GL_FRONT, GL_SHININESS, self.shininess)
		glMaterialfv(GL_FRONT, GL_AMBIENT, self.ambient)
		# The box itself anlong with normal vectors
		
		glBegin(GL_QUADS)
		glNormal3f(self.xi, self.yi, self.zi + 1)
		glVertex3f(self.xi, self.yi, self.zi)
		glNormal3f(self.xf, self.yi, self.zi + 1)
		glVertex3f(self.xf, self.yi, self.zi)
		glNormal3f(self.xf, self.yf, self.zi + 1)
		glVertex3f(self.xf, self.yf, self.zi)
		glNormal3f(self.xi, self.yf, self.zi + 1)
		glVertex3f(self.xi, self.yf, self.zi)

		glNormal3f(self.xi - 1, self.yi, self.zi)
		glVertex3f(self.xi, self.yi, self.zi)
		glNormal3f(self.xi - 1, self.yi, self.zf)
		glVertex3f(self.xi, self.yi, self.zf)
		glNormal3f(self.xi - 1, self.yf, self.zf)
		glVertex3f(self.xi, self.yf, self.zf)
		glNormal3f(self.xi - 1, self.yf, self.zi)
		glVertex3f(self.xi, self.yf, self.zi)

		glNormal3f(self.xi, self.yi, self.zf - 1)
		glVertex3f(self.xi, self.yi, self.zf)
		glNormal3f(self.xi, self.yf, self.zf - 1)
		glVertex3f(self.xi, self.yf, self.zf)
		glNormal3f(self.xf, self.yf, self.zf - 1)
		glVertex3f(self.xf, self.yf, self.zf)
		glNormal3f(self.xf, self.yi, self.zf - 1)
		glVertex3f(self.xf, self.yi, self.zf)

		glNormal3f(self.xf + 1, self.yf, self.zf)
		glVertex3f(self.xf, self.yf, self.zf)
		glNormal3f(self.xf + 1, self.yf, self.zi)
		glVertex3f(self.xf, self.yf, self.zi)
		glNormal3f(self.xf + 1, self.yi, self.zi)
		glVertex3f(self.xf, self.yi, self.zi)
		glNormal3f(self.xf + 1, self.yi, self.zf)
		glVertex3f(self.xf, self.yi, self.zf)

		glNormal3f(self.xi, self.yf + 1, self.zi)
		glVertex3f(self.xi, self.yf, self.zi)
		glNormal3f(self.xi, self.yf + 1, self.zf)
		glVertex3f(self.xi, self.yf, self.zf)
		glNormal3f(self.xf, self.yf + 1, self.zf)
		glVertex3f(self.xf, self.yf, self.zf)
		glNormal3f(self.xf, self.yf + 1, self.zi)
		glVertex3f(self.xf, self.yf, self.zi)

		glNormal3f(self.xi, self.yi - 1, self.zi)
		glVertex3f(self.xi, self.yi, self.zi)
		glNormal3f(self.xi, self.yi - 1, self.zf)
		glVertex3f(self.xi, self.yi, self.zf)
		glNormal3f(self.xf, self.yi - 1, self.zf)
		glVertex3f(self.xf, self.yi, self.zf)
		glNormal3f(self.xf, self.yi - 1, self.zi)
		glVertex3f(self.xf, self.yi, self.zi)

		glEnd()

		
class GLdemo(QtGui.QWidget):
	def __init__(self):
		QtGui.QWidget.__init__(self)
		self.widget = EMScene3D()
		#self.widget.camera.usePrespective(50, 0.5)
		self.cube1 = glCube(50.0)
		self.widget.addChild(self.cube1)    # Something to Render something..... (this could just as well be one of Ross's SGnodes)
		#self.widget.activatenode(cube1)
		self.cube2 = glCube(50.0)
		self.widget.addChild(self.cube2)
		#self.widget.activatenode(cube2)

		self.inspector = EMInspector3D(self.widget)
		self.widget.setInspector(self.inspector)
		
		rootnode = self.inspector.addTreeNode("root node", self.widget)
		self.inspector.addTreeNode("cube1", self.cube1, rootnode)
		self.inspector.addTreeNode("cube2", self.cube2, rootnode)
		
		# QT stuff to display the widget
		vbox = QtGui.QVBoxLayout()
		vbox.addWidget(self.widget)
		self.setLayout(vbox)
		self.setGeometry(300, 300, 600, 600)
		self.setWindowTitle('BCM EM Viewer')
	
	def show_inspector(self):
		self.inspector.show()
		
if __name__ == "__main__":
	import sys
	app = QtGui.QApplication(sys.argv)
	window = GLdemo()
	window.show()
	window.show_inspector()
	app.exec_()

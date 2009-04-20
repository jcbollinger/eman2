#!/usr/bin/env python

#
# Author: Steven Ludtke, 03/06/2009 (sludtke@bcm.edu)
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

# e2parallel.py Steven Ludtke
# This program implements, via various options, the parallelism system for EMAN2

from EMAN2 import *
from EMAN2PAR import *
from optparse import OptionParser
from math import *
import time
import os
import sys
import socket

debug=False
logid=None

def main():
	global debug,logid
	progname = os.path.basename(sys.argv[0])
	commandlist=("dcserver","dcclient","dckill")
	usage = """%prog [options] <command> ...
	
This program implements much of EMAN2's coarse-grained parallelism mechanism. There are several flavors available via
different options in this program. The simplest, and easiest to use is probably the client/server Distriuted Computing system.

<command> is one of: dcserver, dcclient, dckill

client-server DC system:
run e2parallel.py dcserver on the machine containing your data and project files
run e2parallel.py dcclient on as many other machines as possible, pointing at the server machine

"""

	parser = OptionParser(usage=usage,version=EMANVERSION)

#	parser.add_option("--bgmask",type="int",help="Compute the background power spectrum from the edge of the image, specify a mask radius in pixels which would largely mask out the particles. Default is boxsize/2.",default=0)
#	parser.add_option("--apix",type="float",help="Angstroms per pixel for all images",default=0)
	parser.add_option("--server",type="string",help="Specifies host of the server to connect to",default="localhost")
	parser.add_option("--port",type="int",help="Specifies server port, default is automatic assignment",default=-1)
	parser.add_option("--verbose",type="int",help="debugging level (0-9) default=0)",default=0)
	parser.add_option("--cpus",type="int",help="Number of CPUs/Cores for the clients to use on the local machine")
	parser.add_option("--idleonly",action="store_true",help="Will only use CPUs on the local machine when idle",default=False)
	
	(options, args) = parser.parse_args()
	if len(args)<1 or args[0] not in commandlist: parser.error("command required: "+str(commandlist))

	if args[0]=="dcserver" :
		rundcserver(options.port,options.verbose)
		
	elif args[0]=="dcclient" :
		rundcclient(options.server,options.port,options.verbose)
		
	elif args[0]=="dckill" :
		killdcserver(options.server,options.port,options.verbose)
	
def rundcserver(port,verbose):
	"""Launches a DCServer. If port is <1 or None, will autodetermine. Does not return."""

	server=runEMDCServer(port,verbose)			# never returns


def rundcclient(host,port,verbose):
	"""Starts a DC client running, runs forever"""
	client=EMDCTaskClient(host,port,verbose)
	client.run()

def killdcserver(server,port,verbose):
	EMDCsendonecom(server,port,"QUIT",None)

if __name__== "__main__":
	main()
	

	
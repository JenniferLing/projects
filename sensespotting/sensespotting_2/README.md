#################################
								#
	SENSESPOTTING 2.0			#
								#
#################################

SenseSpotting has the following folder structure.
We define a working dir (e.g. /home/x/xx/sensespotting). 
In this directory, the tool will create all folders. 
For bigger files, an additional working directory can be defined to keep the main (small) content in one directory, 
but all models and corpus files are saved in the big file directory.

FIRST STEPS:

1. Define paths in install_all.sh
2. Installation of external tools: sh install_all.sh
3. Finish installation of SRILM manually:
	- Go to the folder tools/srilm
	- Open Makefile, define the SRILM variable: change "# SRILM = /home/speech/stolcke/project/srilm/devel" to "SRILM = /home/x/xx/sensespotting/tools/srilm" and close the file
	- run in the console (in tools/srilm): make World 
	- check that a directory i686, i686-m64 or similar was created in tools/srilm/bin (Note: the name of the directory in bin differs from system to system)
	- run in the console: export PATH=/home/x/xx/sensespotting/tools/srilm/bin:/home/x/xxx/sensespotting/tools/srilm/bin/i686-m64:$PATH" (adapt the name of the directory in bin)
	- run in the console: make test
	- run in the console: make cleanest

4. Prepare configuration file.
5. Start sensespotting with: python3 sensespotting.py -c <path/to/config/file> -v <verbosity> -u <use_existing>
	- verbosity and use_existing are OPTIONAL and have a default value of 1
	- IF verbosity = 1 => print log statements; else print only configuration and final performance
	- IF use_existing = 1 => if models are found in the default folders or are provided in the config file, no new models are created, but the existing ones are used
	

NOTE:
Training of the PSD classifier can need a long time! 
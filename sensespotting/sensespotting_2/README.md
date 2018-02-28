#################################
								#
	SENSESPOTTING 2.0			#
								#
#################################

SenseSpotting has the following folder structure.
We define a working dir (e.g. /home/x/xx/sensespotting). 
In this directory, the tool will create all folders. 
For bigger files, an additional working directory can be defined to keep the main (small) content in one directory, 
but save all models and corpus files in the big file directory.

FIRST STEPS:
1. Download srilm.tar.gz from http://www.speech.sri.com/projects/srilm/download.html and put the zipped file in the *installation_scripts* directory
2. Open *install_all.sh* in edit mode and define *srilm_path*, *script_path*, *working_dir* and *tools_path*
3. a. Installation of all external tools (SRILM, TreeTagger, fast_align, GIZA++, mgiza, mosesdecoder, vowpal wabbit): 
	> sh install_all.sh  

OR

3. b. If you already have some of these tools installed, go to *installation_scripts* and run the scripts of the needed tools; e.g.
	> sh install_fast_align.sh
4. If SRILM was installed, finish its installation manually:
	- Go to the folder tools/srilm
	- Open Makefile, define the SRILM variable: change "# SRILM = /home/speech/stolcke/project/srilm/devel" to "SRILM = /home/x/xx/sensespotting/tools/srilm" and close the file
	- run in the console (in tools/srilm): 
		> make World 
	- check that a directory i686, i686-m64 or similar was created in tools/srilm/bin (Note: the name of the directory in bin differs from system to system)
	- run in the console (but adapt *export* command to the name of the directory in bin): 
		> export PATH=/home/x/xx/sensespotting/tools/srilm/bin:/home/x/xxx/sensespotting/tools/srilm/bin/i686-m64:$PATH" (
		> make test
		> make cleanest

5. Prepare hansards dataset: unzip all files and concatenate them
	> cd sensespotting_2/corpus/hansards
	> unzip train\*.\*.zip
	> cat train1.fr train2.fr train3.fr > train.fr
	> cat train1.en train2.en train3.en > train.en
	
6. Prepare configuration file (see instructions in *config.txt*).
7. Start sensespotting
	> python3 run_sensespotting.py -c <path/to/config/file> -v <verbosity> -u <use_existing>
	- verbosity and use_existing are OPTIONAL and have a default value of 1
	- IF verbosity = 1 => print log statements; else print only configuration and final performance
	- IF use_existing = 1 => if models are found in the default folders or are provided in the config file, no new models are created, but the existing ones are used
	

**NOTE:**
Training of the PSD classifier may need a long time! 

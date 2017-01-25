"""
TwitIETagger usable with Python; calls jar.
"""
import subprocess
import codecs
import os
import psutil

"""
Since running a file is much more efficient 
than tagging each tweet alone,
write (tokenized) tweets (each per line) to input file 
and return tagged tweets in output file. 
"""
def runFile(input_file, output_file):
    p = subprocess.Popen('java -XX:ParallelGCThreads=2 -Xmx1000m -jar twitie_tag.jar models/gate-EN-twitter.model "' + input_file + '"',stdout=subprocess.PIPE)
    
    o = codecs.open(output_file,'w',encoding='utf-8')
    while p.poll() is None:
        l = p.stdout.readline()
        l = l.decode('cp1252')
        o.write(l)
        o.flush()
    
    o.close()

"""
Tagging of a tokenized string.
@param tokenized string to be tagged
"""
def runString(string_to_be_tagged):
    file_name = 'temp_file.txt'
    o = codecs.open(file_name,'w',encoding='utf-8')
    uniS = string_to_be_tagged.decode('utf-8')
    o.write(uniS)
    o.close()
    l = ''
    p = subprocess.Popen('java -XX:ParallelGCThreads=2 -Xmx500m -jar twitie_tag.jar models/gate-EN-twitter.model ' + file_name,stdout=subprocess.PIPE)

    while p.poll() is None:
        l = p.stdout.readline()
        break

    p.kill()
    psutil.pids()
    
    os.remove(file_name)
    return l
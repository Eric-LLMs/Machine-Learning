import logging
import os,sys 

def SetLoger(logDir,logName) :
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename = os.path.join(logDir,logName),
                        filemode='w')
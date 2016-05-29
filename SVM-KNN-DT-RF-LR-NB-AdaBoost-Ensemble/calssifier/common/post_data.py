import os,sys
import numpy

def output_pre_data(path,data):
    with open(path, "w") as f:
        label = ''
        for value in data:
           if value == 0:
              label = 'negative'
           else :
               label = 'positive'
           f.write(label+'\n')
    f.close()

def output_data(path,data):
    with open(path, "w") as f:
        for value in data:
           f.write(value)
    f.close()

def output_data_array(path,data):
    numpy.save(path,data)
import random 
import math
def generat_randow(min,max,percert)   :
  result=[] 
  gap = math.ceil(max/((max-min)*percert))
  result = range(min,max,int(gap))
  return result 
 
def output_data(path,data):
    with open(path, "w") as f:
         for str in data:
             f.write(str+'\n')
    f.close()

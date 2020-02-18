# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:02:43 2020

@author: BRENDA
"""

import os
# Function to rename multiple files
def main():
   i = 0
   path='C:/Users/BRENDA/Desktop/THESIS\Dataz/Informal/'
   for filename in os.listdir(path):
      my_dest ="Informal." + str(i) + ".tif"
      my_source =path + filename
      my_dest =path + my_dest
      # rename() function will
      # rename all the files
      os.rename(my_source, my_dest)
      i += 1
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()
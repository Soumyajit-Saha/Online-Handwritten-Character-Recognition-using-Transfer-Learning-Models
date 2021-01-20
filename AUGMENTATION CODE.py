#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import ndimage, misc
import numpy as np
import os
import cv2
import random



def main():
    for i in range(1,51):
        f=[]
        onlyfiles = [f for f in os.listdir ("C:/Users/Soumyajit Saha/Dropbox/TRANSFER LEARNING/final/final/Train" +  "/" + str(i))]
        print(onlyfiles)
        t=0
        for j in onlyfiles:
            #path1 = onlyfiles[t] 
            # iterate through the names of contents of the folder
            #print(path1)
            #t=t+1

            # create the full input path and read the file
        
            input_path = "C:/Users/Soumyajit Saha/Dropbox/TRANSFER LEARNING/final/final/Train" +  "/" + str(i) + "/" + str(j)
            print(input_path)
            image_to_rotate = cv2.imread(input_path)
            print(image_to_rotate)

            # rotate the image
            
            
            val=random.randint(-60,60)
            rotated = ndimage.rotate(image_to_rotate, val)

                # create full output path, 'example.jpg' 
                # becomes 'rotate_example.jpg', save the file to disk
            out="D:/Project/Transfer learning/Bangla/Rotated_imgs/" + str(i) + "/" + str(j) 
            cv2.imwrite(out, rotated)           

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





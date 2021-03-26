##K-Means clustering
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image
##Loading image file
img=Image.open('4.2.07.tiff')
X=np.array(img)
#Converting 3D to 2D data
S=[]
for i in range(len(X)):
    for j in range(len(X[0])):
        S.append(X[i][j])


#Number of clusters 
n =8
#Implementing K-Means algorithm 
clf =  KMeans(n_clusters = n)
clf.fit(S)
centers= clf.cluster_centers_

labels=clf.predict(S)


cen = np.zeros((len(centers),len(centers[0])),dtype=int)
for i in range(len(centers)):
  for j in range(len(centers[0])):
    cen[i][j]= int(centers[i][j])

Y=[]

for i in range(len(labels)):
  for j in range(0,n):
    if(labels[i]==j):
      Y.append(cen[j])
  
##Reshaping to original image size
Y1 = np.reshape(Y,(512,512,3))
#Plotting original and clustered image 

plt.imshow(X)
plt.title('Original Image')
plt.show()


plt.title('Clustered Image for n=8')
plt.imshow(Y1)
plt.show()

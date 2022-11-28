import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier






imagen_cv=cv2.imread('input.jpg')

print(imagen_cv.shape)
imagen=imagen_cv.copy()
imagen[:,:,[2,1,0]]=imagen_cv #cv2 lee en orden BGR'
print('dimensiones de la imagen: ', imagen.shape, 'es decir: ', imagen.shape[0]*imagen.shape[1], '=', imagen.shape[0],'x',imagen.shape[1],' pixeles RGB')
print('tipo de datos: ', imagen.dtype)

cv2.imshow('Window',imagen_cv)
cv2.waitKey(0)

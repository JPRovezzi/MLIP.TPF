#%%
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

# %%
imagen_cv=cv2.imread('input.jpg')
imagen_cv=imagen_cv[15:,15:,:]
print(imagen_cv.shape)
imagen=imagen_cv.copy()
imagen[:,:,[2,1,0]]=imagen_cv #cv2 lee en orden BGR'
print('dimensiones de la imagen: ', imagen.shape, 'es decir: ', imagen.shape[0]*imagen.shape[1], '=', imagen.shape[0],'x',imagen.shape[1],' pixeles RGB')
print('tipo de datos: ', imagen.dtype)
#%%
cv2.startWindowThread()
#cv2.namedWindow("preview")
cv2.imshow('ImageWindow',imagen_cv)
cv2.waitKey()
#%%
dims=imagen.shape
cant_filas=dims[0]
cant_columnas=dims[1]
cant_pixeles=cant_filas*cant_columnas
data0=np.zeros([cant_pixeles,4],dtype='int')
for f in range(cant_filas):
    for c in range(cant_columnas):
        nro=f*cant_columnas+c
        data0[nro,0]=nro
        data0[nro,1:4]=imagen[f,c,:]
        
df = pd.DataFrame(data0, columns=["pixel_id","rojo","verde","azul"])

print('Tamaño de Base de pixeles:', df.shape)
#df.head()

variables=["rojo","verde","azul"]# variables o características

variable1=variables[1] # puede ser 0, 1 o 2, es decir, rojo, verde o azul
variable2=variables[2]

f1 = df[variable1].values
f2 = df[variable2].values

plt.scatter(f1, f2,c='gray', s=1)
plt.xlabel(variable1)
plt.ylabel(variable2)
plt.show()
# %%
import plotly.express as px

fig = px.scatter(df, y=variable2, x=variable1, text="pixel_id")
fig.update_traces(marker_size=2)
#fig.show() #en Jupyter
fig.show(renderer="colab") #En Colab



# %%

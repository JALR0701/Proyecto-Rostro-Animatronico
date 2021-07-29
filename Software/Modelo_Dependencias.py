#Autor:         Jorge Lorenzana - 17302
#Nombre:        Modelo de reconocimiento de emociones
#Descripcion:   Codigo base por Karan Sethi modificado
#               para el reconocimiento de 7 emociones
#Fecha:         26/07/2021

#################### Librerias a usar ####################
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os
##########################################################

num_classes=7           #Cantidad de emociones
img_rows,img_cols=48,48 #Tamaño del array de la imagen
batch_size=32           #Numero de muestras procesadas antes de actualizar el modelo

### Cargamos la direccion del dataset de imagenes en variables ###
train_data_dir='Dataset/train'
validation_data_dir='Dataset/validation'
##################################################################

# Expandimos el dataset de manera artificial haciendo modificaciones #
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
######################################################################

### Estandarizamos las cualidades de las imagenes del dataset ###
print("Training Images:")
train_generator = train_datagen.flow_from_directory(
 train_data_dir,                                        #Direccion del data set
 color_mode='grayscale',                                #Escala de grises ya que no importa el color solo las expresiones
 target_size=(img_rows,img_cols),                       #Convertimos las imagenes en un tamaño uniforme
 batch_size=batch_size,                                 #Muestras de entrenamiento
 class_mode='categorical',                              #Usamos categorica porque lo queremos por clases
 shuffle=True)                                          #Con el Shuffle encendido se tienen mejores resultados

print("\nValidation Images:")
validation_generator = validation_datagen.flow_from_directory(
 validation_data_dir,                                   #Direccion del data set
 color_mode='grayscale',                                #Escala de grises ya que no importa el color solo las expresiones
 target_size=(img_rows,img_cols),                       #Convertimos las imagenes en un tamaño uniforme
 batch_size=batch_size,                                 #Muestras de entrenamiento
 class_mode='categorical',                              #Usamos categorica porque lo queremos por clases
 shuffle=True)                                          #Con el Shuffle encendido se tienen mejores resultados
#################################################################

model = Sequential()    #Usamos modelo secuencial para que los bloques se ejecuten uno despues de otro

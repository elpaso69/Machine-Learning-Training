# Keras use images that is very special
# datasets can be found on Kaggle 
# Fetaure scaling is compulsory for computer vision 

'Part 1 - Building the CNN (no need to preprocess)'
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense 

# initialisng the CNN 
classifier = Sequential()

# Convolution Step
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation ='relu')) #Using Tensorflow Backend it's (dimensions, no. channels = 3) for Theano it's the opposite 

# Pooling Step 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#adding another convolution layer 
classifier.add(Convolution2D(32, 3, 3, activation ='relu')) #you remove input layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening step 
classifier.add(Flatten())

# Fully connected layer 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compling the CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

' Part 2 - Fitting the CNN to the images ' 
# read the Keras Doucmentation. Look at image preprocessing 
# Overfitting happen if we have small dataset. Data augmentation help prevent overfitting 
# Goes through the images in batches so there is less overfitting. This is image augementation 

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/training_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')        

classifier.fit_generator(training_set,
                        steps_per_epoch= 8000,
                        epochs=10,
                        validation_data= test_set,
                        nb_val_samples= 2000)
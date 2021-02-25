from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import os

batch_size = 32
img_size = 299
n_images_test = 33000
modelname = ''
dirTest = ''

if not os.path.exists('./../Models/StaticFullFov.h5'):
    print("please download and extract models first")

def test():
    test_generator = train_datagen.flow_from_directory(
        dirTest,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')
    model=load_model(modelname)
    result=model.evaluate_generator(
        test_generator,
        max_queue_size=2,
        steps = n_images_test // batch_size,
        verbose = 2
        )
    print(result)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

# static full fov
modelname = './../Models/StaticFullFov.h5'
dirTest='./../Dataset/TestSetStaticFull/imgNormal'
test()

#static crop fov
modelname = './../Models/StaticCropFov.h5'
dirTest='./../Dataset/TestSetStaticCropped/imgNormal'
test()


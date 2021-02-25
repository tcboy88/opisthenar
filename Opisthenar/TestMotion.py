from tweaked_ImageGenerator_v2 import ImageDataGenerator
from tensorflow.python.keras.models import load_model

batch_size = 32
img_size = 299
n_images_test = 18000
channel = 1
dir1=''
dir2=''
model_name=''

def generate_generator_multiple(generator, dir1, dir2, color_mode, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=True,
                                          color_mode=color_mode,
                                          seed=7,
                                          frames_per_step=1,
                                          )

    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=True,
                                          color_mode=color_mode,
                                          seed=7,
                                          frames_per_step=1
                                          )
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label



def test():
    testgenerator = generate_generator_multiple(generator=train_datagen,
                                                dir1=dir1,
                                                dir2=dir2,
                                                batch_size=batch_size,
                                                img_height=img_size,
                                                color_mode='grayscale',
                                                img_width=img_size,
                                                )

    model = load_model(model_name)

    result = model.evaluate_generator(
        testgenerator,
        max_queue_size=1,
        steps= n_images_test // batch_size,
        verbose=2
    )
    print(result)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

dir1 = "./../Dataset/TestSetMotionFull/imgNormal"
dir2 = "./../Dataset/TestSetMotionFull/imgMotion"
model_name='./../Models/MotionFullFov.h5' 
test()

dir1="./../Dataset/TestSetMotionCropped/imgNormal"
dir2="./../Dataset/TestSetMotionCropped/imgMotion"
model_name='./../Models/MotionCropFov.h5' 
test()

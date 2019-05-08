from keras.preprocessing.image import ImageDataGenerator


TRAIN_DIR = 'example/train'
TEST_DIR = 'example/test'
VAL_DIR = 'example/val'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(200, 200),
        batch_size=2,
        class_mode='binary'
    )

validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(200, 200),
        batch_size=2,
        class_mode='binary'
    )
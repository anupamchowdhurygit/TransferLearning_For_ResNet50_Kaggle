from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling1D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

num_classes = 2
resnet_weights_path = '/Users/anupamchowdhury/PycharmProjects/TransferLearning_For_ResNet50_Kaggle/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_model = Sequential()
my_model.add(ResNet50(include_top =False, pooling = 'avg', weights = resnet_weights_path))
my_model.add(Dense(num_classes, activation='softmax'))

my_model.layers[0].trainable = False

my_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics =['accuracy'])

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory('/Users/anupamchowdhury/PycharmProjects/TransferLearning_For_ResNet50_Kaggle/data/train',
                                                     target_size=(image_size, image_size),
                                                     batch_size=24,
                                                     class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    '/Users/anupamchowdhury/PycharmProjects/TransferLearning_For_ResNet50_Kaggle/data/val',
                                                     target_size=(image_size, image_size),
                                                     class_mode='categorical')

my_model.fit_generator(train_generator,
                       steps_per_epoch=3,
                       validation_data=validation_generator,
                       validation_steps=1)
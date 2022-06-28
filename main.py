import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from codecarbon import EmissionsTracker
base_dir = '/Users/furkangulkan/PycharmProjects/arastirma/minimias'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'gogus')
# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'pektoral')
# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'gogus')
# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'pektoral')
# Image Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Training and Validation Sets
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 8, class_mode = 'binary', target_size = (224, 224))
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 8, class_mode = 'binary', target_size = (224, 224))
# Dataset has been set

# Xception
print("Xception State")

base_model = tf.keras.applications.xception.Xception(
    input_shape = (224, 224, 3),
    include_top=False,
    weights = None
    #weights= 'imagenet'
)

for layer in base_model.layers:
    layer.trainable = False
# Compile and Fit
    # Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.5
x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001,momentum=0.8, nesterov=True),loss='binary_crossentropy', metrics=['acc'])
# Saving the models weight to reset before each step for 1-10-50 epoch
print("Xception is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
xception_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 10, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Xception_10epoch")

# MobileNetv2
print("MobileNetv2 State")
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top=False,
    weights=None,
    #weights = 'imagenet'
)

for layer in base_model.layers:
    layer.trainable = False
# Compile and Fit
    # Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.5
x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001,momentum=0.8, nesterov=True),loss='binary_crossentropy', metrics=['acc'])
# Fitting the model
print("MobileNetv2 is being run for 10 epoch")
model.save("MobileNetv2_10epoch")
tracker = EmissionsTracker()
tracker.start()
mobilenetv2_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 10, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")

### 1.Very Deep Convolutional Networks for Large-Scale Image Recognition(VGG-16)
print("VGG16 State")
# Loading the Base Model
from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = None)#"imagenet"

for layer in base_model.layers:
    layer.trainable = False
# Compile and Fit
    # Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001,momentum=0.8, nesterov=True),loss='binary_crossentropy', metrics=['acc'])
# Fitting the model
print("VGG16 is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
vgg_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 10, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("vgg16_10epoch")


### 2.Inception
print("Inception State")
# Loading the Base Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (224, 224, 3), include_top = False, weights = None)

# Compile and Fit
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001,momentum=0.8, nesterov=True),loss='binary_crossentropy', metrics=['acc'])
# Fitting the model
print("Inceptionv3 is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
inc_history10 = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch = 10, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Inceptionv3_10epoch")


### 3. ResNet50v2
print("ResNet50v2 State")
# Import the base model
from tensorflow.keras.applications import ResNet50V2
base_model = ResNet50V2(input_shape=(224, 224,3), include_top=False, weights=None)

for layer in base_model.layers:
    layer.trainable = False

# Build and Compile the Model
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001,momentum=0.8, nesterov=True),loss='binary_crossentropy', metrics=['acc'])

# Fitting the model
print("Resnet50v2 is being run for 10 epoch")
tracker = EmissionsTracker()
tracker.start()
resnet_history10 = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 10)
emissions: float = tracker.stop()
print("%.4f" % (emissions*1000) + " gram")
model.save("Resnet50v2_10epoch")

import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def extract_color_histogram_tf(image, bins):
    image = tf.image.rgb_to_hsv(image)      # image values are in [0, 1]

    histogram = []
    for i in range(3):  
        image_channel = image[..., i]   # extract i-th channel
        image_channel = tf.reshape(image_channel, [-1])  # obtain 1D vector

        h = tf.histogram_fixed_width(image_channel, [0.0, 1.0], nbins = bins)
        h = tf.cast(h, tf.float32)

        # normalization
        sum = tf.reduce_sum(h)
        h = h / sum

        histogram.append(h)
    
    return tf.concat(histogram, axis = 0)  #  the shape of this is 3 * 16 = 48

def load_images(labels_csv, images_dir, img_size = (100, 100), batch_size = 64, shuffle = False):
    def parse(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels = 3)
        image = tf.image.resize(image, img_size)

        # normalize
        image = tf.cast(image, tf.float32) / 255.0

        # obtain histogram
        histogram = extract_color_histogram_tf(image, bins = 16)

        return (image, histogram), label

    # read img names and labels
    df = pd.read_csv(labels_csv)
    img_names = df.iloc[:, 0].values
    img_paths = [os.path.join(images_dir, img_name + ".png") for img_name in img_names]
    labels = df.iloc[:, 1].astype(int).values

    data = tf.data.Dataset.from_tensor_slices((img_paths, labels))  # zip img with labels
    data = data.map(parse)

    # shuffle
    if shuffle:
        data = data.shuffle(buffer_size = 25000)

    data = data.batch(batch_size)
    
    return data

def cnn_model(input_shape = (100, 100, 3), hist_shape = (48,), num_classes = 5):
    # image input
    image_input = layers.Input(shape = input_shape)

    # convolution BLOCK 1
    x = layers.Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling2D(2)(x)
    x_block1 = x        # save the output for residual later on

    # convolution BLOCK 2
    x = layers.Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    # convolution BLOCK 3
    x = layers.Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D(2)(x)

    # process residual block 1
    res_block1 = layers.Conv2D(128, 1, padding = 'same', kernel_initializer = 'he_normal')(x_block1)
    res_block1 = layers.BatchNormalization()(res_block1)
    res_block1 = layers.LeakyReLU(0.1)(res_block1)
    res_block1 = layers.MaxPooling2D(4)(res_block1)

    # add residual
    x = layers.Add()([x, res_block1])
    x_block3 = x        # save the output for residual later on

    # convolution BLOCK 4
    x = layers.Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)

    # convolution BLOCK 5
    x = layers.Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)

    # process residual block 3
    res_block3 = layers.Conv2D(512, 1, padding = 'same', kernel_initializer = 'he_normal')(x_block3)
    res_block3 = layers.BatchNormalization()(res_block3)
    res_block3 = layers.LeakyReLU(0.1)(res_block3)
    x = layers.Add()([x, res_block3])

    x = layers.GlobalAveragePooling2D()(x)

    # histogram input
    hist_input = layers.Input(shape = hist_shape)
    h = layers.Dense(64, activation = 'relu')(hist_input)

    # merge conv output with histogram
    x = layers.Concatenate()([x, h])

    # dense neural network
    x = layers.Dense(512, kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs = [image_input, hist_input], outputs = outputs)
    return model

# LOAD IMAGES
train_dataset = load_images(
    labels_csv = "train_augmented.csv",
    images_dir = "train_augmented",
    shuffle = True
)
validation_dataset = load_images(
    labels_csv = "validation.csv",
    images_dir = "validation",
    shuffle = False
)

# BUILD & COMPILE MODEL
model = cnn_model()
model.compile(
    optimizer = optimizers.AdamW(learning_rate = 1e-3, weight_decay = 1e-4),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# CALLBACKS
early_stop = callbacks.EarlyStopping(patience = 13, restore_best_weights = True, monitor='val_accuracy')
lr_scheduler = callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.5, patience = 4, verbose = 1)
checkpoint = callbacks.ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', save_best_only = True, verbose = 1)

# TRAIN
history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = 100,
    callbacks = [early_stop, lr_scheduler, checkpoint]
)


# RESULTS
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
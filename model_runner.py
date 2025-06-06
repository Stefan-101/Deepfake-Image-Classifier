import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def extract_color_histogram_tf(image, bins = 16):
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

def process_image(path, img_size = (100, 100)):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, img_size) / 255.0
    histogram = extract_color_histogram_tf(img)
    return img, histogram

def build_test_dataset(csv_path, img_dir, batch_size = 64):
    df = pd.read_csv(csv_path)
    img_names = df.iloc[:, 0].values
    paths = [os.path.join(img_dir, img_name + ".png") for img_name in img_names]

    def _load(path):
        img, histogram = process_image(path)
        return {'input_layer': img, 'input_layer_1': histogram}

    data = tf.data.Dataset.from_tensor_slices(paths)
    data = data.map(_load)
    data = data.batch(batch_size)
    return data, df.iloc[:, 0].tolist()

# load model
model = load_model('best_model.h5')

# load data
test_ds, ids = build_test_dataset('test.csv', 'test')

# run model
preds = model.predict(test_ds)
pred_labels = preds.argmax(axis = 1)

# save output
submission = pd.DataFrame({'image_id': ids, 'label': pred_labels})
submission.to_csv('submission.csv', index = False)
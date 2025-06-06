import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

image_size = (100, 100)
train_csv = 'train.csv'
val_csv = 'validation.csv'
test_csv = 'test.csv'
train_dir = 'train'
val_dir = 'validation'
test_dir = 'test'
submission_path = 'submission.csv'

def extract_color_histogram(image, bins = 16):
    arr = np.array(image) / 255.0  # normalize
    hist = []
    for i in range(3):  # R, G, B channels
        channel = arr[..., i].flatten()
        h, _ = np.histogram(channel, bins = bins, range = (0.0, 1.0), density = True)
        hist.append(h)
    return np.concatenate(hist)  # shape: 3 * bins


# img loading function
def load_images(csv_file, image_dir, image_size, has_labels = True, use_histogram = True, bins = 16):
    df = pd.read_csv(csv_file)
    features, labels, ids = [], [], []

    for _, row in df.iterrows():
        image_id = row[0]
        ids.append(image_id)
        path = os.path.join(image_dir, image_id + '.png')
        image = Image.open(path).convert('RGB').resize(image_size)
        if use_histogram:
            feature = extract_color_histogram(image, bins)
        else:
            feature = np.array(image).flatten()
        features.append(feature)
        if has_labels:
            labels.append(int(row[1]))

    features = np.array(features)
    if has_labels:
        return features, np.array(labels), ids
    else:
        return features, ids

# load
X_train, y_train, _ = load_images(train_csv, train_dir, image_size)
X_val, y_val, _ = load_images(val_csv, val_dir, image_size)
X_test, test_ids = load_images(test_csv, test_dir, image_size, has_labels = False)

# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# run model for different k values
best_k = None
best_acc = 0
for k in [3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors = k, metric = 'manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"k={k}, Val Acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest k={best_k} with accuracy={best_acc:.4f}")

# run the model again for the best k
knn = KNeighborsClassifier(n_neighbors = best_k, metric = 'manhattan')
knn.fit(X_train, y_train)
y_test_pred = knn.predict(X_test)

# save submission
submission_df = pd.DataFrame({
    'image_id': test_ids,
    'label': y_test_pred
})
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

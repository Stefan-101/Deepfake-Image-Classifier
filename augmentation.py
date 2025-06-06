import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T

input_csv = 'train.csv'
input_dir = 'train'
augmented_dir = 'train_augmented'
output_csv = 'train_augmented.csv'
os.makedirs(augmented_dir, exist_ok = True)

df = pd.read_csv(input_csv)

# augmentation transforms
augmentations = T.Compose([
    T.RandomResizedCrop(100, scale = (0.7, 1.0)), 
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
    T.ToTensor(),
    T.ToPILImage()
])

augmentations_per_image = 3
augmented_data = []

for _, row in df.iterrows():
    image_id, label = row[0], row[1]
    image_path = os.path.join(input_dir, image_id + '.png')
    original_image = Image.open(image_path).convert('RGB')

    # save original to augmented_dir
    original_copy_path = os.path.join(augmented_dir, image_id + '.png')
    original_image.save(original_copy_path)
    augmented_data.append([image_id, label])

    # generate augmented imgs
    for i in range(augmentations_per_image):
        aug_img = augmentations(original_image)
        new_image_id = f"{image_id}_aug{i}"
        new_image_path = os.path.join(augmented_dir, new_image_id + '.png')
        aug_img.save(new_image_path)
        augmented_data.append([new_image_id, label])


augmented_df = pd.DataFrame(augmented_data, columns=['image_id', 'label'])
augmented_df.to_csv(output_csv, index = False)

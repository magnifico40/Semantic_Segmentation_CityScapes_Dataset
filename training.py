import glob
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model

#we use only train content of cityscapes dataset for shorter training
img_dir = '/content/data/cityscapes_train/Cityscape_Dataset/leftImg8bit/train'
mask_dir = '/content/data/cityscapes_train/Fine_Annotations/gtFine/train'

batch_size = 16
img_width = 512
img_height = 256
classes = 34 


def load_data(img_path:tf.string, mask_path:tf.string) -> Tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (img_height, img_width))
    img = tf.cast(img, tf.float32) /255.0 #uint8->float32 + normalization (0-1)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (img_height, img_width), method='nearest')
    return img, mask


def make_dataset(img_list: List[str], mask_list: List[str]) -> tf.data.Dataset:
    #much faster, loads only needed in paralell
    dataset = tf.data.Dataset.from_tensor_slices((img_list, mask_list))
    #without () - we pass instruction, not call
    #tf.data.AUTOTUNE - chooses "parallelism"
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)#iterates dateset and executes preprocess_data on each element 
    dataset = dataset.cache().shuffle(200).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def split_dataset(img_dir:str, mask_dir:str, train=0.7, val=0.15, test=0.15):
    img_paths:list[str] = sorted(glob.glob(os.path.join(img_dir, '*/*.png')))
    mask_paths:list[str] = sorted(glob.glob((os.path.join(mask_dir, '*/*_gtFine_labelIds.png'))))

    print(f"Number of images: {len(img_paths)}")
    print(f"Number of masks: {(len(mask_paths))}")
    
    if len(img_paths) != len(mask_paths):
        print("Size mismatch between images and masks")
        return
    
    idx = np.arange(len(img_paths)) #0, 1, 2..., n-1
    np.random.seed(42)
    np.random.shuffle(idx)
    img_paths = [img_paths[i] for i in idx]
    mask_paths = [mask_paths[i] for i in idx]

    total = len(img_paths)
    train_n = int(total*train)
    val_n = int(total*val)
    test_n = int(total*test)

    train_img = img_paths[0:train_n]
    train_mask = mask_paths[0:train_n]

    val_img = img_paths[train_n:train_n+val_n]
    val_mask = mask_paths[train_n:train_n+val_n]

    test_img = img_paths[train_n+val_n:train_n+val_n+test_n]
    test_mask = mask_paths[train_n+val_n:train_n+val_n+test_n]

    train_ds = make_dataset(train_img, train_mask)
    val_ds = make_dataset(val_img, val_mask)
    test_ds = make_dataset(test_img, test_mask)
    
    return train_ds, val_ds, test_ds

def build_unet_model(input_shape, num_classes):
    inputs = Input(input_shape)

    #  ENCODER (Down) 
    # Block 1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Block 2
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Block 3
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Block 4
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # BOTTLENECK 
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # DECODER (Up)
    # Block 6
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    # Block 7
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    # Block 8
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    # Block 9
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # OUTPUT 
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def show_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1) #row, col, id
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.title('loss graph')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.title('accuracy graph')
    plt.show()

def show_results(model, dataset, n=3):
    np.random.seed(42)
    rand_color=[]
    for i in range(classes):
        rand_color.append(random.choices(range(256), k=3))
    rand_color = np.array(rand_color)
    
    
    plt.figure(figsize=(15, 5*n))
    for i, (img, mask) in enumerate(dataset.take(n)):
        prediction = model.predict(img)
        best_mask = tf.argmax(prediction, axis=-1) #for each pixel returns class with biggest propability (among 34 likehoods)

        #org image
        plt.subplot(n, 3, 3*i+1)
        plt.imshow(img[0])
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(n, 3, i * 3 + 2)
        gt_img = rand_color[mask[0].numpy().squeeze()]
        plt.imshow(gt_img.astype(np.uint8))
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(n, 3, i * 3 + 3)
        pred_img = rand_color[best_mask[0].numpy().squeeze()]
        plt.imshow(pred_img.astype(np.uint8))
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.show()




if __name__=="__main__":
    print("Loading and splitting data")
    train_ds, val_ds, test_ds = split_dataset(img_dir, mask_dir)

    print("Building unet model")
    unet = build_unet_model((img_height, img_width, 3), classes)
    unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint('unet_cityscapes_raw.keras', save_best_only=True), #saves best weights model
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss') #learnig stops, when for next 5 epochs, val_loss does not improve
    ]

    print("Training model")
    #main learing loop
    history = unet.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        steps_per_epoch=100,      # Process only 100 batches per epoch 
        validation_steps=60       # Validate on 20 batches
    )
    
    print("Plotting history")
    show_history(history)

    print("Show result examples")
    show_results(unet, test_ds, n=3)

    print("Final results")
    results = unet.evaluate(test_ds)

    print(f"Final Test Loss:     {results[0]:.4f}")
    print(f"Final Test Accuracy: {results[1]*100:.2f}%")

    print("Saving model")
    unet.save('cityscapes_unet_final.keras')
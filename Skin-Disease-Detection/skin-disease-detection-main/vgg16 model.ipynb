{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "800bc18f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from glob import glob\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from keras.utils.np_utils import to_categorical # convert to one-hot-encoding\n",
    "\n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras import layers\n",
    "from keras import Model\n",
    "from keras.applications.vgg16 import VGG16, preprocess_input\n",
    "from keras.optimizers import Adam\n",
    "from keras.callbacks import ReduceLROnPlateau, EarlyStopping\n",
    "\n",
    "\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "541e0715",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = np.load(\"256_192_train.npy\")\n",
    "y_train = np.load(\"train_labels.npy\")\n",
    "X_val = np.load(\"256_192_val.npy\")\n",
    "y_val = np.load(\"val_labels.npy\")\n",
    "\n",
    "X_train.shape, X_val.shape\n",
    "y_train.shape, y_val.shape\n",
    "\n",
    "y_train = to_categorical(y_train)\n",
    "y_val = to_categorical(y_val)\n",
    "y_train.shape, y_val.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "89381c45",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
      "58889256/58889256 [==============================] - 52s 1us/step\n",
      "input_1\n",
      "block1_conv1\n",
      "block1_conv2\n",
      "block1_pool\n",
      "block2_conv1\n",
      "block2_conv2\n",
      "block2_pool\n",
      "block3_conv1\n",
      "block3_conv2\n",
      "block3_conv3\n",
      "block3_pool\n",
      "block4_conv1\n",
      "block4_conv2\n",
      "block4_conv3\n",
      "block4_pool\n",
      "block5_conv1\n",
      "block5_conv2\n",
      "block5_conv3\n",
      "block5_pool\n",
      "19\n"
     ]
    }
   ],
   "source": [
    "pre_trained_model = VGG16(input_shape=(192, 256, 3), include_top=False, weights=\"imagenet\")\n",
    "for layer in pre_trained_model.layers:\n",
    "    print(layer.name)\n",
    "    layer.trainable = False\n",
    "    \n",
    "print(len(pre_trained_model.layers))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "645e7c88",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "last layer output shape: (None, 6, 8, 512)\n"
     ]
    }
   ],
   "source": [
    "last_layer = pre_trained_model.get_layer('block5_pool')\n",
    "print('last layer output shape:', last_layer.output_shape)\n",
    "last_output = last_layer.output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "64196c4a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " input_1 (InputLayer)        [(None, 192, 256, 3)]     0         \n",
      "                                                                 \n",
      " block1_conv1 (Conv2D)       (None, 192, 256, 64)      1792      \n",
      "                                                                 \n",
      " block1_conv2 (Conv2D)       (None, 192, 256, 64)      36928     \n",
      "                                                                 \n",
      " block1_pool (MaxPooling2D)  (None, 96, 128, 64)       0         \n",
      "                                                                 \n",
      " block2_conv1 (Conv2D)       (None, 96, 128, 128)      73856     \n",
      "                                                                 \n",
      " block2_conv2 (Conv2D)       (None, 96, 128, 128)      147584    \n",
      "                                                                 \n",
      " block2_pool (MaxPooling2D)  (None, 48, 64, 128)       0         \n",
      "                                                                 \n",
      " block3_conv1 (Conv2D)       (None, 48, 64, 256)       295168    \n",
      "                                                                 \n",
      " block3_conv2 (Conv2D)       (None, 48, 64, 256)       590080    \n",
      "                                                                 \n",
      " block3_conv3 (Conv2D)       (None, 48, 64, 256)       590080    \n",
      "                                                                 \n",
      " block3_pool (MaxPooling2D)  (None, 24, 32, 256)       0         \n",
      "                                                                 \n",
      " block4_conv1 (Conv2D)       (None, 24, 32, 512)       1180160   \n",
      "                                                                 \n",
      " block4_conv2 (Conv2D)       (None, 24, 32, 512)       2359808   \n",
      "                                                                 \n",
      " block4_conv3 (Conv2D)       (None, 24, 32, 512)       2359808   \n",
      "                                                                 \n",
      " block4_pool (MaxPooling2D)  (None, 12, 16, 512)       0         \n",
      "                                                                 \n",
      " block5_conv1 (Conv2D)       (None, 12, 16, 512)       2359808   \n",
      "                                                                 \n",
      " block5_conv2 (Conv2D)       (None, 12, 16, 512)       2359808   \n",
      "                                                                 \n",
      " block5_conv3 (Conv2D)       (None, 12, 16, 512)       2359808   \n",
      "                                                                 \n",
      " block5_pool (MaxPooling2D)  (None, 6, 8, 512)         0         \n",
      "                                                                 \n",
      " global_max_pooling2d (Globa  (None, 512)              0         \n",
      " lMaxPooling2D)                                                  \n",
      "                                                                 \n",
      " dense (Dense)               (None, 512)               262656    \n",
      "                                                                 \n",
      " dropout (Dropout)           (None, 512)               0         \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 7)                 3591      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 14,980,935\n",
      "Trainable params: 266,247\n",
      "Non-trainable params: 14,714,688\n",
      "_________________________________________________________________\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\chama\\anaconda3\\lib\\site-packages\\keras\\optimizers\\optimizer_v2\\adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
      "  super().__init__(name, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "# Flatten the output layer to 1 dimension\n",
    "x = layers.GlobalMaxPooling2D()(last_output)\n",
    "# Add a fully connected layer with 512 hidden units and ReLU activation\n",
    "x = layers.Dense(512, activation='relu')(x)\n",
    "# Add a dropout rate of 0.5\n",
    "x = layers.Dropout(0.5)(x)\n",
    "# Add a final sigmoid layer for classification\n",
    "x = layers.Dense(7, activation='softmax')(x)\n",
    "\n",
    "# Configure and compile the model\n",
    "\n",
    "model = Model(pre_trained_model.input, x)\n",
    "optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)\n",
    "model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "aa732ee4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\chama\\AppData\\Local\\Temp\\ipykernel_8308\\3698051878.py:10: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
      "  history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/3\n",
      "126/126 [==============================] - 2038s 16s/step - loss: 1.4026 - accuracy: 0.5834 - val_loss: 1.0571 - val_accuracy: 0.6696\n",
      "Epoch 2/3\n",
      "126/126 [==============================] - 1994s 16s/step - loss: 1.1078 - accuracy: 0.6612 - val_loss: 1.0040 - val_accuracy: 0.6696\n",
      "Epoch 3/3\n",
      "126/126 [==============================] - 2009s 16s/step - loss: 1.0385 - accuracy: 0.6676 - val_loss: 1.0132 - val_accuracy: 0.6540\n"
     ]
    }
   ],
   "source": [
    "train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,\n",
    "                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')\n",
    "\n",
    "train_datagen.fit(X_train)\n",
    "\n",
    "val_datagen = ImageDataGenerator()\n",
    "val_datagen.fit(X_val)\n",
    "batch_size = 64\n",
    "epochs = 3\n",
    "history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),\n",
    "                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),\n",
    "                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size), \n",
    "                              validation_steps=(X_val.shape[0] // batch_size))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cbc56ecd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " input_1 (InputLayer)        [(None, 192, 256, 3)]     0         \n",
      "                                                                 \n",
      " block1_conv1 (Conv2D)       (None, 192, 256, 64)      1792      \n",
      "                                                                 \n",
      " block1_conv2 (Conv2D)       (None, 192, 256, 64)      36928     \n",
      "                                                                 \n",
      " block1_pool (MaxPooling2D)  (None, 96, 128, 64)       0         \n",
      "                                                                 \n",
      " block2_conv1 (Conv2D)       (None, 96, 128, 128)      73856     \n",
      "                                                                 \n",
      " block2_conv2 (Conv2D)       (None, 96, 128, 128)      147584    \n",
      "                                                                 \n",
      " block2_pool (MaxPooling2D)  (None, 48, 64, 128)       0         \n",
      "                                                                 \n",
      " block3_conv1 (Conv2D)       (None, 48, 64, 256)       295168    \n",
      "                                                                 \n",
      " block3_conv2 (Conv2D)       (None, 48, 64, 256)       590080    \n",
      "                                                                 \n",
      " block3_conv3 (Conv2D)       (None, 48, 64, 256)       590080    \n",
      "                                                                 \n",
      " block3_pool (MaxPooling2D)  (None, 24, 32, 256)       0         \n",
      "                                                                 \n",
      " block4_conv1 (Conv2D)       (None, 24, 32, 512)       1180160   \n",
      "                                                                 \n",
      " block4_conv2 (Conv2D)       (None, 24, 32, 512)       2359808   \n",
      "                                                                 \n",
      " block4_conv3 (Conv2D)       (None, 24, 32, 512)       2359808   \n",
      "                                                                 \n",
      " block4_pool (MaxPooling2D)  (None, 12, 16, 512)       0         \n",
      "                                                                 \n",
      " block5_conv1 (Conv2D)       (None, 12, 16, 512)       2359808   \n",
      "                                                                 \n",
      " block5_conv2 (Conv2D)       (None, 12, 16, 512)       2359808   \n",
      "                                                                 \n",
      " block5_conv3 (Conv2D)       (None, 12, 16, 512)       2359808   \n",
      "                                                                 \n",
      " block5_pool (MaxPooling2D)  (None, 6, 8, 512)         0         \n",
      "                                                                 \n",
      " global_max_pooling2d (Globa  (None, 512)              0         \n",
      " lMaxPooling2D)                                                  \n",
      "                                                                 \n",
      " dense (Dense)               (None, 512)               262656    \n",
      "                                                                 \n",
      " dropout (Dropout)           (None, 512)               0         \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 7)                 3591      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 14,980,935\n",
      "Trainable params: 7,345,671\n",
      "Non-trainable params: 7,635,264\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "for layer in model.layers[:15]:\n",
    "    layer.trainable = False\n",
    "for layer in model.layers[15:]:\n",
    "    layer.trainable = True\n",
    "optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)\n",
    "model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['acc'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2772b393",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\chama\\AppData\\Local\\Temp\\ipykernel_8308\\1422572830.py:5: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
      "  history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n",
      "126/126 [==============================] - 2442s 19s/step - loss: 0.9208 - acc: 0.6887 - val_loss: 0.8281 - val_acc: 0.6830 - lr: 1.0000e-04\n",
      "Epoch 2/30\n",
      "126/126 [==============================] - 2467s 20s/step - loss: 0.7668 - acc: 0.7228 - val_loss: 0.8067 - val_acc: 0.6897 - lr: 1.0000e-04\n",
      "Epoch 3/30\n",
      "126/126 [==============================] - 2472s 20s/step - loss: 0.7190 - acc: 0.7421 - val_loss: 0.7671 - val_acc: 0.7076 - lr: 1.0000e-04\n",
      "Epoch 4/30\n",
      "126/126 [==============================] - 2555s 20s/step - loss: 0.6819 - acc: 0.7512 - val_loss: 0.7166 - val_acc: 0.7321 - lr: 1.0000e-04\n",
      "Epoch 5/30\n",
      "126/126 [==============================] - 2622s 21s/step - loss: 0.6446 - acc: 0.7661 - val_loss: 0.7310 - val_acc: 0.7076 - lr: 1.0000e-04\n",
      "Epoch 6/30\n",
      "126/126 [==============================] - 2476s 20s/step - loss: 0.6250 - acc: 0.7706 - val_loss: 0.7496 - val_acc: 0.7188 - lr: 1.0000e-04\n",
      "Epoch 7/30\n",
      "126/126 [==============================] - 2861s 23s/step - loss: 0.5984 - acc: 0.7834 - val_loss: 0.6943 - val_acc: 0.7478 - lr: 1.0000e-04\n",
      "Epoch 8/30\n",
      "126/126 [==============================] - 2478s 20s/step - loss: 0.5728 - acc: 0.7979 - val_loss: 0.7048 - val_acc: 0.7299 - lr: 1.0000e-04\n",
      "Epoch 9/30\n",
      "126/126 [==============================] - 14975s 120s/step - loss: 0.5583 - acc: 0.7979 - val_loss: 0.5999 - val_acc: 0.7835 - lr: 1.0000e-04\n",
      "Epoch 10/30\n",
      "126/126 [==============================] - 2485s 20s/step - loss: 0.5365 - acc: 0.8075 - val_loss: 0.6743 - val_acc: 0.7500 - lr: 1.0000e-04\n",
      "Epoch 11/30\n",
      "126/126 [==============================] - 3160s 25s/step - loss: 0.5004 - acc: 0.8224 - val_loss: 0.6373 - val_acc: 0.7522 - lr: 1.0000e-04\n",
      "Epoch 12/30\n",
      "126/126 [==============================] - ETA: 0s - loss: 0.4893 - acc: 0.8222 \n",
      "Epoch 12: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.\n",
      "126/126 [==============================] - 2454s 19s/step - loss: 0.4893 - acc: 0.8222 - val_loss: 0.6330 - val_acc: 0.7835 - lr: 1.0000e-04\n",
      "Epoch 13/30\n",
      "126/126 [==============================] - 2992s 24s/step - loss: 0.4386 - acc: 0.8426 - val_loss: 0.7135 - val_acc: 0.7567 - lr: 5.0000e-05\n",
      "Epoch 14/30\n",
      "126/126 [==============================] - 3238s 26s/step - loss: 0.4047 - acc: 0.8555 - val_loss: 0.6265 - val_acc: 0.7879 - lr: 5.0000e-05\n",
      "Epoch 15/30\n",
      "126/126 [==============================] - 2446s 19s/step - loss: 0.4014 - acc: 0.8568 - val_loss: 0.7224 - val_acc: 0.7388 - lr: 5.0000e-05\n",
      "Epoch 16/30\n",
      "126/126 [==============================] - 2922s 23s/step - loss: 0.3867 - acc: 0.8618 - val_loss: 0.5899 - val_acc: 0.7723 - lr: 5.0000e-05\n",
      "Epoch 17/30\n",
      "126/126 [==============================] - ETA: 0s - loss: 0.3698 - acc: 0.8679 \n",
      "Epoch 17: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-05.\n",
      "126/126 [==============================] - 2598s 21s/step - loss: 0.3698 - acc: 0.8679 - val_loss: 0.6378 - val_acc: 0.7768 - lr: 5.0000e-05\n",
      "Epoch 18/30\n",
      "126/126 [==============================] - 3402s 27s/step - loss: 0.3356 - acc: 0.8818 - val_loss: 0.5826 - val_acc: 0.7946 - lr: 2.5000e-05\n",
      "Epoch 19/30\n",
      "126/126 [==============================] - 2474s 20s/step - loss: 0.3191 - acc: 0.8855 - val_loss: 0.6258 - val_acc: 0.7924 - lr: 2.5000e-05\n",
      "Epoch 20/30\n",
      "126/126 [==============================] - 2444s 19s/step - loss: 0.3164 - acc: 0.8831 - val_loss: 0.6435 - val_acc: 0.7835 - lr: 2.5000e-05\n",
      "Epoch 21/30\n",
      "126/126 [==============================] - 2442s 19s/step - loss: 0.3073 - acc: 0.8921 - val_loss: 0.6108 - val_acc: 0.7746 - lr: 2.5000e-05\n",
      "Epoch 22/30\n",
      "126/126 [==============================] - 2432s 19s/step - loss: 0.3054 - acc: 0.8933 - val_loss: 0.6002 - val_acc: 0.8080 - lr: 2.5000e-05\n",
      "Epoch 23/30\n",
      "126/126 [==============================] - 2436s 19s/step - loss: 0.2952 - acc: 0.8954 - val_loss: 0.6233 - val_acc: 0.7991 - lr: 2.5000e-05\n",
      "Epoch 24/30\n",
      "126/126 [==============================] - 2435s 19s/step - loss: 0.2894 - acc: 0.8951 - val_loss: 0.5971 - val_acc: 0.8080 - lr: 2.5000e-05\n",
      "Epoch 25/30\n",
      "126/126 [==============================] - ETA: 0s - loss: 0.2757 - acc: 0.9023 \n",
      "Epoch 25: ReduceLROnPlateau reducing learning rate to 1.249999968422344e-05.\n",
      "126/126 [==============================] - 2435s 19s/step - loss: 0.2757 - acc: 0.9023 - val_loss: 0.6371 - val_acc: 0.8058 - lr: 2.5000e-05\n",
      "Epoch 26/30\n",
      "126/126 [==============================] - 2432s 19s/step - loss: 0.2623 - acc: 0.9100 - val_loss: 0.6484 - val_acc: 0.7879 - lr: 1.2500e-05\n",
      "Epoch 27/30\n",
      "126/126 [==============================] - 2678s 21s/step - loss: 0.2502 - acc: 0.9124 - val_loss: 0.6582 - val_acc: 0.8147 - lr: 1.2500e-05\n",
      "Epoch 28/30\n",
      "126/126 [==============================] - 2437s 19s/step - loss: 0.2510 - acc: 0.9100 - val_loss: 0.6561 - val_acc: 0.7768 - lr: 1.2500e-05\n",
      "Epoch 29/30\n",
      "126/126 [==============================] - 2444s 19s/step - loss: 0.2479 - acc: 0.9121 - val_loss: 0.5672 - val_acc: 0.8192 - lr: 1.2500e-05\n",
      "Epoch 30/30\n",
      "126/126 [==============================] - 2438s 19s/step - loss: 0.2453 - acc: 0.9121 - val_loss: 0.6971 - val_acc: 0.7835 - lr: 1.2500e-05\n"
     ]
    }
   ],
   "source": [
    "learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, \n",
    "                                            min_lr=0.000001, cooldown=3)\n",
    "batch_size = 64\n",
    "epochs = 30\n",
    "history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),\n",
    "                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),\n",
    "                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),\n",
    "                              validation_steps=(X_val.shape[0] // batch_size), callbacks=[learning_rate_reduction])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8eedfb32",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "29/29 [==============================] - 207s 7s/step - loss: 0.6209 - acc: 0.7905\n",
      "Validation: accuracy = 0.790466  ;  loss_v = 0.620855\n",
      "32/32 [==============================] - 229s 7s/step - loss: 0.6150 - acc: 0.8034\n",
      "Test: accuracy = 0.803393  ;  loss = 0.615045\n"
     ]
    }
   ],
   "source": [
    "loss_val, acc_val = model.evaluate(X_val, y_val, verbose=1)\n",
    "print(\"Validation: accuracy = %f  ;  loss_v = %f\" % (acc_val, loss_val))\n",
    "X_test = np.load(\"256_192_test.npy\")\n",
    "y_test = np.load(\"test_labels.npy\")\n",
    "y_test = to_categorical(y_test)\n",
    "loss_test, acc_test = model.evaluate(X_test, y_test, verbose=1)\n",
    "print(\"Test: accuracy = %f  ;  loss = %f\" % (acc_test, loss_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "723af4d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"VGG16.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f41ffb8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

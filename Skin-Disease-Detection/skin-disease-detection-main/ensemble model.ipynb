{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "df7442de",
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
    "from keras import Model, Input\n",
    "from keras.applications.inception_v3 import InceptionV3\n",
    "from keras.applications.densenet import DenseNet201\n",
    "\n",
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
   "execution_count": 2,
   "id": "76fdd066",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_val = np.load(\"256_192_val.npy\")\n",
    "X_test = np.load(\"256_192_test.npy\")\n",
    "y_val = np.load(\"val_labels.npy\")\n",
    "y_test = np.load(\"test_labels.npy\")\n",
    "y_val = to_categorical(y_val)\n",
    "y_test = to_categorical(y_test)\n",
    "X_val.shape, X_test.shape, y_val.shape, y_test.shape\n",
    "input_shape = X_val[0,:,:,:].shape\n",
    "model_input = Input(shape=input_shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "26a081bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "last layer output shape: (None, 4, 6, 2048)\n"
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
    "inception = InceptionV3(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)\n",
    "for layer in inception.layers:\n",
    "    layer.trainable = True\n",
    "inception_last_layer = inception.get_layer('mixed10')\n",
    "print('last layer output shape:', inception_last_layer.output_shape)\n",
    "inception_last_output = inception_last_layer.output\n",
    "# Flatten the output layer to 1 dimension\n",
    "x_inception = layers.GlobalMaxPooling2D()(inception_last_output)\n",
    "# Add a fully connected layer with 512 hidden units and ReLU activation\n",
    "x_inception = layers.Dense(512, activation='relu')(x_inception)\n",
    "# Add a dropout rate of 0.7\n",
    "x_inception = layers.Dropout(0.5)(x_inception)\n",
    "# Add a final sigmoid layer for classification\n",
    "x_inception = layers.Dense(7, activation='softmax')(x_inception)\n",
    "\n",
    "# Configure and compile the model\n",
    "\n",
    "inception_model = Model(model_input, x_inception)\n",
    "optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)\n",
    "inception_model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['accuracy'])\n",
    "inception_model.load_weights(\"InceptionV3.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "84a5b624",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "last layer output shape: (None, 6, 8, 1920)\n"
     ]
    }
   ],
   "source": [
    "denseNet = DenseNet201(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)\n",
    "for layer in denseNet.layers:\n",
    "    layer.trainable = True\n",
    "denseNet_last_layer = denseNet.get_layer('relu')\n",
    "print('last layer output shape:', denseNet_last_layer.output_shape)\n",
    "denseNet_last_output = denseNet_last_layer.output\n",
    "# Flatten the output layer to 1 dimension\n",
    "x_denseNet = layers.GlobalMaxPooling2D()(denseNet_last_output)\n",
    "# Add a fully connected layer with 512 hidden units and ReLU activation\n",
    "x_denseNet = layers.Dense(512, activation='relu')(x_denseNet)\n",
    "# Add a dropout rate of 0.7\n",
    "x_denseNet = layers.Dropout(0.5)(x_denseNet)\n",
    "# Add a final sigmoid layer for classification\n",
    "x_denseNet = layers.Dense(7, activation='softmax')(x_denseNet)\n",
    "\n",
    "# Configure and compile the model\n",
    "\n",
    "denseNet_model = Model(model_input, x_denseNet)\n",
    "optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)\n",
    "denseNet_model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['accuracy'])\n",
    "denseNet_model.load_weights(\"DenseNet.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cb2f1535",
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
    "from keras.applications.vgg16 import VGG16, preprocess_input\n",
    "\n",
    "vgg16 = VGG16(input_shape=input_shape, input_tensor=model_input,include_top=False, weights=\"imagenet\")\n",
    "for layer in vgg16.layers:\n",
    "    layer.trainable = True\n",
    "vgg16_last_layer = vgg16.get_layer('block5_pool')\n",
    "print('last layer output shape:', vgg16_last_layer.output_shape)\n",
    "vgg16_last_output = vgg16_last_layer.output\n",
    "# Flatten the output layer to 1 dimension\n",
    "x_vgg = layers.GlobalMaxPooling2D()(vgg16_last_output)\n",
    "# Add a fully connected layer with 512 hidden units and ReLU activation\n",
    "x_vgg = layers.Dense(512, activation='relu')(x_vgg)\n",
    "# Add a dropout rate of 0.7\n",
    "x_vgg = layers.Dropout(0.5)(x_vgg)\n",
    "# Add a final sigmoid layer for classification\n",
    "x_vgg = layers.Dense(7, activation='softmax')(x_vgg)\n",
    "\n",
    "# Configure and compile the model\n",
    "\n",
    "vgg_model = Model(model_input, x_vgg)\n",
    "optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)\n",
    "vgg_model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['accuracy'])\n",
    "vgg_model.load_weights(\"VGG16.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "07c02e72",
   "metadata": {},
   "outputs": [],
   "source": [
    "def ensemble(models, model_input):\n",
    "    outputs = [model.outputs[0] for model in models]\n",
    "    y = layers.Average()(outputs)\n",
    "    model = Model(model_input, y, name='ensemble')\n",
    "    return model\n",
    "ensemble_model = ensemble([denseNet_model, inception_model, vgg_model], model_input)\n",
    "ensemble_model.compile(loss='categorical_crossentropy',\n",
    "              optimizer=optimizer,\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "69122e9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "29/29 [==============================] - 864s 27s/step - loss: 0.3974 - accuracy: 0.8470\n",
      "Validation: accuracy = 0.847007  ;  loss_v = 0.397386\n",
      "32/32 [==============================] - 825s 26s/step - loss: 0.4350 - accuracy: 0.8503\n",
      "Test: accuracy = 0.850299  ;  loss = 0.434979\n"
     ]
    }
   ],
   "source": [
    "loss_val, acc_val = ensemble_model.evaluate(X_val, y_val, verbose=1)\n",
    "print(\"Validation: accuracy = %f  ;  loss_v = %f\" % (acc_val, loss_val))\n",
    "loss_test, acc_test = ensemble_model.evaluate(X_test, y_test, verbose=1)\n",
    "print(\"Test: accuracy = %f  ;  loss = %f\" % (acc_test, loss_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0bab3808",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "35ce2277",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "828a843b",
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

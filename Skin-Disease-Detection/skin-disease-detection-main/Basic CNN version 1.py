{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ec68730f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "306ed58e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv('hmnist_28_28_RGB.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "050debc5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pixel0000</th>\n",
       "      <th>pixel0001</th>\n",
       "      <th>pixel0002</th>\n",
       "      <th>pixel0003</th>\n",
       "      <th>pixel0004</th>\n",
       "      <th>pixel0005</th>\n",
       "      <th>pixel0006</th>\n",
       "      <th>pixel0007</th>\n",
       "      <th>pixel0008</th>\n",
       "      <th>pixel0009</th>\n",
       "      <th>...</th>\n",
       "      <th>pixel2343</th>\n",
       "      <th>pixel2344</th>\n",
       "      <th>pixel2345</th>\n",
       "      <th>pixel2346</th>\n",
       "      <th>pixel2347</th>\n",
       "      <th>pixel2348</th>\n",
       "      <th>pixel2349</th>\n",
       "      <th>pixel2350</th>\n",
       "      <th>pixel2351</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>192</td>\n",
       "      <td>153</td>\n",
       "      <td>193</td>\n",
       "      <td>195</td>\n",
       "      <td>155</td>\n",
       "      <td>192</td>\n",
       "      <td>197</td>\n",
       "      <td>154</td>\n",
       "      <td>185</td>\n",
       "      <td>202</td>\n",
       "      <td>...</td>\n",
       "      <td>173</td>\n",
       "      <td>124</td>\n",
       "      <td>138</td>\n",
       "      <td>183</td>\n",
       "      <td>147</td>\n",
       "      <td>166</td>\n",
       "      <td>185</td>\n",
       "      <td>154</td>\n",
       "      <td>177</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>25</td>\n",
       "      <td>14</td>\n",
       "      <td>30</td>\n",
       "      <td>68</td>\n",
       "      <td>48</td>\n",
       "      <td>75</td>\n",
       "      <td>123</td>\n",
       "      <td>93</td>\n",
       "      <td>126</td>\n",
       "      <td>158</td>\n",
       "      <td>...</td>\n",
       "      <td>60</td>\n",
       "      <td>39</td>\n",
       "      <td>55</td>\n",
       "      <td>25</td>\n",
       "      <td>14</td>\n",
       "      <td>28</td>\n",
       "      <td>25</td>\n",
       "      <td>14</td>\n",
       "      <td>27</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>192</td>\n",
       "      <td>138</td>\n",
       "      <td>153</td>\n",
       "      <td>200</td>\n",
       "      <td>145</td>\n",
       "      <td>163</td>\n",
       "      <td>201</td>\n",
       "      <td>142</td>\n",
       "      <td>160</td>\n",
       "      <td>206</td>\n",
       "      <td>...</td>\n",
       "      <td>167</td>\n",
       "      <td>129</td>\n",
       "      <td>143</td>\n",
       "      <td>159</td>\n",
       "      <td>124</td>\n",
       "      <td>142</td>\n",
       "      <td>136</td>\n",
       "      <td>104</td>\n",
       "      <td>117</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>38</td>\n",
       "      <td>19</td>\n",
       "      <td>30</td>\n",
       "      <td>95</td>\n",
       "      <td>59</td>\n",
       "      <td>72</td>\n",
       "      <td>143</td>\n",
       "      <td>103</td>\n",
       "      <td>119</td>\n",
       "      <td>171</td>\n",
       "      <td>...</td>\n",
       "      <td>44</td>\n",
       "      <td>26</td>\n",
       "      <td>36</td>\n",
       "      <td>25</td>\n",
       "      <td>12</td>\n",
       "      <td>17</td>\n",
       "      <td>25</td>\n",
       "      <td>12</td>\n",
       "      <td>15</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>158</td>\n",
       "      <td>113</td>\n",
       "      <td>139</td>\n",
       "      <td>194</td>\n",
       "      <td>144</td>\n",
       "      <td>174</td>\n",
       "      <td>215</td>\n",
       "      <td>162</td>\n",
       "      <td>191</td>\n",
       "      <td>225</td>\n",
       "      <td>...</td>\n",
       "      <td>209</td>\n",
       "      <td>166</td>\n",
       "      <td>185</td>\n",
       "      <td>172</td>\n",
       "      <td>135</td>\n",
       "      <td>149</td>\n",
       "      <td>109</td>\n",
       "      <td>78</td>\n",
       "      <td>92</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 2353 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   pixel0000  pixel0001  pixel0002  pixel0003  pixel0004  pixel0005  \\\n",
       "0        192        153        193        195        155        192   \n",
       "1         25         14         30         68         48         75   \n",
       "2        192        138        153        200        145        163   \n",
       "3         38         19         30         95         59         72   \n",
       "4        158        113        139        194        144        174   \n",
       "\n",
       "   pixel0006  pixel0007  pixel0008  pixel0009  ...  pixel2343  pixel2344  \\\n",
       "0        197        154        185        202  ...        173        124   \n",
       "1        123         93        126        158  ...         60         39   \n",
       "2        201        142        160        206  ...        167        129   \n",
       "3        143        103        119        171  ...         44         26   \n",
       "4        215        162        191        225  ...        209        166   \n",
       "\n",
       "   pixel2345  pixel2346  pixel2347  pixel2348  pixel2349  pixel2350  \\\n",
       "0        138        183        147        166        185        154   \n",
       "1         55         25         14         28         25         14   \n",
       "2        143        159        124        142        136        104   \n",
       "3         36         25         12         17         25         12   \n",
       "4        185        172        135        149        109         78   \n",
       "\n",
       "   pixel2351  label  \n",
       "0        177      2  \n",
       "1         27      2  \n",
       "2        117      2  \n",
       "3         15      2  \n",
       "4         92      2  \n",
       "\n",
       "[5 rows x 2353 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "be6c7615",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6, 1, 4, 2, 0, 5, 3], dtype=int64)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fractions=np.array([0.8,0.2])\n",
    "df=df.sample(frac=1)\n",
    "train_set, test_set = np.array_split(\n",
    "    df, (fractions[:-1].cumsum() * len(df)).astype(int))\n",
    "     \n",
    "df.label.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1272e68a",
   "metadata": {},
   "outputs": [],
   "source": [
    "classes={0:('akiec', 'actinic keratoses and intraepithelial carcinomae'),\n",
    "         1:('bcc' , 'basal cell carcinoma'),\n",
    "         2:('bkl', 'benign keratosis-like lesions'),\n",
    "         3:('df', 'dermatofibroma'),\n",
    "         4:('nv', ' melanocytic nevi'),\n",
    "         5:('vasc', ' pyogenic granulomas and hemorrhage'),\n",
    "         6:('mel', 'melanoma'),}\n",
    "     \n",
    "\n",
    "y_train=train_set['label']\n",
    "x_train=train_set.drop(columns=['label'])\n",
    "y_test=test_set['label']\n",
    "x_test=test_set.drop(columns=['label'])\n",
    "\n",
    "columns=list(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f2bb5957",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='label', ylabel='count'>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "sns.countplot(train_set['label'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e3136077",
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import RandomOverSampler \n",
    "oversample = RandomOverSampler()\n",
    "x_train,y_train  = oversample.fit_resample(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8665c19f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='label', ylabel='count'>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "79bdf05f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "num=random.randint(0,8000)\n",
    "x_train=np.array(x_train, dtype=np.uint8).reshape(-1,28,28,3)\n",
    "\n",
    "plt.imshow(x_train[num].reshape(28,28,3))\n",
    "plt.title(\"Random image from training data\")\n",
    "plt.show()\n",
    "num=random.randint(0,8000)\n",
    "plt.imshow(x_train[num].reshape(28,28,3))\n",
    "plt.title(\"Random image from training data\")\n",
    "plt.show()\n",
    "\n",
    "num=random.randint(0,8000)\n",
    "plt.imshow(x_train[num].reshape(28,28,3))\n",
    "plt.title(\"Random image from training data\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "be20201e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 28, 28, 16)        448       \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2D  (None, 14, 14, 16)       0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " batch_normalization (BatchN  (None, 14, 14, 16)       64        \n",
      " ormalization)                                                   \n",
      "                                                                 \n",
      " conv2d_1 (Conv2D)           (None, 12, 12, 32)        4640      \n",
      "                                                                 \n",
      " conv2d_2 (Conv2D)           (None, 10, 10, 64)        18496     \n",
      "                                                                 \n",
      " max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         \n",
      " 2D)                                                             \n",
      "                                                                 \n",
      " batch_normalization_1 (Batc  (None, 5, 5, 64)         256       \n",
      " hNormalization)                                                 \n",
      "                                                                 \n",
      " conv2d_3 (Conv2D)           (None, 3, 3, 128)         73856     \n",
      "                                                                 \n",
      " conv2d_4 (Conv2D)           (None, 1, 1, 256)         295168    \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 256)               0         \n",
      "                                                                 \n",
      " dropout (Dropout)           (None, 256)               0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 256)               65792     \n",
      "                                                                 \n",
      " batch_normalization_2 (Batc  (None, 256)              1024      \n",
      " hNormalization)                                                 \n",
      "                                                                 \n",
      " dropout_1 (Dropout)         (None, 256)               0         \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 128)               32896     \n",
      "                                                                 \n",
      " batch_normalization_3 (Batc  (None, 128)              512       \n",
      " hNormalization)                                                 \n",
      "                                                                 \n",
      " dense_2 (Dense)             (None, 64)                8256      \n",
      "                                                                 \n",
      " batch_normalization_4 (Batc  (None, 64)               256       \n",
      " hNormalization)                                                 \n",
      "                                                                 \n",
      " dropout_2 (Dropout)         (None, 64)                0         \n",
      "                                                                 \n",
      " dense_3 (Dense)             (None, 32)                2080      \n",
      "                                                                 \n",
      " batch_normalization_5 (Batc  (None, 32)               128       \n",
      " hNormalization)                                                 \n",
      "                                                                 \n",
      " dense_4 (Dense)             (None, 7)                 231       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 504,103\n",
      "Trainable params: 502,983\n",
      "Non-trainable params: 1,120\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D\n",
    "import tensorflow as tf\n",
    "     \n",
    "\n",
    "model = Sequential()\n",
    "model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))\n",
    "model.add(MaxPool2D(pool_size = (2,2)))\n",
    "model.add(tf.keras.layers.BatchNormalization())\n",
    "model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))\n",
    "model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))\n",
    "model.add(MaxPool2D(pool_size = (2,2)))\n",
    "model.add(tf.keras.layers.BatchNormalization())\n",
    "model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))\n",
    "model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu'))\n",
    "model.add(Flatten())\n",
    "model.add(tf.keras.layers.Dropout(0.2))\n",
    "model.add(Dense(256,activation='relu'))\n",
    "model.add(tf.keras.layers.BatchNormalization())\n",
    "model.add(tf.keras.layers.Dropout(0.2))\n",
    "model.add(Dense(128,activation='relu'))\n",
    "model.add(tf.keras.layers.BatchNormalization())\n",
    "model.add(Dense(64,activation='relu'))\n",
    "model.add(tf.keras.layers.BatchNormalization())\n",
    "model.add(tf.keras.layers.Dropout(0.2))\n",
    "model.add(Dense(32,activation='relu'))\n",
    "model.add(tf.keras.layers.BatchNormalization())\n",
    "model.add(Dense(7,activation='softmax'))\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7815baa9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:`lr` is deprecated, please use `learning_rate` instead, or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0313 - accuracy: 0.9898WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 48s 182ms/step - loss: 0.0313 - accuracy: 0.9898 - val_loss: 0.1335 - val_accuracy: 0.9389\n",
      "Epoch 2/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0265 - accuracy: 0.9910WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 40s 169ms/step - loss: 0.0265 - accuracy: 0.9910 - val_loss: 0.4953 - val_accuracy: 0.8406\n",
      "Epoch 3/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0332 - accuracy: 0.9890WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 36s 155ms/step - loss: 0.0332 - accuracy: 0.9890 - val_loss: 0.1742 - val_accuracy: 0.9433\n",
      "Epoch 4/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0275 - accuracy: 0.9914WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 32s 138ms/step - loss: 0.0275 - accuracy: 0.9914 - val_loss: 0.2350 - val_accuracy: 0.9190\n",
      "Epoch 5/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0175 - accuracy: 0.9941WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 29s 125ms/step - loss: 0.0175 - accuracy: 0.9941 - val_loss: 0.1095 - val_accuracy: 0.9567\n",
      "Epoch 6/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0199 - accuracy: 0.9935WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 29s 125ms/step - loss: 0.0199 - accuracy: 0.9935 - val_loss: 0.0886 - val_accuracy: 0.9672\n",
      "Epoch 7/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0321 - accuracy: 0.9896WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 29s 125ms/step - loss: 0.0321 - accuracy: 0.9896 - val_loss: 0.4708 - val_accuracy: 0.8540\n",
      "Epoch 8/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0210 - accuracy: 0.9935WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 29s 124ms/step - loss: 0.0210 - accuracy: 0.9935 - val_loss: 0.4687 - val_accuracy: 0.8443\n",
      "Epoch 9/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0233 - accuracy: 0.9924WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 30s 129ms/step - loss: 0.0233 - accuracy: 0.9924 - val_loss: 0.0787 - val_accuracy: 0.9731\n",
      "Epoch 10/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0253 - accuracy: 0.9915WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 34s 145ms/step - loss: 0.0253 - accuracy: 0.9915 - val_loss: 0.2310 - val_accuracy: 0.9405\n",
      "Epoch 11/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0170 - accuracy: 0.9945WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 30s 126ms/step - loss: 0.0170 - accuracy: 0.9945 - val_loss: 0.0778 - val_accuracy: 0.9756\n",
      "Epoch 12/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0146 - accuracy: 0.9955WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 30s 127ms/step - loss: 0.0146 - accuracy: 0.9955 - val_loss: 0.0259 - val_accuracy: 0.9892\n",
      "Epoch 13/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0176 - accuracy: 0.9946WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 30s 128ms/step - loss: 0.0176 - accuracy: 0.9946 - val_loss: 0.2083 - val_accuracy: 0.9384\n",
      "Epoch 14/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0133 - accuracy: 0.9957WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 36s 154ms/step - loss: 0.0133 - accuracy: 0.9957 - val_loss: 0.0098 - val_accuracy: 0.9965\n",
      "Epoch 15/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0095 - accuracy: 0.9973WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 43s 181ms/step - loss: 0.0095 - accuracy: 0.9973 - val_loss: 0.1040 - val_accuracy: 0.9668\n",
      "Epoch 16/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0281 - accuracy: 0.9920WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 46s 194ms/step - loss: 0.0281 - accuracy: 0.9920 - val_loss: 0.0251 - val_accuracy: 0.9896\n",
      "Epoch 17/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0149 - accuracy: 0.9957WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 47s 199ms/step - loss: 0.0149 - accuracy: 0.9957 - val_loss: 0.2939 - val_accuracy: 0.9211\n",
      "Epoch 18/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0129 - accuracy: 0.9961WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 52s 223ms/step - loss: 0.0129 - accuracy: 0.9961 - val_loss: 0.7839 - val_accuracy: 0.8271\n",
      "Epoch 19/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0223 - accuracy: 0.9935WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 42s 180ms/step - loss: 0.0223 - accuracy: 0.9935 - val_loss: 0.0389 - val_accuracy: 0.9868\n",
      "Epoch 20/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0091 - accuracy: 0.9973WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 36s 152ms/step - loss: 0.0091 - accuracy: 0.9973 - val_loss: 0.0189 - val_accuracy: 0.9928\n",
      "Epoch 21/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0131 - accuracy: 0.9957WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 36s 152ms/step - loss: 0.0131 - accuracy: 0.9957 - val_loss: 0.4954 - val_accuracy: 0.8530\n",
      "Epoch 22/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0129 - accuracy: 0.9958WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 37s 156ms/step - loss: 0.0129 - accuracy: 0.9958 - val_loss: 0.0354 - val_accuracy: 0.9848\n",
      "Epoch 23/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0152 - accuracy: 0.9952WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 41s 174ms/step - loss: 0.0152 - accuracy: 0.9952 - val_loss: 0.2361 - val_accuracy: 0.9242\n",
      "Epoch 24/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0222 - accuracy: 0.9928WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 41s 173ms/step - loss: 0.0222 - accuracy: 0.9928 - val_loss: 0.1148 - val_accuracy: 0.9641\n",
      "Epoch 25/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0144 - accuracy: 0.9953WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 39s 166ms/step - loss: 0.0144 - accuracy: 0.9953 - val_loss: 0.2405 - val_accuracy: 0.9318\n",
      "Epoch 26/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0128 - accuracy: 0.9960WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 37s 159ms/step - loss: 0.0128 - accuracy: 0.9960 - val_loss: 0.1131 - val_accuracy: 0.9607\n",
      "Epoch 27/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0060 - accuracy: 0.9985WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 41s 174ms/step - loss: 0.0060 - accuracy: 0.9985 - val_loss: 0.0631 - val_accuracy: 0.9769\n",
      "Epoch 28/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0089 - accuracy: 0.9974WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 45s 193ms/step - loss: 0.0089 - accuracy: 0.9974 - val_loss: 0.0354 - val_accuracy: 0.9888\n",
      "Epoch 29/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0139 - accuracy: 0.9961WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 42s 177ms/step - loss: 0.0139 - accuracy: 0.9961 - val_loss: 0.2838 - val_accuracy: 0.9134\n",
      "Epoch 30/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0113 - accuracy: 0.9965WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 41s 175ms/step - loss: 0.0113 - accuracy: 0.9965 - val_loss: 0.1190 - val_accuracy: 0.9643\n",
      "Epoch 31/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0090 - accuracy: 0.9968WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 44s 186ms/step - loss: 0.0090 - accuracy: 0.9968 - val_loss: 0.0193 - val_accuracy: 0.9931\n",
      "Epoch 32/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0155 - accuracy: 0.9954WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 46s 196ms/step - loss: 0.0155 - accuracy: 0.9954 - val_loss: 0.3272 - val_accuracy: 0.9130\n",
      "Epoch 33/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0120 - accuracy: 0.9967WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 37s 159ms/step - loss: 0.0120 - accuracy: 0.9967 - val_loss: 0.0062 - val_accuracy: 0.9976\n",
      "Epoch 34/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0150 - accuracy: 0.9957WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 39s 166ms/step - loss: 0.0150 - accuracy: 0.9957 - val_loss: 0.2377 - val_accuracy: 0.9267\n",
      "Epoch 35/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0125 - accuracy: 0.9963WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 34s 143ms/step - loss: 0.0125 - accuracy: 0.9963 - val_loss: 0.1807 - val_accuracy: 0.9473\n",
      "Epoch 36/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0084 - accuracy: 0.9975WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 32s 137ms/step - loss: 0.0084 - accuracy: 0.9975 - val_loss: 0.5824 - val_accuracy: 0.8582\n",
      "Epoch 37/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0107 - accuracy: 0.9967WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 45s 190ms/step - loss: 0.0107 - accuracy: 0.9967 - val_loss: 1.2497 - val_accuracy: 0.7474\n",
      "Epoch 38/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0120 - accuracy: 0.9959WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 44s 187ms/step - loss: 0.0120 - accuracy: 0.9959 - val_loss: 0.1249 - val_accuracy: 0.9679\n",
      "Epoch 39/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0160 - accuracy: 0.9953WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 36s 154ms/step - loss: 0.0160 - accuracy: 0.9953 - val_loss: 0.0627 - val_accuracy: 0.9785\n",
      "Epoch 40/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0079 - accuracy: 0.9974WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 39s 166ms/step - loss: 0.0079 - accuracy: 0.9974 - val_loss: 0.0433 - val_accuracy: 0.9900\n",
      "Epoch 41/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0086 - accuracy: 0.9972WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 36s 155ms/step - loss: 0.0086 - accuracy: 0.9972 - val_loss: 0.0846 - val_accuracy: 0.9756\n",
      "Epoch 42/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0069 - accuracy: 0.9980WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 39s 164ms/step - loss: 0.0069 - accuracy: 0.9980 - val_loss: 0.0293 - val_accuracy: 0.9891\n",
      "Epoch 43/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0074 - accuracy: 0.9974WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 40s 170ms/step - loss: 0.0074 - accuracy: 0.9974 - val_loss: 0.4440 - val_accuracy: 0.8871\n",
      "Epoch 44/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0068 - accuracy: 0.9976WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 39s 165ms/step - loss: 0.0068 - accuracy: 0.9976 - val_loss: 0.0020 - val_accuracy: 0.9989\n",
      "Epoch 45/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0076 - accuracy: 0.9978WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 44s 186ms/step - loss: 0.0076 - accuracy: 0.9978 - val_loss: 0.2870 - val_accuracy: 0.9210\n",
      "Epoch 46/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0176 - accuracy: 0.9951WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 56s 236ms/step - loss: 0.0176 - accuracy: 0.9951 - val_loss: 0.0034 - val_accuracy: 0.9987\n",
      "Epoch 47/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0080 - accuracy: 0.9976WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 60s 254ms/step - loss: 0.0080 - accuracy: 0.9976 - val_loss: 0.0148 - val_accuracy: 0.9933\n",
      "Epoch 48/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0047 - accuracy: 0.9989WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 59s 253ms/step - loss: 0.0047 - accuracy: 0.9989 - val_loss: 3.2013 - val_accuracy: 0.5057\n",
      "Epoch 49/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0131 - accuracy: 0.9965WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "235/235 [==============================] - 48s 203ms/step - loss: 0.0131 - accuracy: 0.9965 - val_loss: 0.0782 - val_accuracy: 0.9764\n",
      "Epoch 50/50\n",
      "235/235 [==============================] - ETA: 0s - loss: 0.0081 - accuracy: 0.9978WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Can save best model only with val_acc available, skipping.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r",
      "235/235 [==============================] - 35s 149ms/step - loss: 0.0081 - accuracy: 0.9978 - val_loss: 0.0132 - val_accuracy: 0.9961\n"
     ]
    }
   ],
   "source": [
    "callback = tf.keras.callbacks.ModelCheckpoint(filepath='model2.h5',\n",
    "                                                  monitor='val_acc', mode='max',\n",
    "                                                 verbose=1, save_best_only=True)\n",
    "     \n",
    "\n",
    "optimizer=tf.keras.optimizers.Adam(lr=0.001)\n",
    "model.compile(loss = 'sparse_categorical_crossentropy',\n",
    "             optimizer =optimizer,\n",
    "              metrics = ['accuracy'])\n",
    "history = model.fit(x_train,\n",
    "                    y_train,\n",
    "                    validation_split=0.2,\n",
    "                    batch_size = 128,\n",
    "                    epochs = 50,\n",
    "                    shuffle=True,\n",
    "                    callbacks=[callback])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a38bb17d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.title('model accuracy')\n",
    "plt.ylabel('accuracy')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'val'], loc='upper left')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2684095d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'val'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c38c5ab1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "63/63 - 1s - loss: 1.5639 - accuracy: 0.7454 - 858ms/epoch - 14ms/step\n"
     ]
    }
   ],
   "source": [
    "model.save('model2.h5')\n",
    "model.load_weights('model2.h5')\n",
    "x_test=np.array(x_test).reshape(-1,28,28,3)\n",
    "loss, acc = model.evaluate(x_test, y_test, verbose=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "92a93a58",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "63/63 [==============================] - 1s 8ms/step\n",
      "[[  17   14   11    0    9    0    7]\n",
      " [  10   48    8    5   21    2   10]\n",
      " [   6    9   93    2   83    1   13]\n",
      " [   2    0    4    7    7    0    0]\n",
      " [   3   14   46    2 1232    1   49]\n",
      " [   0    2    2    0    9   25    0]\n",
      " [   7    9   31    0  110    1   71]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "y_pred = model.predict(x_test)\n",
    "y_pred = np.argmax(y_pred, axis=1)\n",
    "conf_mat = confusion_matrix(y_test, y_pred)\n",
    "print(conf_mat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "74f3e6c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "fig, ax = plt.subplots(figsize=(10,10)) \n",
    "ax = sns.heatmap(conf_mat, annot=True, cmap='Blues', linewidths=.9, ax=ax)\n",
    "ax.set_title('Confusion Matrix with labels\\n\\n');\n",
    "ax.set_xlabel('\\nPredicted Values')\n",
    "ax.set_ylabel('Actual Values ');\n",
    "ax.xaxis.set_ticklabels(['0','1','2','3','4','5','6'])\n",
    "ax.yaxis.set_ticklabels(['0','1','2','3','4','5','6'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "398932e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.38      0.29      0.33        58\n",
      "           1       0.50      0.46      0.48       104\n",
      "           2       0.48      0.45      0.46       207\n",
      "           3       0.44      0.35      0.39        20\n",
      "           4       0.84      0.91      0.87      1347\n",
      "           5       0.83      0.66      0.74        38\n",
      "           6       0.47      0.31      0.37       229\n",
      "\n",
      "    accuracy                           0.75      2003\n",
      "   macro avg       0.56      0.49      0.52      2003\n",
      "weighted avg       0.72      0.75      0.73      2003\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import sklearn\n",
    "print(sklearn.metrics.classification_report(y_test, y_pred))"
   ]
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

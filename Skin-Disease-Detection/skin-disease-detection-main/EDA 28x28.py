{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2fa65f27",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "from glob import glob\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d6cbb564",
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
       "      <th>lesion_id</th>\n",
       "      <th>image_id</th>\n",
       "      <th>dx</th>\n",
       "      <th>dx_type</th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>localization</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0027419</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0025030</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0026769</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0025661</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>HAM_0001466</td>\n",
       "      <td>ISIC_0031633</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>75.0</td>\n",
       "      <td>male</td>\n",
       "      <td>ear</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     lesion_id      image_id   dx dx_type   age   sex localization\n",
       "0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp\n",
       "1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp\n",
       "2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp\n",
       "3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp\n",
       "4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "skin_df = pd.read_csv('OneDrive/Desktop/skin dataset/HAM10000_metadata.csv') # load in the data\n",
    "skin_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c8083466",
   "metadata": {},
   "outputs": [],
   "source": [
    "lesion_type_dict = {\n",
    "    'nv': 'Melanocytic nevi',\n",
    "    'mel': 'Melanoma',\n",
    "    'bkl': 'Benign keratosis-like lesions ',\n",
    "    'bcc': 'Basal cell carcinoma',\n",
    "    'akiec': 'Actinic keratoses',\n",
    "    'vasc': 'Vascular lesions',\n",
    "    'df': 'Dermatofibroma'\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "84dee78f",
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
       "      <th>lesion_id</th>\n",
       "      <th>image_id</th>\n",
       "      <th>dx</th>\n",
       "      <th>dx_type</th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>localization</th>\n",
       "      <th>path</th>\n",
       "      <th>cell_type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0027419</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0027...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0025030</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0025...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0026769</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0026...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0025661</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0025...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>HAM_0001466</td>\n",
       "      <td>ISIC_0031633</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>75.0</td>\n",
       "      <td>male</td>\n",
       "      <td>ear</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0031...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     lesion_id      image_id   dx dx_type   age   sex localization  \\\n",
       "0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp   \n",
       "1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp   \n",
       "2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp   \n",
       "3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp   \n",
       "4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear   \n",
       "\n",
       "                                                path  \\\n",
       "0  OneDrive/Desktop/skin dataset/images/ISIC_0027...   \n",
       "1  OneDrive/Desktop/skin dataset/images/ISIC_0025...   \n",
       "2  OneDrive/Desktop/skin dataset/images/ISIC_0026...   \n",
       "3  OneDrive/Desktop/skin dataset/images/ISIC_0025...   \n",
       "4  OneDrive/Desktop/skin dataset/images/ISIC_0031...   \n",
       "\n",
       "                        cell_type  \n",
       "0  Benign keratosis-like lesions   \n",
       "1  Benign keratosis-like lesions   \n",
       "2  Benign keratosis-like lesions   \n",
       "3  Benign keratosis-like lesions   \n",
       "4  Benign keratosis-like lesions   "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "skin_df['path'] = 'OneDrive/Desktop/skin dataset/images/'+skin_df['image_id']+'.jpg'\n",
    "skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) \n",
    "skin_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a1d77388",
   "metadata": {},
   "outputs": [],
   "source": [
    "lesion_danger = {\n",
    "    'nv': 0, \n",
    "    'mel': 1, \n",
    "    'bkl': 0, \n",
    "    'bcc': 1, \n",
    "    'akiec': 1, \n",
    "    'vasc': 0,\n",
    "    'df': 0\n",
    "}\n",
    "# 0 for benign\n",
    "# 1 for malignant"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2e16e503",
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
       "      <th>lesion_id</th>\n",
       "      <th>image_id</th>\n",
       "      <th>dx</th>\n",
       "      <th>dx_type</th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>localization</th>\n",
       "      <th>path</th>\n",
       "      <th>cell_type</th>\n",
       "      <th>Malignant</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0027419</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0027...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>HAM_0000118</td>\n",
       "      <td>ISIC_0025030</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0025...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0026769</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0026...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>HAM_0002730</td>\n",
       "      <td>ISIC_0025661</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>80.0</td>\n",
       "      <td>male</td>\n",
       "      <td>scalp</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0025...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>HAM_0001466</td>\n",
       "      <td>ISIC_0031633</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>75.0</td>\n",
       "      <td>male</td>\n",
       "      <td>ear</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0031...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     lesion_id      image_id   dx dx_type   age   sex localization  \\\n",
       "0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp   \n",
       "1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp   \n",
       "2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp   \n",
       "3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp   \n",
       "4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear   \n",
       "\n",
       "                                                path  \\\n",
       "0  OneDrive/Desktop/skin dataset/images/ISIC_0027...   \n",
       "1  OneDrive/Desktop/skin dataset/images/ISIC_0025...   \n",
       "2  OneDrive/Desktop/skin dataset/images/ISIC_0026...   \n",
       "3  OneDrive/Desktop/skin dataset/images/ISIC_0025...   \n",
       "4  OneDrive/Desktop/skin dataset/images/ISIC_0031...   \n",
       "\n",
       "                        cell_type  Malignant  \n",
       "0  Benign keratosis-like lesions           0  \n",
       "1  Benign keratosis-like lesions           0  \n",
       "2  Benign keratosis-like lesions           0  \n",
       "3  Benign keratosis-like lesions           0  \n",
       "4  Benign keratosis-like lesions           0  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "skin_df[\"Malignant\"] = skin_df[\"dx\"].map(lesion_danger.get)\n",
    "skin_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "774a08d7",
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
       "      <th>lesion_id</th>\n",
       "      <th>image_id</th>\n",
       "      <th>dx</th>\n",
       "      <th>dx_type</th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>localization</th>\n",
       "      <th>path</th>\n",
       "      <th>cell_type</th>\n",
       "      <th>Malignant</th>\n",
       "      <th>cell_type_idx</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2469</th>\n",
       "      <td>HAM_0006834</td>\n",
       "      <td>ISIC_0029193</td>\n",
       "      <td>bcc</td>\n",
       "      <td>histo</td>\n",
       "      <td>85.0</td>\n",
       "      <td>male</td>\n",
       "      <td>face</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0029...</td>\n",
       "      <td>Basal cell carcinoma</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2458</th>\n",
       "      <td>HAM_0004413</td>\n",
       "      <td>ISIC_0026068</td>\n",
       "      <td>vasc</td>\n",
       "      <td>consensus</td>\n",
       "      <td>55.0</td>\n",
       "      <td>female</td>\n",
       "      <td>abdomen</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0026...</td>\n",
       "      <td>Vascular lesions</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>HAM_0002761</td>\n",
       "      <td>ISIC_0029068</td>\n",
       "      <td>bkl</td>\n",
       "      <td>histo</td>\n",
       "      <td>60.0</td>\n",
       "      <td>male</td>\n",
       "      <td>face</td>\n",
       "      <td>OneDrive/Desktop/skin dataset/images/ISIC_0029...</td>\n",
       "      <td>Benign keratosis-like lesions</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        lesion_id      image_id    dx    dx_type   age     sex localization  \\\n",
       "2469  HAM_0006834  ISIC_0029193   bcc      histo  85.0    male         face   \n",
       "2458  HAM_0004413  ISIC_0026068  vasc  consensus  55.0  female      abdomen   \n",
       "7     HAM_0002761  ISIC_0029068   bkl      histo  60.0    male         face   \n",
       "\n",
       "                                                   path  \\\n",
       "2469  OneDrive/Desktop/skin dataset/images/ISIC_0029...   \n",
       "2458  OneDrive/Desktop/skin dataset/images/ISIC_0026...   \n",
       "7     OneDrive/Desktop/skin dataset/images/ISIC_0029...   \n",
       "\n",
       "                           cell_type  Malignant  cell_type_idx  \n",
       "2469            Basal cell carcinoma          1              1  \n",
       "2458                Vascular lesions          0              6  \n",
       "7     Benign keratosis-like lesions           0              2  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "skin_df[\"cell_type_idx\"] = pd.Categorical(skin_df[\"cell_type\"]).codes # give each cell type a category id\n",
    "skin_df.sample(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "3e4d51c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:title={'center':'Benign vs Malignant'}>"
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
    "skin_df[\"Malignant\"].value_counts().plot(kind=\"bar\", title=\"Benign vs Malignant\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7cb734f4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:title={'center':'Counts for each type of Lesions'}>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax1 = plt.subplots(1,1,figsize=(10,5))\n",
    "skin_df[\"cell_type\"].value_counts().plot(kind=\"bar\", ax=ax1, title=\"Counts for each type of Lesions\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e9612848",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:title={'center':'Location of Lesions'}>"
      ]
     },
     "execution_count": 10,
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
    "skin_df[\"localization\"].value_counts().plot(kind='bar', title=\"Location of Lesions\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4d3ffef3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:title={'center':'Treatment received'}>"
      ]
     },
     "execution_count": 11,
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
    "skin_df[\"dx_type\"].value_counts().plot(kind='bar', title=\"Treatment received\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0ee9ef02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[188, 147, 191],\n",
       "        [186, 148, 189],\n",
       "        [187, 150, 191],\n",
       "        ...,\n",
       "        [196, 155, 171],\n",
       "        [197, 156, 170],\n",
       "        [197, 157, 168]],\n",
       "\n",
       "       [[186, 149, 193],\n",
       "        [187, 152, 194],\n",
       "        [189, 153, 191],\n",
       "        ...,\n",
       "        [194, 156, 169],\n",
       "        [195, 159, 169],\n",
       "        [192, 159, 168]],\n",
       "\n",
       "       [[185, 148, 192],\n",
       "        [189, 152, 195],\n",
       "        [190, 153, 196],\n",
       "        ...,\n",
       "        [196, 155, 169],\n",
       "        [198, 157, 171],\n",
       "        [194, 156, 169]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[157, 124, 155],\n",
       "        [156, 121, 154],\n",
       "        [159, 124, 154],\n",
       "        ...,\n",
       "        [177, 146, 161],\n",
       "        [176, 144, 159],\n",
       "        [175, 141, 155]],\n",
       "\n",
       "       [[155, 122, 151],\n",
       "        [156, 123, 154],\n",
       "        [156, 123, 152],\n",
       "        ...,\n",
       "        [178, 147, 163],\n",
       "        [175, 144, 159],\n",
       "        [175, 142, 159]],\n",
       "\n",
       "       [[154, 119, 151],\n",
       "        [153, 120, 149],\n",
       "        [154, 121, 152],\n",
       "        ...,\n",
       "        [176, 147, 167],\n",
       "        [175, 147, 161],\n",
       "        [173, 143, 155]]], dtype=uint8)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from skimage.io import imread\n",
    "skin_df[\"image\"] = skin_df[\"path\"].map(imread)\n",
    "skin_df.iloc[0][\"image\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4a742473",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(450, 600, 3)    10015\n",
       "Name: image, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "skin_df[\"image\"].map(lambda x: x.shape).value_counts() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f408a6fd",
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>2343</th>\n",
       "      <th>2344</th>\n",
       "      <th>2345</th>\n",
       "      <th>2346</th>\n",
       "      <th>2347</th>\n",
       "      <th>2348</th>\n",
       "      <th>2349</th>\n",
       "      <th>2350</th>\n",
       "      <th>2351</th>\n",
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
       "     0    1    2    3    4    5    6    7    8    9  ...  2343  2344  2345  \\\n",
       "0  192  153  193  195  155  192  197  154  185  202  ...   173   124   138   \n",
       "1   25   14   30   68   48   75  123   93  126  158  ...    60    39    55   \n",
       "2  192  138  153  200  145  163  201  142  160  206  ...   167   129   143   \n",
       "3   38   19   30   95   59   72  143  103  119  171  ...    44    26    36   \n",
       "4  158  113  139  194  144  174  215  162  191  225  ...   209   166   185   \n",
       "\n",
       "   2346  2347  2348  2349  2350  2351  label  \n",
       "0   183   147   166   185   154   177      2  \n",
       "1    25    14    28    25    14    27      2  \n",
       "2   159   124   142   136   104   117      2  \n",
       "3    25    12    17    25    12    15      2  \n",
       "4   172   135   149   109    78    92      2  \n",
       "\n",
       "[5 rows x 2353 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from PIL import Image\n",
    "reshaped_image = skin_df[\"path\"].map(lambda x: np.asarray(Image.open(x).resize((28,28), resample=Image.LANCZOS).\\\n",
    "                                                          convert(\"RGB\")).ravel())\n",
    "out_vec = np.stack(reshaped_image, 0)\n",
    "out_df = pd.DataFrame(out_vec)\n",
    "out_df[\"label\"] = skin_df[\"cell_type_idx\"]\n",
    "out_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "02f595ab",
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
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>...</th>\n",
       "      <th>2343</th>\n",
       "      <th>2344</th>\n",
       "      <th>2345</th>\n",
       "      <th>2346</th>\n",
       "      <th>2347</th>\n",
       "      <th>2348</th>\n",
       "      <th>2349</th>\n",
       "      <th>2350</th>\n",
       "      <th>2351</th>\n",
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
       "     0    1    2    3    4    5    6    7    8    9  ...  2343  2344  2345  \\\n",
       "0  192  153  193  195  155  192  197  154  185  202  ...   173   124   138   \n",
       "1   25   14   30   68   48   75  123   93  126  158  ...    60    39    55   \n",
       "2  192  138  153  200  145  163  201  142  160  206  ...   167   129   143   \n",
       "3   38   19   30   95   59   72  143  103  119  171  ...    44    26    36   \n",
       "4  158  113  139  194  144  174  215  162  191  225  ...   209   166   185   \n",
       "\n",
       "   2346  2347  2348  2349  2350  2351  label  \n",
       "0   183   147   166   185   154   177      2  \n",
       "1    25    14    28    25    14    27      2  \n",
       "2   159   124   142   136   104   117      2  \n",
       "3    25    12    17    25    12    15      2  \n",
       "4   172   135   149   109    78    92      2  \n",
       "\n",
       "[5 rows x 2353 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from PIL import Image\n",
    "reshaped_image = skin_df[\"path\"].map(lambda x: np.asarray(Image.open(x).resize((75,75), resample=Image.LANCZOS).\\\n",
    "                                                          convert(\"RGB\")).ravel())\n",
    "out_vec = np.stack(reshaped_image, 0)\n",
    "out_df = pd.DataFrame(out_vec)\n",
    "out_df[\"label\"] = skin_df[\"cell_type_idx\"]\n",
    "out_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "fc2652d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "out_path = \"hmnist_28x28_RBG.csv\"\n",
    "out_df.to_csv(out_path, index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a0deb974",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(600, 450)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "img = Image.open(skin_df[\"path\"][0])\n",
    "img.size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "540df474",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Benign keratosis-like lesions ', 'Melanocytic nevi',\n",
       "       'Dermatofibroma', 'Melanoma', 'Vascular lesions',\n",
       "       'Basal cell carcinoma', 'Actinic keratoses'], dtype=object)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "skin_df[\"cell_type\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fa0a17d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\chama\\anaconda3\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='label', ylabel='count'>"
      ]
     },
     "execution_count": 19,
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
    "sns.countplot(out_df['label'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "edd79ed8",
   "metadata": {},
   "outputs": [],
   "source": [
    "out_vec = out_vec.astype(\"float32\")\n",
    "out_vec /= 255\n",
    "labels = skin_df[\"cell_type_idx\"].values\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train_orig, X_test, y_train_orig, y_test = train_test_split(out_vec, labels, test_size=0.1,random_state=0)\n",
    "np.save(\"256_192_test.npy\", X_test)\n",
    "np.save(\"test_labels.npy\", y_test)\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.1, random_state=1)\n",
    "np.save(\"256_192_val.npy\", X_val)\n",
    "np.save(\"val_labels.npy\", y_val)\n",
    "np.save(\"256_192_train.npy\", X_train)\n",
    "np.save(\"train_labels.npy\", y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0dd474a7",
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

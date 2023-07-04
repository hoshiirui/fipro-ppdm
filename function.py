import pickle, joblib
from pathlib import Path
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, transform, util
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model

properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
sudut = ['0','45','90','135','180']
kolom = []
for prop in properties:
    for x in sudut:
        kolom.append(prop + "_" + x)


PATH = Path(__file__).parent
TMPDIR = PATH / 'tmp'
model_directory = os.path.join(PATH, 'model_rangga.pkl')
model = pickle.load(open('model_rangga.pkl', 'rb'))
# model = joblib.load('model_rangga.pkl')

# def save_files(file:str|None=None) -> str:
#     with open(os.path.join(TMPDIR, file.name), 'wb') as f:
#         f.write(file.getbuffer())
#     return file.name

# def del_file(path:os.PathLike=TMPDIR) -> None:
#     for file in os.listdir(path):
#         os.remove(os.path.join(path, file))

def load_image(file) -> np.ndarray:
    img = io.imread(file)
    img = color.rgb2gray(img)
    height, width = img.shape[:2]
    aspect = width/height

    fix_width = 100
    fix_height = int(fix_width/aspect)

    img = util.img_as_ubyte(img)
    return img, fix_height, fix_width

def glcm_feature(image, props, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], lvl=256, sym=True, norm=True):
    glcm = graycomatrix(image, distances=dists, angles=agls, levels=lvl,symmetric=sym, normed=norm)
    feature = []
    glcm_props = []
    for prop in props:
          for x in graycoprops(glcm, prop)[0]:
                glcm_props.append(x)
    for item in glcm_props:
            feature.append(item)

    return feature


def predict(file, treshold:int=0.5, properties=properties) -> str:
    features = []

    features.append(glcm_feature(file, props=properties))

    hasil = pd.DataFrame(features, columns=kolom)

    fitur_tes = hasil.to_numpy()

    prediction = model.predict(fitur_tes).flatten()
    if(prediction > treshold):
        return 'Happy'
    else:
        return 'Sad'
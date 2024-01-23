import numpy as np 
import matplotlib.pyplot as plt
import pydicom as dicom
import os
import re
import math 
import cv2
from scipy import ndimage
from skimage import morphology
from scipy.ndimage import label, find_objects, generate_binary_structure
from scipy.ndimage.measurements import sum as ndi_sum

################## Traitement des données DICOM #########################
def extract_number(filename):
    """
    Extracts the last number from a filename, returns 0 if no number is found.
    """
    matches = re.findall(r'(\d+)', filename)
    return int(matches[-1]) if matches else 0

def get_slice_location(dicom_file):
    """
    Returns the slice location from a DICOM file.
    """
    ds = dicom.dcmread(dicom_file)
    return float(ds.SliceLocation)

def extract_dicom(folder_path = "./THORAX_EP/"):
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_files.sort(key=get_slice_location)
    img = [dicom.dcmread(f).pixel_array for f in dicom_files]
    img = np.array(img)
    return img

def update_dicom_with_new_data(new_data, original_folder = "./THORAX_EP_1/", new_folder = "./test/"):
    """ Attention les nouvelles données doivent être des entiers """
    if np.max(new_data) <=1:
        new_data = np.clip(new_data * 1700, 0, 1700).astype(np.uint16)
    if new_data.dtype != np.uint16:
        new_data.astype(np.uint16)
    dicom_files = [f for f in os.listdir(original_folder) if f.endswith('.dcm')]
    dicom_files.sort(key=lambda x: get_slice_location(os.path.join(original_folder, x)))

    for i, file_name in enumerate(dicom_files):
        # Lire le fichier DICOM original
        ds = dicom.dcmread(os.path.join(original_folder, file_name))

        # Mettre à jour les données de l'image avec la slice correspondante de new_data
        ds.PixelData = new_data[i].tobytes()

        # Sauvegarder le fichier DICOM mis à jour dans le nouveau dossier
        new_file_path = os.path.join(new_folder, file_name)
        ds.save_as(new_file_path)


def extract_dicom_and_windowing(folder_path = "./THORAX_EP_1/"):
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_files.sort(key=get_slice_location)
    
    img = []
    for file_path in dicom_files:
        ds = dicom.dcmread(file_path)
        pixel_array = ds.pixel_array
        
        # Appliquer Rescale Intercept et Rescale Slope
        if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

        # Appliquer le Window Center et Window Width si disponible
        if 'WindowCenter' in ds and 'WindowWidth' in ds:
            window_center = ds.WindowCenter
            window_width = ds.WindowWidth
            if isinstance(window_center, dicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, dicom.multival.MultiValue):
                window_width = window_width[0]

            pixel_array = apply_windowing(pixel_array, window_center, window_width)
        
        img.append(pixel_array)
    
    img = np.array(img)
    return img

def dicom_data_extract(file_dicom = "THORAX_EP_1/1_THORAX_EP_126536115341126212123511_img_0.dcm"):
    ds = dicom.dcmread(file_dicom)

    # Afficher toutes les métadonnées
    for tag in ds.dir():
        try:
            data_element = ds[tag]
            print(f"{tag}: {data_element.value}")
        except KeyError:
            pass

#########################################################

################# Traitement images #####################
"""Fenêtre pulmonaire :

WL : -600 à -700 HU (unités Hounsfield)
WW : 1500 à 2000 HU
Utilisation : Idéale pour visualiser les structures pulmonaires, les vaisseaux dans les poumons, et les pathologies telles que les embolies pulmonaires.
Fenêtre médiastinale :

WL : 30 à 60 HU
WW : 350 à 500 HU
Utilisation : Convient pour évaluer les structures médiastinales, y compris le cœur, les gros vaisseaux, et les ganglions lymphatiques.
Fenêtre osseuse :

WL : 300 à 500 HU
WW : 1500 à 3000 HU
Utilisation : Utilisée pour examiner les os et détecter des fractures ou d'autres pathologies osseuses.
Fenêtre cérébrale :

WL : 20 à 40 HU
WW : 70 à 100 HU
Utilisation : Spécifique pour l'examen du cerveau, permettant de distinguer le tissu cérébral, les saignements, et les lésions.
Fenêtre pour tissus mous :

WL : 40 à 60 HU
WW : 350 à 400 HU
Utilisation : Optimale pour observer les tissus mous du corps, y compris les muscles et les organes internes.
Fenêtre abdominale :

WL : 40 à 60 HU
WW : 350 à 400 HU
Utilisation : Similaire à la fenêtre pour tissus mous, mais axée sur l'abdomen."""

def apply_windowing(ct_slice, window_level, window_width):
    """
    Apply windowing to a CT slice.

    Parameters:
    ct_slice (numpy array): The CT slice as a 2D numpy array.
    window_level (int): The window level (WL) in Hounsfield units (HU).
    window_width (int): The window width (WW) in Hounsfield units (HU).

    Returns:
    numpy array: The windowed CT slice.
    """
    # # # Calculate window boundaries
    # lower_bound = window_level - (window_width / 2)
    # upper_bound = window_level + (window_width / 2)

    # # Apply windowing
    # windowed_slice = np.clip(ct_slice, lower_bound, upper_bound)

    # # Normalize to range [0, 1] for display purposes
    # windowed_slice = (windowed_slice - lower_bound) / window_width
    img_min = window_level - window_width // 2
    img_max = window_level + window_width // 2
    windowed_slice = np.clip(ct_slice, img_min, img_max)
    windowed_slice = (windowed_slice - img_min) / (img_max - img_min)
    return windowed_slice


def binarize(img, threshold=0.1):
    img[img <= threshold] = 0
    img[img > threshold] = 1
    return img

def invert(img):
    return 1 - img


def selected_particles(close, structure = 1, min_size = 500):
    s = generate_binary_structure(3, structure)

    # Effectuer l'analyse des composantes connectées
    labeled_volume, num_features = label(close, structure=s)

    # Calculer le volume de chaque particule
    volumes = ndi_sum(close, labeled_volume, index=range(1, num_features + 1))

    # Filtrer les particules en fonction de leur taille
    max_size = np.inf # Taille maximale des particules à conserver

    result = {}
    for particle_label, volume in enumerate(volumes, start=0):
        result[particle_label] = volume
        # print(f'Particule {particle_label} : {volume} voxels')


    indices = np.argsort(volumes)[-10:][::-1]
    particles = []
    for i in indices:
        mask = np.zeros_like(close)
        mask[labeled_volume == i + 1] = 1
        particles.append(mask)
    particles = np.array(particles)

    return particles


########################################################

#################### Utils #############################
def plot(img, a=200, b=250, c=273):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(img[:, :, a], cmap="gray")
    ax[1].imshow(img[:, b, :], cmap="gray")
    ax[2].imshow(img[c, :, :], cmap="gray") 

def hist_without_background(img):
    non_background_mask = img != 0

    # Filtrage des données pour exclure le background
    filtered_data = img[non_background_mask]

    # Création de l'histogramme
    plt.hist(filtered_data.ravel(), bins=200, color='blue', alpha=0.7)
    plt.title('Distribution des valeurs (sans background)')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.show()

#########################################################
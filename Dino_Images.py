import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import warnings
import math

from Images import Images

'''
Object for storing image data for dino files
    :attr num_images: Number of images
    :attr image_list: The list of images in ndarray
    :attr dic_path: The path to folder of the data
    :attr image_path: The path to all images under dic_path
    :attr camera_matrix: 2darray which stores (K, R, T matrix) where
        camera_matrix[n] is the nth camera matrix (In flatten form)
'''

class Dino_Images(Images):


    def __init__(self, input_dic_path="dinoSparseRing"):

        # Check user input
        assert os.path.isdir(input_dic_path) and not os.path.isfile(
            input_dic_path), \
            "Folder does not exist"

        dir = os.listdir(input_dic_path)
        assert len(dir) != 0, "Folder is empty"

        # Record path to data dic
        self.dic_path = input_dic_path

        # Read dino camera info
        self.num_images, self.image_path, self.camera_matrix = \
            self.parse_par("/dinoSR_par.txt", self.input_validation_dino_sparse)

        # Read dino images
        self.image_list = self.parse_images()

    '''
    Parse file format from dinosaur
        Input: 
            input_validator: Input file validator 
            input_file_path: Path to camera meta file
        Output: 
            num_images: Number of images included in this dataset
            image_path: The local path to images
            camera_matrix: The data matrix correlated to the dataset (K, R, T matrix)
    '''

    def parse_par(self, file_name, input_validator):

        f = open(self.dic_path + file_name, 'r')
        num_images = int(f.readline())
        data = f.read()

        lines = data.split('\n')
        data_array_str, image_path = [], []

        for i in range(num_images):
            line = lines[i].split(' ')
            # first word is image path, not needed right now, so start at index 1
            image_path.append(line[0])
            data_array_str.append(line[1:])

        data_array = [list(map(float, data_array_str[i])) for i in
                      range(len(data_array_str))]

        # 16 images, 9 parameters for calibration,
        # 9 parameters for rotation,
        # 3 parameters for translation
        # total of 21 parameters
        input_validator(num_images, np.array(data_array))

        return num_images, image_path, np.array(data_array)

    '''
    Parse file format from dinosaur
        Output: 
            image_list: The list of images in ndarray
    '''

    def parse_images(self):

        # Read image
        image_list = []
        for f in self.image_path:
            path = self.dic_path + "/" + f
            if os.path.splitext(path)[1] in self.EXTENSION_LIST:
                image_list.append(image.imread(path))

        # Make sure it aligns
        assert len(image_list) == self.num_images

        return image_list

    '''
    Validate file format from dinosaur dataset
        Input: 
            num_images: Number of images included in this dataset
            camera_matrix: The data matrix correlated to the dataset (K, R, T matrix)
        Output: 
            Error if failed to validate dino dataset
    '''

    def input_validation_dino_sparse(self, num_images, camera_matrix):

        # FOR DINO SPARSE SET
        assert (camera_matrix.shape == (16, 21))

        # CHECK IF THIS IS THE RIGHT FLATTENING
        center_calibration_inv = np.linalg.inv(
            camera_matrix[0, :9].reshape(3, 3))

        # CHECK IF THIS IS THE RIGHT FLATTENING
        center_rotation_inv = np.linalg.inv(
            camera_matrix[0, 9:18].reshape(3, 3))

        center_t = camera_matrix[0, 18:]
        centered_calibration_matrices = np.zeros((num_images, 3, 3))
        centered_rotation_matrices = np.zeros((num_images, 3, 3))
        centered_translation_vectors = np.zeros((num_images, 3))
        for i in range(num_images):
            centered_calibration_matrices[i] = center_calibration_inv @ (
                camera_matrix[i, :9]).reshape(3, 3)
            centered_rotation_matrices[i] = center_rotation_inv @ (
                camera_matrix[i, 9:18]).reshape(3, 3)
            centered_translation_vectors[i] = camera_matrix[i, 18:] - center_t

        # print(centered_calibration_matrices[0])
        # print(centered_rotation_matrices[0])

        # errors in np.lin.alg.inv, so just check that they are close to identity
        assert (np.linalg.norm(
            centered_calibration_matrices[0] - np.identity(3) < 0.00001))
        assert (np.linalg.norm(
            centered_rotation_matrices[0] - np.identity(3)) < 0.00001)
        assert (np.linalg.norm(
            centered_translation_vectors[0] - np.zeros((1, 3))) < 0.0001)

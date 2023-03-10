'''
Author(s): Sijie Xu (s362xu), Leon Yao(lclyao)
    This file handles file loading from dino data
'''

import matplotlib.image as image
import numpy as np
import os

from mylibs.Images import Images

'''
Object for storing image data for dino files
    :attr num_images: Number of images
    :attr image_list: The list of images in ndarray
    :attr dic_path: The path to folder of the data
    :attr image_path: The path to all images under dic_path
    :attr K_Matrix: Camera calibra matrix in 3darray where K_Matrix[n] 
        is a 3x3 matrix correlated to the nth images
    :attr R_Matrix: Camera rotation matrix in 3darray where R_Matrix[n] 
        is a 3x3 matrix correlated to the nth images
    :attr T_Matrix: Camera translation matrix in 3darray where T_Matrix[n] 
        is a 3x1 vector correlated to the nth images
    :attr params: The Camera matrix in a dictionary (blame on Mike)
'''
class Dino_Images(Images):

    def __init__(self, input_dic_path="dinoSparseRing", par_path = "/dinoSR_par.txt"):

        # Check user input
        assert os.path.isdir(input_dic_path) and not os.path.isfile(
            input_dic_path), \
            "Folder does not exist"

        dir = os.listdir(input_dic_path)
        assert len(dir) != 0, "Folder is empty"

        # Record path to data dic
        self.dic_path = input_dic_path

        self.num_images, self.image_path, self.K_Matrix, self.R_Matrix, \
        self.T_Matrix, self.params = self.parse_par(par_path)

        # Read dino images
        self.image_list = self.parse_images()
        self.sequential_image_order = self.find_sequence(self.R_Matrix, self.T_Matrix)

    '''
    Parse file format from dinosaur
        Input: 
            input_validator: Input file validator 
            input_file_path: Path to camera meta file
        Output: 
            num_images: Number of images included in this dataset
            image_path: The local path to images
            K_Matrix, R_Matrix, T_Matrix: The data matrix correlated to the dataset 
    '''
    def parse_par(self, file_name):

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

        # 9 parameters for calibration,
        # 9 parameters for rotation,
        # 3 parameters for translation
        # total of 21 parameters
        camera_matrix = np.array(data_array)

        calibration_matrices = np.zeros((num_images, 3, 3))
        rotation_matrices = np.zeros((num_images, 3, 3))
        translation_vectors = np.zeros((num_images, 3))

        param_dict = {}
        for i in range(num_images):

            K = camera_matrix[i, :9].reshape(3,3)
            R = camera_matrix[i, 9:18].reshape(3,3)
            T = camera_matrix[i, 18:].reshape(3,1)
            param_dict[i] = [K, R, T]
            calibration_matrices[i] = camera_matrix[i, :9].reshape(3, 3)
            rotation_matrices[i] = camera_matrix[i, 9:18].reshape(3, 3)
            translation_vectors[i] = camera_matrix[i, 18:]

        return num_images, image_path, calibration_matrices, rotation_matrices, translation_vectors, param_dict

    '''
    Parse file format from dinosaur
        Output: 
            image_list: The list of images in ndarray
    '''
    def find_sequence(self, rotations, translations):
        '''
            Naive method to find the sequence of cameras views we will use to reconstruct the object. The first camera will be
            the one at index 0, and every subsequent camera will be the one closest in position to the last camera in the current order
        '''
        # assuming len(rotations) == len(translations)
        remaining = [i for i in range(1, len(rotations))]

        current_order = [0]

        # super slow method, prob betters
        while len(remaining) > 0:
            curr_index = current_order[-1]
            curr_translation = translations[curr_index]
            curr_rotation = rotations[curr_index]
            closest_index = remaining[0]
            diff = np.sum(np.square(curr_translation - translations[closest_index])) + np.sum(np.square(curr_rotation - rotations[closest_index]))

            # iterate over remaining cameras and miminize difference in position
            for j in range(1, len(remaining)):
                test_index = remaining[j]
                test_translation = translations[test_index]
                test_rotation = rotations[test_index]
                test_diff = np.sum(np.square(test_translation - curr_translation)) + np.sum(np.square(test_rotation - curr_rotation))
                if test_diff < diff:
                    closest_index = test_index
                    diff = test_diff
            remaining.remove(closest_index)
            current_order.append(closest_index)
        return current_order

    '''
    Parse file format from dinosaur
        Output: 
            image_list: A list of ordering of how the pair wise comparisons are done
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



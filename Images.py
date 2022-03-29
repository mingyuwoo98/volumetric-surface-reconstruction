import numpy as np
from abc import ABC, abstractmethod, abstractproperty


class Images(ABC):
    '''
        An abstract class that represents input data format

        :attr EXTENSION_LIST (List(str)): The acceptable image file extensions
        '''

    EXTENSION_LIST = [".jpg", ".png", ".JPG", ".PNG"]

    @abstractmethod
    def parse_par(self, file_name):
        '''
        Provides a method to par the camera matrix
        '''
        pass

    @abstractmethod
    def parse_images(self):
        '''
        Provides a method to par the images
        '''
        pass

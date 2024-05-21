# Authors: Melisa Mete (150200316),
#          Öykü Eren (150200326),
#          Bora Boyacıoğlu (150200310)

"""
Custom class for images data.

Extra methods:
    - apply(func): apply a function to all images in the dictionary.
    - read(data_dir): read images from a directory.
"""

# Import necessary libraries.
import os
import cv2

class Images():
    def __init__(self, images=None):
        self.images = images if images else {}
        self.length = None
        self.landmarks = {}

    def apply(self, func, *args, **kwargs):        
        """ Apply a function to all images in one of the dictionaries. """
        obj = {}
        for i, (name, photos) in enumerate(self.images.items()):
            print(f"\rApplying {func.__name__}... {100 * (i + 1) / self.length:.2f}%", end="")
            obj[name] = [
                func(*args, image=img, **kwargs)
                for img in photos
            ]
        print("")
        return obj
    
    def read(self, data_dir):
        """ Read images from a directory. """
        
        for i, name in enumerate(os.listdir(data_dir)):
            print(f"\rReading images... {100 * (i + 1) / len(os.listdir(data_dir)):.2f}%", end="")
            name_path = os.path.join(data_dir, name)
            
            if not os.path.isdir(name_path):
                continue
            
            self.images[name] = [
                cv2.imread(os.path.join(name_path, img))
                for img in os.listdir(name_path)
            ]
        print("")
        
        self.length = len(self.images)

import pickle
import cv2
import os

from utils import get_image_paths
from utils import face_encodings


root_dir = "C:/Users/Bishal/OneDrive/Desktop/Face-Recognization/dataset"
class_names = os.listdir(root_dir)

# get the paths to the images
image_paths = get_image_paths(root_dir, class_names)
# initialize a dictionary to store the name of each person and the corresponding encodings
name_encondings_dict = {}

# initialize the number of images processed
nb_current_image = 1
# now we can loop over the image paths, locate the faces, and encode them
for image_path in image_paths:
    print(f"Image processed {nb_current_image}/{len(image_paths)}")
    # load the image
    image = cv2.imread(image_path)
    # get the face embeddings
    encodings = face_encodings(image)
    # get the name from the image path
    name = image_path.split(os.path.sep)[-2]
    '''[-2] retrieves the second-to-last item in the list. In the example above, it retrieves the string "person1". This is because we assume that the directory structure is as follows: root_dir/person_name/image.jpg. So, the second-to-last item in the list will always be the name of the person.'''
    # get the encodings for the current name
    e = name_encondings_dict.get(name, [])
    # update the list of encodings for the current name
    e.extend(encodings) #e+encoding
    # update the list of encodings for the current name
    name_encondings_dict[name] = e
    nb_current_image += 1

# save the name encodings dictionary to disk
with open("encodings.pickle", "wb") as f:
    pickle.dump(name_encondings_dict, f)

    '''pickle is a Python module that is used for serializing and de-serializing Python objects. It is used to convert a Python object hierarchy into a byte stream that can be stored in a file, database or transmitted over a network. The pickle.dump() method serializes the Python object name_encondings_dict into a byte stream and writes it to a file named "encodings.pickle" in binary mode (hence the "wb" mode).'''
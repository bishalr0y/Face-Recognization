import dlib
from glob import glob
import cv2
import numpy as np
import os

# The face
# detector (face_detector) is used to detect the face regions in the input image.

# The face encoder model (face_encoder) is used to generate the face embeddings.
# load the face detector, landmark predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
    "C:/Users/Bishal/OneDrive/Desktop/Face-Recognization/models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(
    "C:/Users/Bishal/OneDrive/Desktop/Face-Recognization/models/dlib_face_recognition_resnet_model_v1.dat")

# change this to include other image formats you want to support (e.g. .webp)
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def get_image_paths(root_dir, class_names):
    """ grab the paths to the images in our dataset"""
    image_paths = []

    # ********************** to get the file path ************************
    # loop over the class names
    for class_name in class_names:
        # grab the paths to the files in the current class directory
        class_dir = os.path.sep.join([root_dir, class_name])
        # The glob() function returns a list of paths to the files in the current class directory.
        #     We loop over the file paths and extract the file extension of the current file.
        #         If the file extension is not valid (not an image), we will skip it and continue to the next one.
        class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))

        # loop over the file paths in the current class directory
        for file_path in class_file_paths:
            # extract the file extension of the current file
            ext = os.path.splitext(file_path)[1]

            # if the file extension is not in the valid extensions list, ignore the file
            if ext.lower() not in VALID_EXTENSIONS:
                print("Skipping file: {}".format(file_path))
                continue

            # add the path to the current image to the list of image paths
            image_paths.append(file_path)

    return image_paths


def face_rects(image):
    # convert the image to grayscale
    # The cv2.cvtColor() function from the OpenCV library is used to convert the input image from one color space to another.
    #  Specifically, in this code snippet, it is used to convert the input image from the BGR (Blue, Green, Red) color space to grayscale.
    #  BGR color space to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = face_detector(gray, 1)
    # return the bounding boxes
    return rects


def face_landmarks(image):
    # The landmark predictor (shape_predictor) is used to localize the facial landmarks in the face region.
    # We will need the facial landmarks to generate the face embeddings.
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]
    # The function first calls the face_landmarks function to obtain the facial landmark points for each face in the image. It then uses a pre-trained face encoding model, represented by the face_encoder object, to compute a 128-dimensional vector that describes each face in the image.


def face_encodings(image):
    '''The function uses a list comprehension to apply the compute_face_descriptor function to each face in the input image, using the facial landmark points returned by face_landmarks(image). The resulting list of 128-dimensional vectors represents the facial embeddings for each face in the image.'''
    # compute the facial embeddings for each face
    # in the input image. the compute_face_descriptor
    # function returns a 128-d vector that describes the face in an image
    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark))
            for face_landmark in face_landmarks(image)]


def nb_of_matches(known_encodings, unknown_encoding):
    # compute the euclidean distance between the current face encoding
    # and all the face encodings in the database
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= 0.4  # 0.6 initial value
    # print(small_distances)
    # return the number of matches
    return sum(small_distances)

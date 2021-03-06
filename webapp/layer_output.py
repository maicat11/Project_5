from keras.models import Model
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import pandas as pd


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def get_image_tensor(img_path):
    """Gets the image tensor from the path"""
    img_tensor = path_to_tensor(img_path) / 255.0
    return img_tensor


# def get_dense_layer(model, img_path):
#     """Get calculations at the requested layer in CNN"""
#     img_tensor = get_image_tensor(img_path)

#     # Creates a model that will return these outputs, given the model input
#     activation_model = Model(inputs=model.input, outputs=model.get_layer('vectors').output)

#     # Returns a list of Numpy arrays: one array per layer activation
#     activation = activation_model.predict(img_tensor)

#     return activation[0]

#
# def get_dense_layers(model, paths_list):
#     """Returns a matrix of all calculations in list of paths
#     and a dataframe with the path as the index."""
#
#     input_tensors = [get_image_tensor(filename) for filename in paths_list]
#     activation_model = Model(inputs=model.input, outputs=model.get_layer('vectors').output)
#     output = activation_model.predict(np.array(input_tensors).reshape(-1, 224, 224, 3))
#     outputs_df = pd.DataFrame(output, index=paths_list)
#     return output, outputs_df
#

def get_dense_layers(model, paths_list):
    """Returns a matrix of all calculations in list of paths
    and a dataframe with the path as the index."""
    input_tensors = [get_image_tensor(filename) for filename in paths_list]
    output =  get_dense_layers_from_image(model, input_tensors)
    outputs_df = pd.DataFrame(output, index=paths_list)
    return output, outputs_df

def get_dense_layers_from_image(model, input_tensors):
    activation_model = Model(inputs=model.input, outputs=model.get_layer('vectors').output)
    output = activation_model.predict(np.array(input_tensors).reshape(-1, 224, 224, 3))
    return output

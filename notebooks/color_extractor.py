#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import PIL


def get_canonical_set():
    """Returns list of colors from the XKCD color survey.

    Returns all named colors (900++) and the associated color names"""
    canonical = pd.read_csv('https://xkcd.com/color/rgb.txt', sep='\t', skiprows=1, header=None)
    canonical['r'] = canonical.loc[:,1].apply(lambda x: int(x[1:3], 16))
    canonical['g'] = canonical.loc[:,1].apply(lambda x: int(x[3:5], 16))
    canonical['b'] = canonical.loc[:,1].apply(lambda x: int(x[5:7], 16))

    return canonical

def get_important_canonical_set():
    """Returns a list of "important" colors.

    From https://blog.xkcd.com/2010/05/03/color-survey-results/, some of the
    colors were indicated as names most people used, instead of the more 
    niche color names.
    """
    # These are taken from https://blog.xkcd.com/2010/05/03/color-survey-results/
#     important = ['purple', 'green', 'blue', 'pink', 'red', 'light blue', 'teal', 'coffee', 
#                  'magenta', 'yellow', 'grey', 'violet', 'turquoise', 'lavender', 'orange',
#                  'dark blue', 'tan', 'olive green', 'beige', 'royal blue', 'black', 'hot pink', 
#                  'light brown', 'lime', 'indigo', 'mustard', 'dark purple', 'sea green', 
#                 'olive green', 'light pink', 'light green', 'dark pink', 'bright green', 
#                 'salmon', 'pale green', 'peach', 'lilac','periwinkle', 'lime green', 
#                 'light purple','maroon', 'cyan',  'mauve', 'aqua', 'dark purple', 
#                 'dark green', 'forest green', 'navy blue', 'sky blue']
    
    # important colors for shoe clustering 
    important = ['brick red', 'grape', 'beige', 'charcoal', 
                 'taupe', 'brownish', 'light grey', 'slate']

    short_canonical = get_canonical_set().set_index(0).loc[important]
    return short_canonical.iloc[:, 2:]


short_canonical=get_important_canonical_set()
short_canonical

color_nn = NearestNeighbors(n_neighbors=1)
color_nn.fit(short_canonical)


def median_pixel_from_name(name, cutoff=240):
    """Gets the average (median) value of a pixel from an image.

    Inputs

    name: filename of image
    cutoff: if the average (median) value of a pixel's channel is greater than
            cutoff, mask the pixel's value.

    Returns

    A numpy array with the median of the average non-masked pixel value in each
    channel. e.g. An RGB image will return the median R, median G, and median B
    in a single numpy array.
    """
    image = PIL.Image.open(name)
    return median_pixel(image, cutoff)

def median_pixel(image, cutoff=240):
    imgdata = np.array(image.resize((20,20)).getdata()).astype(float)
    pixels_to_include = (imgdata.mean(axis=1) < cutoff)
    avg_color = np.median(imgdata[pixels_to_include], axis = 0)
    return avg_color

def classify_image(name, cutoff=240):
    sample = [median_pixel_from_name(name, cutoff)]
    _, index =  color_nn.kneighbors(sample)
    index = index.reshape(1,)
    return index[0], short_canonical.index[index[0]]


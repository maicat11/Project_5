from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras.models import load_model
from keras import backend
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from visuals import view_nn_images, view_one_image
from layer_output import path_to_tensor, get_dense_layers_from_image
import os
import importlib


if backend.backend() != 'theano':
    os.environ['KERAS_BACKEND'] = 'theano'
    importlib.reload(backend)

# get saved clothing model
clothing_to_vector_model = load_model('../notebooks/saved_models/clothes2_cnn.h5')
clothing_to_vector_model.load_weights('../notebooks/saved_models/weights.best.from_scratch.hdf5')

# get saved dress mapped to shoe model
dress_to_shoe = load_model('../notebooks/saved_models/dress2shoe_aws.h5')
dress_to_shoe.load_weights('../notebooks/saved_models/dress2shoe_weights_aws.hdf5')

# get saved shoe vectors for all images
shoe_df = pd.read_pickle('../notebooks/saved_models/dense_shoe_df_aws.pickle')

# get saved nearest neighbors
# top_nn_shoe = NearestNeighbors(n_neighbors=4, metric='cosine').fit(shoe_df)
top_nn_shoe = NearestNeighbors(n_neighbors=4, metric='euclidean').fit(shoe_df)

def get_recommendations(dress_image,
                        nn_model=top_nn_shoe,
                        clothing_to_vector_model=clothing_to_vector_model,
                        dress_to_shoe=dress_to_shoe):

    """Returns a dataframe that contains the paths to the recommended shoes
    dress_image: pass in the image as a tensor
    nn_model: fitted nearest neighbor model
    clothing_to_vector_model: CNN model for clothing training loaded with weights from h5 file
    dress_to_shoe: CNN model for links loaded with weights from h5 file
    get_dense_layers_from_image: imported from layer_output.py
    """

    dress_vector = get_dense_layers_from_image(clothing_to_vector_model, dress_image)
    shoe_vector  = dress_to_shoe.predict(dress_vector)
    distances, indicies = nn_model.kneighbors(-shoe_vector)
    return pd.DataFrame(shoe_df.iloc[indicies.reshape(-1)].index).T


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    print("inside upload function")
    if request.method == 'POST':
        f = request.files['file']
        filename = f'static/uploads/{secure_filename(f.filename)}'
        print(filename)
        f.save(filename)

    tensor = path_to_tensor(filename)
    # get_recommendations(tensor,nn_model=nn_top_shoe)

    df = get_recommendations(tensor, nn_model = top_nn_shoe)
    shoes = df.apply(lambda x: x.tolist(), axis=1)[0]
    shoes = [x.replace('data', 'static') for x in shoes]
    return render_template("recommends2.html", file=filename, shoes=shoes)


@app.route("/recommends", methods=["GET", "POST"])
def recommends():
    return render_template("recommends2.html", name=image, items=items)

if __name__ == '__main__':
    app.run(debug = True)

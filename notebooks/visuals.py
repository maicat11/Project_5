import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def view_nn_images(df, indices):
    plt.figure(figsize=(10,10))

    plt.subplot(2,2,1)
    im = mpimg.imread(str(df.iloc[indices][0]))
    plt.imshow(im);
    plt.title('First Item')
    plt.axis('off')

    plt.subplot(2,2,2)
    im = mpimg.imread(str(df.iloc[indices][1]))
    plt.imshow(im);
    plt.title('Second Item')
    plt.axis('off')

    plt.subplot(2,2,3)
    im = mpimg.imread(str(df.iloc[indices][2]))
    plt.imshow(im);
    plt.title('Third Item')
    plt.axis('off')

    plt.subplot(2,2,4)
    im = mpimg.imread(str(df.iloc[indices][3]))
    plt.imshow(im);
    plt.title('Fourth Item')
    plt.axis('off');


def view_images(path1, path2):
    plt.figure(figsize=(10,10))

    plt.subplot(1,2,1)
    im = mpimg.imread(path1)
    plt.imshow(im);
    plt.title('Clothing Item')
    plt.axis('off')

    plt.subplot(1,2,2)
    im = mpimg.imread(path2)
    plt.imshow(im);
    plt.title('Shoe Item')
    plt.axis('off');
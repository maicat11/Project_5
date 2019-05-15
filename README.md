# "KEANU"

## Personalized Shoe Recommender from a Fashion Image

**Metis Capstone Project**

Mailei Vargas

March 2019



### Project Motivation

Many companies now have recommendation systems built into their product that make the user shopping experience more streamlined and pleasant.  These recommendations are largly based on prior purchases or on purchases of other users that are "similar".  In the case of the suggestions that are made based on prior purchases, these are great if it is an item like shampoo which needs repeat purchases over time.  However, this doesn't work well when the item is a one time purchase.  The example that comes to mind is the woman who keep receiving recommendations for a tiolet seat because she once made that purchase.  "Dear Amazon, I bought a toilet seat because I needed one. Necessity, not desire. I do not collect them. I am not a toilet seat addict. No matter how temptingly you email me, I am not going think, oh go on then, just one more toilet seat, I'll treat myself." - Jac Rayner

The case where recommendations are based on "like" users that have made similar purchases to me, this works much better if the items are everyday use items that don't require me to make a decision based on "style" or "comfort".  Here in lies the problem of recmmendation systems for items that are seasonal, revolving, and personally unique…fashion items.  A person's sense of fashion is largely based on personal preferences and whatever is popular at the time.  So for this project, I made an attempt of creating a product that would recommend shoes to me based on what I told it I liked and disliked as well as some other "smartness" under the hood such as matching color.  

### Workflow

For my recommendation system, I collected 4500+ images from Google API and 50k Zappos.  Removed the "noisy" images in which either the background was not plain, or the image may not have fit into the category it was labeled as.  For example, Google search keyword "womens skinny blue jeans" and some of the images returned are of denim shorts not pants. Also more than half of the Zappos images were removed because they contained kids and mens shoes.  

Once the data was "cleaned" and organized into categories the next steps were building a convolutional neural network (CNN) in order to train the model to correctly classify the images as the category they belonged to.  This was done by building an architechture of 12 layers in which I could feed in both the clothing images and the shoe images.  Below is the layers and architecture. 

![image-20190327115339055](/Users/maileivargas/Library/Application%20Support/typora-user-images/image-20190327115339055.png)

  

I found an achitecture very similar in a tutorial I did from "Deep Learning using Python", and made a few tweaks trying the get an understanding of what is going on.  When the classification for the clothing came back as 80% accuracy, I did not feel the need to attempt transfer learning since classification of clothing is not my end result.  I felt that 80% was good enough.  Below is the convergence of the training and validation set. 

![image-20190327120513015](/Users/maileivargas/Library/Application%20Support/typora-user-images/image-20190327120513015.png)

**Color addition**

As for the images of shoes, initially these were classifying at 85% accuracy based on 7 categories (sandal, ankle boot, etc.), but a decision was made later in the project to add a color component since many of the clothing items were differentiated this way (blue jeans, black dress, etc.). SO it made sense to also categorize shoe by 8 colors and 7 styles which inflated the categories to 56.  The resulted in the accuracy dropping to 70%, but again good enough since this was not the final result of the project.  

![image-20190327114057037](/Users/maileivargas/Library/Application%20Support/typora-user-images/image-20190327114057037.png)

​                     This shoe above initially was categorized as *sandal*, but later based on the RGB 

​                     average of the of the image, its nearest color neighbor was *charcoal*, a color I 

​                     chose as a representation of black.  



Once I had the training completed for each image, I extracted the dense layer called *vectors* (as highlighted above) and stored them in a dataframe with the path to the image as the unique identifier.  Each vector had length of 300 output and these are the features of the images at the layer prior to the prediction (not accounting for the dropout).      

![image-20190327120831507](/Users/maileivargas/Library/Application%20Support/typora-user-images/image-20190327120831507.png)

![image-20190327120844883](/Users/maileivargas/Library/Application%20Support/typora-user-images/image-20190327120844883.png)




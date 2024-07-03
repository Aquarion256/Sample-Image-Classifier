import pickle
from fastai.vision.all import *



loaded_model= pickle.load(open('final_model.pkl','rb'))
print(type(loaded_model))

img = load_image("5.jpg")
print(loaded_model.predict(img)[0])

#---------------------------------------------------------------------------------------------------------
# Replace the above paths with the path to the model that has been saved and the path to any sample image.
#---------------------------------------------------------------------------------------------------------
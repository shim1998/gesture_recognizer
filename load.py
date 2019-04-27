import numpy as np
import keras.models
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf

def init():
    json_file = open('models/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #loaded_model = load_model("models/model.h5")
    #loaded weights into new models
    loaded_model.load_weights("models/model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded models
    loaded_model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:',loss)
    #print('accuracy',accuracy)
    graph = tf.get_default_graph()
    return loaded_model,graph

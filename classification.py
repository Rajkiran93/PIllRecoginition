import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


def load_graph(frozen_graph_filename, graph_def):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def classification(UPLOAD_PATH):
    print("@ classification",UPLOAD_PATH)
    IMG_SIZE = 256
    image = cv2.imread(UPLOAD_PATH)
    gray = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    roi = gray / 255.0
    batch = np.array(roi).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    model_graph = tf.Graph()
    model_file = "/Users/cumulations/Desktop/Pillrecognitiondemo/model/model.pb"
#    printTensors(model_file)
    with model_graph.as_default():
        model_graph = load_graph(model_file, model_graph)
        x = model_graph.get_tensor_by_name('prefix/conv2d_3_input:0')
        y = model_graph.get_tensor_by_name('prefix/dense_3/Softmax:0')
        with tf.Session(graph=model_graph) as sess:
            result = np.argmax(sess.run(y, feed_dict={x: batch}))    
    if result == 0:
        result = "Abilify 10"
    elif result == 1:
        result = "Abilify 15"
    elif result == 2:
        result = "Abilify 20"
    elif result == 3:
        result = "Abilify 30"
    elif result == 4:
        result = "Almox D"
    elif result == 5:
        result = "Aspirin"
    elif result == 6:
        result = "Dolo"
    elif result == 7:
        result = "Eldoper"
    elif result == 8:
        result = "Glucomust"
    elif result == 9:
        result = "GlutaonD"
    elif result == 10:
        result = "Paramet Black"
    elif result == 11:
        result = "Wellbutrin 100"
    elif result == 12:
        result = "Wellbutrin 150"
    else:
        result = "Wellbutrin 200"
    return result

def printTensors(pb_file):
    
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    
    # print operations
    for op in graph.get_operations():
        print(op.name)


def classifyImages():
    path = input("enter path to your file..")
#    path = "/Users/cumulations/Desktop/Others/Personal\ skills/Pill\ recognition\ demo/TestImages/ab15.png"
    print("printing the path:",path)
    predictedImage = classification(path)
    print("predictedImage is ->",predictedImage)
    return


if __name__ == '__main__':
    classifyImages()


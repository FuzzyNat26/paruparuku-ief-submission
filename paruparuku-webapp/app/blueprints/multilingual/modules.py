import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm

class GradCAM:
    def get_img_array(img_path, size):
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)

        return array

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]

        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        return heatmap.numpy()

    def find_target_layer(model):
        for layer in reversed(model.layers):
            if len(layer.output.shape) == 4 and not layer.name == "average_pooling2d":
                return layer.name

    def superimpose_gradcam(img_path, heatmap, alpha=0.4):
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)
        
        # RESCALE HEATMAP TO A RANGE OF 0-255
        heatmap = np.uint8(255 * heatmap)

        # USE JET COLORMAP 
        jet = cm.get_cmap("jet")

        # USE RGB VALUES
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # CREATE IMAGE WITH RGB COLORIZED HEATMAP
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # SUPERIMPOSE THE HEATMAP ON ORIGINAL IMAGE
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        return superimposed_img

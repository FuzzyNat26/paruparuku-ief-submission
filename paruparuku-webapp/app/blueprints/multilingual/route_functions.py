# IMPORT LIBRARIES
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from app import app

import io
import numpy as np
from base64 import b64encode
from .modules import GradCAM

# SETUP VARIABLES
image_data_generator = ImageDataGenerator(rescale=1./255)

# VARIABLES DETECT PNEUMONIA MODEL
densenet201 = models.load_model("app/models/DenseNet201_pneumonia.model")
mobilenetv2 = models.load_model("app/models/MobileNetV2_pneumonia.model")
nasnetmobile = models.load_model("app/models/NASNetMobile_pneumonia.model")

# VARIABLES DETECT XRAY MODEL
mnv2_detect_xray_model = models.load_model("app/models/MobileNetV2_detect_xray.model")

# SETUP MAIN FUNCTION
def allowed_file(filename):
    return filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def pneumoniaDetection(image_name, path):
    # TEMPORARY IMAGE DATA GEN
    temp_data_gen = image_data_generator.flow_from_directory(
        batch_size = 1,
        directory = 'app/static/',
        classes = ['uploaded_file'],
        shuffle = False, 
        target_size = (224, 224),
        class_mode = "binary",
    )

    # USER IMAGE DATA GEN
    user_image_path = "uploaded_file/" + image_name
    temp_image = []

    for image, name in zip(temp_data_gen, temp_data_gen.filenames):
        if name == user_image_path:
            temp_image.append(image)
            temp_image = temp_image[0][0]
            break

    # CHECK IF IT IS A X-RAY IMAGE
    check_xray_image = is_an_xray_image(temp_image)

    if check_xray_image is False:
        return False, False

    # PREDICTION
    densenet201_pred = round(densenet201.predict(temp_image)[0][0], 2)
    mobilenetv2_pred = round(mobilenetv2.predict(temp_image)[0][0], 2)
    nasnetmobile_pred = round(nasnetmobile.predict(temp_image)[0][0], 2)

    # NORMAL PROBABILITY AVERAGE
    normal_probability_average = round(((densenet201_pred + mobilenetv2_pred + nasnetmobile_pred) / 3), 2)

    # COMPUTE HEAT MAP
    # INITIATE MODEL LIST FOR LOOPING
    models = [
        densenet201,
        mobilenetv2,
        nasnetmobile
    ]

    # TEMPORARY EMPTY ARRAY TO STORE OUTPUT
    CAM_images = []

    # GENERATE GRADCAM IMAGES
    for model in models:
        last_conv_layer_name = GradCAM.find_target_layer(model)
        heatmap = GradCAM.make_gradcam_heatmap(temp_image, model, last_conv_layer_name)

        img = GradCAM.superimpose_gradcam(path, heatmap)

        img = img.convert('RGB')
        data = io.BytesIO()
        img.save(data, "JPEG")

        encoded = b64encode(data.getvalue())
        decoded_image = encoded.decode('utf-8')

        CAM_images.append(decoded_image)

    # SUMMARIZE RESULT
    prediction_results = [
        normal_probability_average * 100,
        densenet201_pred * 100,
        mobilenetv2_pred * 100,
        nasnetmobile_pred * 100
    ]

    return prediction_results, CAM_images

def is_an_xray_image(temp_image):
    # RANDOM IMAGE PROBABILITY
    mobilenetv2_pred = round(mnv2_detect_xray_model.predict(temp_image)[0][0], 2)
    random_probability_average = round(mobilenetv2_pred, 2)

    return random_probability_average < 0.5
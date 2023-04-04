# IMPORT LIBRARIES
import io, os, datetime

from flask import Flask, render_template, request, redirect, url_for, Blueprint, g, current_app, abort
from PIL import Image
from base64 import b64encode
from werkzeug.utils import secure_filename

from app import app
from app.blueprints.multilingual.route_functions import pneumoniaDetection, allowed_file, is_an_xray_image

# BLUEPRINT SETUP
multilingual = Blueprint('multilingual', __name__,
                         template_folder='templates', url_prefix='/<lang_code>')

@multilingual.url_defaults
def add_language_code(endpoint, values):
    values.setdefault('lang_code', g.lang_code)

@multilingual.url_value_preprocessor
def pull_lang_code(endpoint, values):
    g.lang_code = values.pop('lang_code')

@multilingual.before_request
def before_request():
    if g.lang_code not in current_app.config['LANGUAGES']:
        adapter = app.url_map.bind('')

        try:
            endpoint, args = adapter.match(
                '/en' + request.full_path.rstrip('/ ?'))
            
            return redirect(url_for(endpoint, **args), 301)
        except:
            abort(404)

    dfl = request.url_rule.defaults
    if 'lang_code' in dfl:
        if dfl['lang_code'] != request.full_path.split('/')[1]:
            abort(404)


# SETUP MAIN WEB ROUTES
@multilingual.route('/')
def main():
    return render_template(
        'multilingual/index.html',
        lang_code = g.lang_code
        )

@multilingual.route('/sumber', defaults={'lang_code': 'id'})
@multilingual.route('/sources', defaults={'lang_code': 'en'})
@multilingual.route('/來源', defaults={'lang_code': 'zh'})
@multilingual.route('/情報源', defaults={'lang_code': 'ja'})
def sources():
    return render_template(
        "multilingual/sources.html",
        lang_code = g.lang_code
        )

@multilingual.route('/tentang-kami', defaults={'lang_code': 'id'})
@multilingual.route('/about-us', defaults={'lang_code': 'en'})
@multilingual.route('/關於我們', defaults={'lang_code': 'zh'})
@multilingual.route('/私たちについて', defaults={'lang_code': 'ja'})
def about_us():
    return render_template(
        "multilingual/about-us.html", 
        page_name="about_us",
        lang_code = g.lang_code
        )

@multilingual.route('/tujuan-kami', defaults={'lang_code': 'id'})
@multilingual.route('/our-objectives', defaults={'lang_code': 'en'})
@multilingual.route('/我們的目標', defaults={'lang_code': 'zh'})
@multilingual.route('/私たちの目的', defaults={'lang_code': 'ja'})
def our_objectives():
    return render_template(
        "multilingual/our-objectives.html",
        page_name="our_objectives",
        lang_code = g.lang_code
        )

@multilingual.route('/deteksi', defaults={'lang_code': 'id'}, methods=['POST', 'GET'])
@multilingual.route('/detection', defaults={'lang_code': 'en'}, methods=['POST', 'GET'])
@multilingual.route('/檢測', defaults={'lang_code': 'zh'}, methods=['POST', 'GET'])
@multilingual.route('/検出', defaults={'lang_code': 'ja'}, methods=['POST', 'GET'])
def detection():
    # HANDLE REQUEST
    if request.method == 'POST':
        # CHECK WHETHER THE REQUEST HAS THE DESIRED INPUT
        if 'user_image_input' not in request.files:
            return redirect(request.url)

        file = request.files['user_image_input']

        # CHECK IF IMAGE WAS UPLOADED
        if file.filename == '':
            return redirect(request.url)

        # CHECK IF FILE TYPE IS AN IMAGE
        if file and allowed_file(file.filename):
            filename = str(datetime.datetime.now()) + "_" + secure_filename(file.filename)
            
            # SAVING FILE
            COMBINED_PATH = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            file.save(COMBINED_PATH)

            # TEMPORARY IMAGE
            img = Image.open(COMBINED_PATH)
            img = img.convert('RGB')
            data = io.BytesIO()
            img.save(data, "JPEG")

            encoded = b64encode(data.getvalue())
            temp_img = encoded.decode('utf-8')

            # DETECTION AND REMOVE IMAGE
            prediction_results, CAM_images = pneumoniaDetection(image_name = filename, path = COMBINED_PATH)            
            os.remove(COMBINED_PATH)

            # IF IT IS AN X-RAY
            if prediction_results is not False:
                return render_template(
                    "multilingual/result.html",
                    prediction_results = prediction_results, 
                    image_file = temp_img,
                    CAM_images = CAM_images,
                    lang_code = g.lang_code
                    )
            # IF IT'S NOT AN X-RAY
            else:
                return render_template(
                    "multilingual/not-x-ray-image.html",
                    image_file = temp_img,
                    lang_code = g.lang_code
                    )

    
    # JIKA FILE EXTENSION BUKAN IMAGE
    return render_template(
        "multilingual/wrong-extension.html",
        lang_code = g.lang_code
        )

@app.errorhandler(500)
def internal_server_error(e):
    return render_template(
        "multilingual/error500.html",
        lang_code = g.lang_code
        )

@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "multilingual/error404.html",
        )
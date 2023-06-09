from flask import Flask, request, g, redirect, url_for, abort
from flask_babel import Babel
from config import Config

# SETUP APPLICATION
app = Flask(__name__)
app.config.from_object(Config)

# import and register blueprints
from app.blueprints.multilingual import multilingual
app.register_blueprint(multilingual)

# set up babel
babel = Babel(app)


@babel.localeselector
def get_locale():
    if not g.get('lang_code', None):
        g.lang_code = app.config['LANGUAGES'][0]
    return g.lang_code


@app.route('/')
def home():
    if not g.get('lang_code', None):
        get_locale()
    return redirect(url_for('multilingual.main'))

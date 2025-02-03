from flask import Flask
from app.api import api

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['COLORIZED_FOLDER'] = 'static/colorized'

app.register_blueprint(api)

if __name__ == '__main__':
    app.run(debug=True)

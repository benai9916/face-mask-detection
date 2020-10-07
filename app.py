import os
import cv2
from flask import Flask, Response, render_template, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
from web_cam import mask_checker_web_cam
from web_image import mask_checker_img

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "static/upload")
DETECT_FOLDER = os.path.join(os.getcwd(), "static/detect")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER

# CLEAR the image
def delete_img(img_paths):
	paths = os.listdir(img_paths)
	if len(paths) > 1:
  		for i in paths:
  			os.remove(os.path.join(img_paths, i))


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/detect_mask', methods = ['POST', 'GET'])
def detect_mask():
	delete_img(os.path.join(os.getcwd(), 'faces/input'))
	delete_img(os.path.join(os.getcwd(), 'faces/with_mask'))
	delete_img(os.path.join(os.getcwd(), 'faces/without_mask'))

	if request.form['webCam'] == 'for_web_cam':
		web_cam_mask =  Response(mask_checker_web_cam(),
	                    mimetype='multipart/x-mixed-replace; boundary=frame')

		return web_cam_mask


@app.route('/mask_on_image', methods = ['POST', 'GET'])
def mask_on_image():
	if request.method == 'POST':
		delete_img(os.path.join(os.getcwd(), 'static/detect'))
		delete_img(os.path.join(os.getcwd(), 'static/upload'))
		delete_img(os.path.join(os.getcwd(), 'faces/input'))
		delete_img(os.path.join(os.getcwd(), 'faces/with_mask'))
		delete_img(os.path.join(os.getcwd(), 'faces/without_mask'))

		if request.form['on_img'] == 'for_img':
			f = request.files['file']
			filename = secure_filename(f.filename)

			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			# print(filepath)
			f.save(filepath)

			color_img, pred = mask_checker_img(filepath)

			final_pred = 0

			if pred == 1:
				final_pred = 1
			elif pred == 99:
				final_pred = 99
			else:
				final_pred = 0

			print('----------', final_pred)

			img = Image.fromarray(color_img)
			img.save(os.path.join(DETECT_FOLDER, filename))

			return render_template('index.html', fname = filename, final_pred = final_pred)


if __name__ == '__main__':
    app.run(debug=True)
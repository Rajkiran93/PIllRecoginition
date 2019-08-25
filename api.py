from flask import Flask, request
import os
import time
from classification import classification


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
key = "image"
UPLOAD_DIR = "uploads"


@app.route("/upload", methods=["POST"])
def upload():
    start = time.time()
	if not os.path.isdir(UPLOAD_DIR):
		os.mkdir(UPLOAD_DIR)

	for file in request.files.getlist(key):
		UPLOAD_PATH = os.path.join(APP_ROOT, UPLOAD_DIR, file.filename)
		file.save(UPLOAD_PATH)
		result = classification(UPLOAD_PATH)
    print(time.time()-start)
	return result

if __name__ == "__main__":
	app.run(host="0.0.0.0", debug=False, port=5001)

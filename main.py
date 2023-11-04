import os
from flask import Flask, request, jsonify
import requests
import json

from detect import detect
from similarity import predict
from get_image import image_downloader

from flask_cors import CORS

import ssl

app = Flask(__name__)
CORS(app)

os.makedirs("./get_image", exist_ok=True)

@app.route('/', methods=['GET','POST'])
def test():
    data = request.data.decode('utf-8')

    get_data = json.loads(data)["id"]

    image_downloader("image_dual", get_data)

    detect_data = detect("./get_image/test.jpg",3)
    list_data = predict(detect_data)
    json_data = json.dumps(list_data)

    # json_data = json_data.replace('"', "")
    # json_data = json_data.replace('[', "")
    # json_data = json_data.replace(']', "")

    # json_data = json.dumps(json_data)

    print(json_data)
    
    return jsonify(json_data)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5555)
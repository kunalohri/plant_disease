from flask import Flask ,render_template ,request
from keras.models import model_from_json
from resizeimage import resizeimage
import numpy as np
from PIL import Image
from numpy import argmax

app=Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/",methods=["POST"])
def img_upload():
    pred=-1

    if request.method == "POST":

        if request.files:
            try:
                img = request.files['image']
                img.save("img.png")
                imgTest = Image.open("img.png")
                imgTest = resizeimage.resize_cover(imgTest, [256, 256, 3])
                imgTest = np.array(imgTest, dtype='uint8')
                ip = imgTest.reshape(1, 256, 256, 3)
                json_file = open('CNN3.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights("CNN3.h5")
                pred=loaded_model.predict(ip)
                pred = argmax(pred, axis=1, out=None)[0]

                result=''
                if pred == 0:
                    result="Healthy"
                elif pred==1:
                    result="Powdery"
                elif pred==2:
                    result="Rust"
            except:
                result="Upload Image to Get Result"



    #python file -> html
    return render_template("index.html",op=result)

if __name__ == "__main__":
    app.run(debug=True)
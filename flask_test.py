import base64
import numpy as np
import io
from PIL import Image
import cv2
import keras_ocr

import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import render_template
from flask_ngrok import run_with_ngrok

app=Flask(__name__)
CORS(app)

#run_with_ngrok(app)



#app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def startWeb():
    return render_template('hello.html')

@app.route("/hello",methods=["GET","POST"])
def hello():
    
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    
    image=Image.open(io.BytesIO(decoded))
   
    #img=image.save("ocr.jpg"); 
    #print(img)
    #img=keras_ocr.tools.read("C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python36\\ocr.jpg")
    im=np.array(image)
    img=keras_ocr.tools.read(im)
    pipeline = keras_ocr.pipeline.Pipeline()
    predictions = pipeline.recognize(images=[img])[0]
    #drawn = keras_ocr.tools.drawBoxes(image=img, boxes=predictions, boxes_format='predictions')
    x= ([text for text, box in predictions])
    index=0
    pred=[]
    for i in range(0,6):
      if(x[i].isalpha() and len(x[i])==4):
        pred.append(x[i].upper())
        index=i
  
    for i in range(0,6):
      if(len(x[i])>=5 and x[i].isnumeric()):
        pred.append(x[i])

    for i in range(0,6):
      if(len(x[i])==1):
        pred.append(x[i].upper())
    for i in range(0,6):
      if(len(x[i])==4 and i!=index):
        pred.append(x[i].upper())
        
    
    
    


    ans=pred[0]+pred[1]+pred[2]+pred[3]
    
    
    
                    

    response={'prediction':ans}
    
    return jsonify(response)

if __name__=="__main__":
    app.run()
    #app.run(debug=True)
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port)

    

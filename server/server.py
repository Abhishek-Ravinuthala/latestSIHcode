from flask import Flask, request, jsonify, session

from flask_cors import CORS
import easyocr
import cv2
import os
import numpy as np

import bcrypt
from flask_session import Session
import tensorflow as tf
import pickle
import cv2
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import joblib
from tensorflow.keras.models import load_model
from googletrans import Translator

# def translate_to_english(text, source_language='auto'):
    
#     return translation.text

# Example usage:



reader = easyocr.Reader(['en', 'hi'])
leader = easyocr.Reader(['te', 'en'])
feader = easyocr.Reader(['ja', 'en'])
zeader = easyocr.Reader(['ko', 'en'])

# Get the absolute path to the saved model file
# model_file_path = os.path.abspath('model.sav')
# app instance

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)


# mongodb


# db = client["crop-damage-users"]
# collection = db["users"]
# fs = GridFS(db)
# app.config['SECRET_KEY'] = 'charanpics'

# # Configure session type to use server-side sessions
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['SESSION_COOKIE_PATH'] = '/'
# app.config['SESSION_COOKIE_SAMESITE'] = 'None'
# app.config['SESSION_COOKIE_SECURE'] = True
# Session(app)
# print(collection)

# # /api/home


@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Like this video if this helped!",
        'people': ['Jack', 'Harry', 'Arpan']
    })

# Flask route to check session status


@app.route('/api/submit_register_form', methods=['POST'])
def submit_register_form():
    try:
        name = request.form['name']
        email = request.form['email']
        # phone = request.form['phone']
        # name = request.form['name']
        password = request.form['password']

        # image_file = request.files['image']
        # if image_file:
        #     # Save the image to GridFS and get the file ID
        #     file_id = fs.put(image_file.stream, filename=image_file.filename)

        # Store other data and the image file ID in MongoDB
        # Hash the password
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt())

# Store other data and the hashed password in MongoDB
        model_data = {
            'name': name,
            'email': email,
            # 'phone': phone,
            # 'name': name,
            # Store the hashed password as a string
            'password': hashed_password.decode('utf-8')
        }

        model_id = db.models.insert_one(model_data).inserted_id

        # Example: You can send a response back to the client with the model ID
        response_data = {
            'message': 'Form data received and image stored successfully',
            'model_id': str(model_id)
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': e}), 500


@app.route('/api/check_session', methods=['GET'])
def check_session():
    if session.get('authenticated'):
        # User is authenticated
        return jsonify({'authenticated': True, 'name': session['name']}), 200
    else:
        # User is not authenticated
        return jsonify({'authenticated': False}), 200


@app.route('/api/submit_login_form', methods=['POST'])
def submit_login_form():
    try:

        name = request.form['name']
        entered_password = request.form['password']

        stored_hashed_password = db.models.find_one(
            {'name': name})['password']
        if bcrypt.checkpw(entered_password.encode('utf-8'), stored_hashed_password.encode('utf-8')):

            session['authenticated'] = True
            session['name'] = name

            response_data = {
                'message': session['name'],
            }

            return jsonify(response_data), 200

        else:
            # Invalid credentials
            response_data = {
                'error': 'Invalid credentials',
            }
            return jsonify(response_data), 401

        # Example: You can send a response back to the client with the model ID

    except Exception as e:
        return jsonify({'error': e}), 500


@app.route('/api/predict', methods=['POST'])
def predict():

    # if session.get('authenticated'):
    #     return jsonify({'message': 'Session is active'}), 200
    
    requested_image = request.files['image']
    model_path = os.path.join('latestdigitsnew.h5')

    image_path = 'uploads/' + requested_image.filename
    requested_image.save(image_path)

    try:

        img = tf.io.read_file(image_path)

        # plt.imshow(resize.numpy().astype(int))
        # plt.show()

        knn_from_joblib = load_model('latestdigitsnew.h5')
        # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
        # load_model = pickle.load(open('model.sav', 'rb'))
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, [28, 28])

    # Scale the tensor
        img = img / 255
        img = img[:, :, :1]

        yhat = knn_from_joblib.predict(np.expand_dims(img, axis=0))
        classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        classnames = np.array(classnames)
        yes = classnames[yhat[0].argmax()]

        # if os.path.exists(image_path):
        #     os.remove(image_path)
        return jsonify(yes)

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/api/hindi', methods=['POST'])
def hindi():

    # if session.get('authenticated'):
    #     return jsonify({'message': 'Session is active'}), 200
    
   

    
    requested_image = request.files['image']
    model_path = os.path.join('devanagari_model.h5')

    image_path = 'uploads/' + requested_image.filename
    requested_image.save(image_path)

    try:

        img = tf.io.read_file(image_path)

        # plt.imshow(resize.numpy().astype(int))
        # plt.show()

        knn_from_joblib = load_model('devanagari_model.h5')
        # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
        # load_model = pickle.load(open('model.sav', 'rb'))
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, [32, 32])

    # Scale the tensor
        img = img / 255
        img = img[:, :, :1]

        # yhat = knn_from_joblib.predict(np.expand_dims(img, axis=0))
        # # classnames = ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]
        
        # labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']

        # classnames = np.array(labels)
        # yes = classnames[yhat[0].argmax()]

        # if os.path.exists(image_path):
        #     os.remove(image_path)
        # return jsonify(yes)
        from matplotlib import pyplot as plt
        result = reader.readtext(image_path,paragraph=True)

        
         # Example text in French
        
        
        top_left = tuple(result[0][0][0])
        bottom_right = tuple(result[0][0][2])
        text = result[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        

        img = cv2.imread(image_path)
        spacer = 300
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection in result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            #img = cv2.putText(img,text,(20,spacer), font, 0.5,(255,255,0),2,cv2.LINE_AA)
            spacer+=15
        # plt.figure(figsize=(10,10))
        # plt.imshow(img)
        # plt.show()

        for i in result:
            text =  (i[1])
        translator = Translator()
        translation = translator.translate(text, src='hi', dest='en')
        translated_text = translation.text
        return jsonify(text, translated_text)
        

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/api/english', methods=['POST'])
def english():

    # if session.get('authenticated'):
    #     return jsonify({'message': 'Session is active'}), 200
    
    requested_image = request.files['image']
    model_path = os.path.join('english.h5')

    image_path = 'uploads/' + requested_image.filename
    requested_image.save(image_path)

    try:

        img = tf.io.read_file(image_path)

        # plt.imshow(resize.numpy().astype(int))
        # plt.show()

        knn_from_joblib = load_model('english.h5')
        # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
        # load_model = pickle.load(open('model.sav', 'rb'))
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, [28, 28])

    # Scale the tensor
        # img = img / 255
        # img = img[:, :, :1]

        from matplotlib import pyplot as plt
        result = reader.readtext(image_path,paragraph=True)

        

        
        top_left = tuple(result[0][0][0])
        bottom_right = tuple(result[0][0][2])
        text = result[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        

        img = cv2.imread(image_path)
        spacer = 300
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection in result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            #img = cv2.putText(img,text,(20,spacer), font, 0.5,(255,255,0),2,cv2.LINE_AA)
            spacer+=15
        # plt.figure(figsize=(10,10))
        # plt.imshow(img)
        # plt.show()

        for i in result:
            return jsonify(i[1])
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/korean', methods=['POST'])
def korean():

    # if session.get('authenticated'):
    #     return jsonify({'message': 'Session is active'}), 200
    
    requested_image = request.files['image']
    # model_path = os.path.join('hangulkorea.h5')

    image_path = 'uploads/' + requested_image.filename
    requested_image.save(image_path)

    try:

        img = tf.io.read_file(image_path)

        # plt.imshow(resize.numpy().astype(int))
        # plt.show()

        # knn_from_joblib = load_model('hangulkorea.h5')
        # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
        # load_model = pickle.load(open('model.sav', 'rb'))
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, [64, 64])

    # Scale the tensor
        img = img / 255
        img = img[:, :, :1]


        from matplotlib import pyplot as plt
        result = zeader.readtext(image_path,paragraph=True)

        

        
        top_left = tuple(result[0][0][0])
        bottom_right = tuple(result[0][0][2])
        text = result[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        

        img = cv2.imread(image_path)
        spacer = 300
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection in result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            #img = cv2.putText(img,text,(20,spacer), font, 0.5,(255,255,0),2,cv2.LINE_AA)
            spacer+=15
        # plt.figure(figsize=(10,10))
        # plt.imshow(img)
        # plt.show()

        for i in result:
            text =  (i[1])
        translator = Translator()
        translation = translator.translate(text, src='ko', dest='en')
        translated_text = translation.text
        return jsonify(text, translated_text)
        


    except Exception as e:
        return jsonify({'error': str(e)})







@app.route('/api/japanese', methods=['POST'])
def japanese():

    # if session.get('authenticated'):
    #     return jsonify({'message': 'Session is active'}), 200
    
    requested_image = request.files['image']
    model_path = os.path.join('cnn_hiragana.h5')

    image_path = 'uploads/' + requested_image.filename
    requested_image.save(image_path)

    try:

        img = tf.io.read_file(image_path)

        # plt.imshow(resize.numpy().astype(int))
        # plt.show()

        # knn_from_joblib = load_model('english.h5')
        # knn_from_joblib = joblib.load('cnn_hiragana.pkl')
        # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
        # load_model = pickle.load(open('model.sav', 'rb'))
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, [32, 32])

    # Scale the tensor
        img = img / 255
        img = img[:, :, :1]

        from matplotlib import pyplot as plt
        result = feader.readtext(image_path,paragraph=True)

        

        
        top_left = tuple(result[0][0][0])
        bottom_right = tuple(result[0][0][2])
        text = result[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        

        img = cv2.imread(image_path)
        spacer = 300
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection in result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            #img = cv2.putText(img,text,(20,spacer), font, 0.5,(255,255,0),2,cv2.LINE_AA)
            spacer+=15
        # plt.figure(figsize=(10,10))
        # plt.imshow(img)
        # plt.show()

        for i in result:
            text =  (i[1])
        translator = Translator()
        translation = translator.translate(text, src='ja', dest='en')
        translated_text = translation.text
        return jsonify(text, translated_text)
    

    except Exception as e:
        return jsonify({'error': str(e)})







@app.route('/api/telugu', methods=['POST'])
def telugu():

    # if session.get('authenticated'):
    #     return jsonify({'message': 'Session is active'}), 200
    
    requested_image = request.files['image']
    # model_path = os.path.join('telugu.h5')

    image_path = 'uploads/' + requested_image.filename
    requested_image.save(image_path)

    try:

        img = tf.io.read_file(image_path)

        # plt.imshow(resize.numpy().astype(int))
        # plt.show()

        # knn_from_joblib = load_model('telugu.h5')
        # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
        # load_model = pickle.load(open('model.sav', 'rb'))
        img = tf.image.decode_image(img)
        img = tf.image.resize(img, [28, 28])

    # Scale the tensor
        img = img / 255
        

        from matplotlib import pyplot as plt
        result = leader.readtext(image_path,paragraph=True)

        

        
        top_left = tuple(result[0][0][0])
        bottom_right = tuple(result[0][0][2])
        text = result[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        

        img = cv2.imread(image_path)
        spacer = 300
        font = cv2.FONT_HERSHEY_SIMPLEX
        for detection in result:
            top_left = tuple(detection[0][0])
            bottom_right = tuple(detection[0][2])
            text = detection[1]
            img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
            #img = cv2.putText(img,text,(20,spacer), font, 0.5,(255,255,0),2,cv2.LINE_AA)
            spacer+=15
        # plt.figure(figsize=(10,10))
        # plt.imshow(img)
        # plt.show()

        for i in result:
            text =  (i[1])
        translator = Translator()
        translation = translator.translate(text, src='te', dest='en')
        translated_text = translation.text
        return jsonify(text, translated_text)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/api/page', methods=['POST'])
def page():
    image = request.files['image']
    # model_path = os.path.join('telugu.h5')

    image_path = 'uploads/' + image.filename
    image.save(image_path)

    import os
    from PIL import Image

    # Open the image

    try:
        image=Image.open(image_path)

        # Get the width and height of the image
        width, height = image.size

        # Calculate the height of each part
        part_height = height // 3
        # Create a folder to save the parts
        if not os.path.exists("output_folder"):
            os.makedirs("output_folder")

        all = []
        # Loop through and crop the image into 10 parts
        for i in range(3):
            top = i * part_height
            bottom = (i + 1) * part_height
            part = image.crop((0, top, width, bottom))
            
            # Save each part in the output folder with a unique filename
            part.save(os.path.join("output_folder", f"part_{i + 1}.jpg"))
            print(os.path.join('output_folder', 'part_1.jpg'))


    # Close the original image
            
            path = os.path.join("output_folder", f"part_{i + 1}.jpg")
        

        

            img = tf.io.read_file(path)

            # plt.imshow(resize.numpy().astype(int))
            # plt.show()

            # knn_from_joblib = load_model('english.h5')
            # knn_from_joblib = joblib.load('latestdigitsnew.pkl')
            # load_model = pickle.load(open('model.sav', 'rb'))
       

        # Scale the tensor
            # img = img / 255
            # img = img[:, :, :1]
            
            from matplotlib import pyplot as plt
            result = reader.readtext(path,paragraph=True)
            for i in result:
                all.append(i[1])
            

            
       

            

       
        
            

        
        
       
        for i in all:
             
                return jsonify(all[0], all[1], all[2])
                
            # return all
    except Exception as e:
            return jsonify({'error': str(e)})



    

    








if __name__ == "__main__":
    app.run(debug=True, port=8080)


# def predict():
#     try:

#         age = 30
#         salary = 87000
#         load_model = pickle.load(open(model_file_path, 'rb'))

#         # Make a prediction using the loaded model
#         prediction = load_model.predict(([[30, 87000]]))

#         # You can convert the prediction to a human-readable label if needed
#         # For example, if your labels are 0 and 1, you can map them to 'No' and 'Yes'
#         prediction_label = 'Yes' if prediction[0] == 1 else 'No'
#         print(prediction_label)

#         result = {
#             'prediction': prediction_label
#         }
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)})

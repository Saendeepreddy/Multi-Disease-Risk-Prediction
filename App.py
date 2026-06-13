from flask import Flask, render_template,request,session
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


import matplotlib

matplotlib.use('agg')
import sqlite3

import numpy as np




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adminlogin')
def AdminLogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']

        if username=='Admin' and password=='Admin':
            return render_template("AdminApp/AdminHome.html")
        else:
            context={'msg':'Login Failed..!!'}
            return render_template("AdminApp/AdminLogin.html",**context)

@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

global dataset,filepath
@app.route('/CNN')
def CNN():
    global train_generator, validation_generator
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory("Dataset/Cardio/train",
                                                        target_size=(128, 128),
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        subset='training'
                                                        )
    validation_generator = test_datagen.flow_from_directory("Dataset/Cardio/test",
                                                            target_size=(128, 128),
                                                            batch_size=32,
                                                            class_mode='categorical',
                                                            subset='training'
                                                            )
    # Get the number of images in the training set
    num_train_samples = len(train_generator.filenames)

    # Get the number of images in the validation set
    num_validation_samples = len(validation_generator.filenames)
    total = num_train_samples
    print(num_train_samples)
    print(num_validation_samples)
    print(total)

    if os.path.exists("Model\\Cardio_model.h5"):
        base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(4, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights('Model/Cardio_model.h5')
        context = {"data": "Deep Learning Model Loaded Successfully.."}
        return render(request, 'AdminApp/AdminHome.html', context)
    else:
        base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze convolution layers

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(4, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_generator,
                            steps_per_epoch=50,
                            epochs=40,
                            validation_data=validation_generator,
                            validation_steps=50)
        model.save_weights('Model/Cardio_model.h5')
        with open('Model/model_history.pkl', 'wb') as file:
            pickle.dump(history.history, file)
        final_val_accuracy = history.history['accuracy'][-1]
        msg = f'Final Accuracy: {final_val_accuracy:.4f}'
        plt.figure(figsize=(10, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('Static/training_history.png')  # Save plot to Static folder
        plt.show()
        
        return render_template('results.html', msg="AI Model Generated Successfully..!!",acc=str(classification_accuracy))




@app.route('/KidneyModel')
def KidneyModel():
    data = pd.read_csv("Dataset/kidney_disease.csv")
    print(data.dtypes)
    data[['htn', 'dm', 'cad', 'pe', 'ane']] = data[['htn', 'dm', 'cad', 'pe', 'ane']].replace(
        to_replace={'yes': 1, 'no': 0})
    data[['rbc', 'pc']] = data[['rbc', 'pc']].replace(to_replace={'abnormal': 1, 'normal': 0})
    data[['pcc', 'ba']] = data[['pcc', 'ba']].replace(to_replace={'present': 1, 'notpresent': 0})
    data[['appet']] = data[['appet']].replace(to_replace={'good': 1, 'poor': 0, 'no': np.nan})
    data['classification'] = data['classification'].replace(
        to_replace={'ckd': 1.0, 'ckd\t': 1.0, 'notckd': 0.0, 'no': 0.0})
    data.rename(columns={'classification': 'class'}, inplace=True)
    data['pe'] = data['pe'].replace(to_replace='good', value=0)
    data['appet'] = data['appet'].replace(to_replace='no', value=0)
    data['cad'] = data['cad'].replace(to_replace='\tno', value=0)
    data['dm'] = data['dm'].replace(to_replace={'\tno': 0, '\tyes': 1, ' yes': 1, '': np.nan})
    data.drop('id', axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)
    data.apply(pd.to_numeric)
    for i in data.columns:
        data[i] = MinMaxScaler().fit_transform(data[i].astype(float).values.reshape(-1, 1))
    for i in range(0, data.shape[1]):
        if data.dtypes[i] == 'object':
            data['pcv'] = data.pcv.astype(float)
            data['wc'] = data.wc.astype(float)
            data['rc'] = data.rc.astype(float)
            data['dm'] = data.dm.astype(float)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    print("\nTrain & Test Model Generated\n\n")
    print("Total Dataset Size : " + str(len(data)) + "\n")
    print("Split Training Size : " + str(len(X_train)) + "\n")
    print("Split Test Size : " + str(len(X_test)) + "\n")

    random = RandomForestClassifier()
    random.fit(X_train, y_train)
    joblib.dump(random, 'Model/Random_Model.joblib')
    predicted = random.predict(X_test)
    rf_acc=accuracy_score(y_test, predicted)

    svm_model = svm.SVC()
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, 'Model/SVM_model.joblib')
    svpredicted = svm_model.predict(X_test)
    sv_acc=accuracy_score(y_test, svpredicted)

    return render_template('results.html', msg="Kidney Based Model Generated Successfully..!!")

@app.route('/LiverModel')
def LiverModel():
    filename = "Dataset\\Liver_Patient_Dataset.csv"
    data = pd.read_csv(filename, encoding='unicode_escape')
    data.dropna(inplace=True)
    data['Gender of the patient'] = data['Gender of the patient'].map({'Female': 0, 'Male': 1})
    X = data.iloc[:, 0:10]
    y = data.iloc[:, 10:11]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    dtmodel = DecisionTreeClassifier(random_state=0)
    dtmodel.fit(X_train, y_train)
    joblib.dump(dtmodel, "model/DecModel.joblib")
    pred = dtmodel.predict(X_test)
    acc = accuracy_score(y_test, pred)
    decacc = acc * 100

    knnmodel = KNeighborsClassifier()
    knnmodel.fit(X_train, y_train)
    joblib.dump(knnmodel, "model/KNNModel.joblib")
    pred=knnmodel.predict(X_test)
    acc=accuracy_score(y_test, pred)
    knnacc=acc*100

    # bars = ['Decision Tree', 'K-NN']
    # heights = [decacc, knnacc]
    # y_pos = np.arange(len(bars))
    # plt.bar(y_pos, heights)
    # plt.xticks(y_pos, bars)
    # plt.show()
    #
    # fig = plt.figure(figsize=(8, 5))
    # plt.pie(heights, labels=bars)
    # plt.savefig("Static/Liver_Comparision.png")
    # plt.close()
    return render_template('results.html', msg="Liver Based Model Generated Successfully..!!")

@app.route('/userlogin')
def userlogin():
    return render_template('UserApp/Login.html')

@app.route('/register')
def register():
    return render_template('UserApp/Register.html')
@app.route('/RegAction', methods=['POST'])
def RegAction():
    if request.method == 'POST':
        name=request.form['name']
        email=request.form['email']
        mobile=request.form['mobile']
        username=request.form['username']
        password=request.form['password']

        con=sqlite3.connect('database.db')
        cur=con.cursor()
        #cur.execute("create table user(name varchar(100),email varchar(200),mobile varchar(200),username varchar(100),password varchar(100))")
        cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
        data=cur.fetchone()
        if data is None:
            cur=con.cursor()
            cur.execute("insert into user values('"+name+"','"+email+"','"+mobile+"','"+username+"','"+password+"')")
            con.commit()
            return render_template('UserApp/Register.html', msg="Successfully Registered..!!")
        else:
            return render_template('UserApp/Register.html', msg="username and password is already exist..!!")

app.secret_key = '123'
@app.route('/UserAction', methods=['POST'])
def UserAction():
    username=request.form['username']
    password=request.form['password']

    con=sqlite3.connect('database.db')
    cur=con.cursor()
    cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
    data=cur.fetchone()
    if data is None:
        return render_template('UserApp/Login.html', msg="Login Failed..!!")
    else:
        session['username'] =data[3]
        return render_template('UserApp/Home.html',username=session['username'])

@app.route("/UserHome")
def UserHome():
    return render_template('UserApp/Home.html')

@app.route("/CardioRisk")
def CardioRisk():
    return render_template('UserApp/Upload.html')






@app.route('/LiverRisk')
def LiverRisk():
    return render_template('UserApp/LiverRisk.html')

@app.route('/DetectAction', methods=['POST'])
def DetectAction():
    a = request.form['a1']
    b = request.form['a2']
    c = request.form['a3']
    d = request.form['a4']
    e = request.form['a5']
    f = request.form['a6']
    g = request.form['a7']
    h = request.form['a8']
    i = request.form['a9']
    j = request.form['a10']
    ann_model = joblib.load("model/DecModel.joblib")
    pred = ann_model.predict([[a, b, c, d, e, f, g, h, i, j]])
    print("predicted value: " + str(pred))
    output=""
    if pred[0] == 1:
        output='Liver Risk Predicted'
    elif pred[0] == 2:
        output = 'No Liver Risk Predicted'
    return render_template('UserApp/Result.html', result=output)

@app.route("/KidneyRisk")
def KidneyRisk():
    # Load test data
    test_data = pd.read_csv("Test/test.csv")

    # Clean and encode categorical values
    test_data['cad'] = test_data['cad'].replace('\tno', 'no')
    test_data['dm'] = test_data['dm'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes', '': np.nan})
    test_data['pe'] = test_data['pe'].replace('good', 'no')
    test_data['appet'] = test_data['appet'].replace('no', np.nan)

    binary_map = {'yes': 1, 'no': 0}
    test_data[['htn', 'dm', 'cad', 'pe', 'ane']] = test_data[['htn', 'dm', 'cad', 'pe', 'ane']].replace(binary_map)
    test_data[['rbc', 'pc']] = test_data[['rbc', 'pc']].replace({'abnormal': 1, 'normal': 0})
    test_data[['pcc', 'ba']] = test_data[['pcc', 'ba']].replace({'present': 1, 'notpresent': 0})
    test_data['appet'] = test_data['appet'].replace({'good': 1, 'poor': 0})

    # Drop 'id' if present
    if 'id' in test_data.columns:
        test_data.drop('id', axis=1, inplace=True)

    # Convert specific columns to numeric (fix object dtype)
    for col in ['pcv', 'wc', 'rc', 'dm']:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

    # Drop rows with missing values
    test_data.dropna(inplace=True)

    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    X_test = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)

    # Load the pre-trained model
    model = joblib.load("Model/Random_Model.joblib")

    # Predict
    predictions = model.predict(X_test)

    # Add predictions back to test data
    test_data['Predicted'] = predictions
    test_data['Predicted_Label'] = test_data['Predicted'].apply(lambda x: 'kidney Risk Prediction' if x == 1 else 'No Kidney Risk Prediction')

    # Output results
    print(test_data[['Predicted', 'Predicted_Label']])

    # Save to CSV
    test_data.to_csv("Test/Predicted_Test_Results.csv", index=False)

    df = pd.read_csv("Test/Predicted_Test_Results.csv")
    columns = df.columns.tolist()
    rows = df.values.tolist()
    return render_template('UserApp/KidneyResult.html', columns= columns, rows=rows)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def build_model():
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('Model/Cardio_model.h5')

    return model  # ← VERY IMPORTANT

from PIL import Image
import base64
from flask import Flask, request, render_template_string

app.config['UPLOAD_FOLDER'] = 'uploads'  # make sure this folder exists
@app.route("/UploadAction", methods=['POST'])
def UploadAction():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    original_img = cv2.imread(filepath)
    img = cv2.resize(original_img, (128, 128))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # normalization

    # Predict
    model = build_model()
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Class names
    class_names = ['Abnormal_Heartbeat', 'History_of_Myocardial_Infarction', 'Myocardial_Infarction', 'Normal']
    result = class_names[predicted_class]

    # Draw overlay and rectangle
    display_img = original_img.copy()
    output = cv2.resize(display_img, (400, int(display_img.shape[0] * 400 / display_img.shape[1])))


    cv2.putText(output, f"{result}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    output_path = os.path.join('Static', f"output_{file.filename}")
    cv2.imwrite(output_path, output)

    return render_template('UserApp/Result.html', filename=f"output_{file.filename}")




if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)




from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from PIL import Image
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
from scikeras.wrappers import KerasClassifier
from predictor import check

K.set_image_data_format('channels_last')
from matplotlib.pyplot import imshow
import os

#######################################################################################################################
modelSavePath = 'my_model3.h5'
numOfTestPoints = 2
batchSize = 16
numOfEpoches = 10
#######################################################################################################################

classes = []


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


# Crop and rotate image, return 12 images
def getCropImgs(img, needRotations=False):
    # Resize the image to ensure it's large enough for cropping
    img = img.resize((2048, 1536))
    
    # Convert to array and ensure it has 3 color channels (RGB)
    z = np.asarray(img, dtype=np.int8)
    
    # Check if image is grayscale and convert to RGB if needed
    if len(z.shape) == 2:
        # Convert grayscale to RGB by duplicating the single channel
        z = np.stack((z,) * 3, axis=-1)
    elif z.shape[2] == 4:
        # If image has alpha channel (RGBA), remove it to get RGB
        z = z[:, :, :3]
    
    c = []
    for i in range(3):
        for j in range(4):
            try:
                crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]
                # Ensure each crop has the expected shape
                if crop.shape[0] == 512 and crop.shape[1] == 512 and crop.shape[2] == 3:
                    c.append(crop)
                    if needRotations:
                        c.append(np.rot90(np.rot90(crop)))
            except IndexError:
                # If cropping fails, create a blank image of the right size
                blank_crop = np.zeros((512, 512, 3), dtype=np.int8)
                c.append(blank_crop)
                if needRotations:
                    c.append(blank_crop)

    return c

# Get the softmax from folder name
def getAsSoftmax(fname):
    if (fname == 'b'):
        return [1, 0, 0, 0]
    elif (fname == 'is'):
        return [0, 1, 0, 0]
    elif (fname == 'iv'):
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]


# Return all images as numpy array, labels
def get_imgs_frm_folder(path):
    # x = np.empty(shape=[19200,512,512,3],dtype=np.int8)
    # y = np.empty(shape=[400],dtype=np.int8)

    x = []
    y = []

    cnt = 0
    for foldname in os.listdir(path):
        for filename in os.listdir(os.path.join(path, foldname)):
            img = Image.open(os.path.join(os.path.join(path, foldname), filename))
            # img.show()
            crpImgs = getCropImgs(img)
            cnt += 1
            if cnt % 10 == 0:
                print(str(cnt) + " Images loaded")
            for im in crpImgs:
                x.append(np.divide(np.asarray(im, np.float16), 255.))
                # Image.fromarray(np.divide(np.asarray(im, np.float16), 255.), 'RGB').show()
                y.append(getAsSoftmax(foldname))
                # print(getAsSoftmax(foldname))

    print("Images cropped")
    print("Loading as array")

    return x, y, cnt

# Load the dataset
def load_dataset(testNum=numOfTestPoints):
    print("Loading images..")

    train_set_x_orig, train_set_y_orig, cnt = get_imgs_frm_folder(dataTrainPath)

    testNum = numOfTestPoints * 12
    trainNum = (cnt * 12) - testNum

    print(testNum, trainNum)

    train_set_x_orig = np.array(train_set_x_orig, np.float16)
    train_set_y_orig = np.array(train_set_y_orig, np.int8)

    nshapeX = train_set_x_orig.shape
    nshapeY = train_set_y_orig.shape

    # train_set_y_orig = oh

    print("folder trainX" + str(nshapeX))
    print("folder trainY" + str(nshapeY))

    print("Images loaded")

    print("Loading all data")

    test_set_x_orig = train_set_x_orig[trainNum:, :, :, :]
    train_set_x_orig = train_set_x_orig[0:trainNum, :, :, :]

    test_set_y_orig = train_set_y_orig[trainNum:]
    train_set_y_orig = train_set_y_orig[0:trainNum]

    classes = np.array(os.listdir(dataTrainPath))  # the list of classes

    # train_set_y_orig = np.array(train_set_y_orig).reshape((np.array(train_set_y_orig, np.float16).shape[1],
    #                                                       np.array(train_set_y_orig, np.float16).shape[0]))
    # test_set_y_orig = np.array(test_set_y_orig).reshape((np.array(test_set_y_orig, np.float16).shape[1],
    #                                                     np.array(test_set_y_orig, np.float16).shape[0]))
    print(train_set_y_orig[0:50, :])
    print(train_set_x_orig[1])
    print("Data load complete")

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def defModel(input_shape):
    X_input = Input(input_shape)

    # The max pooling layers use a stride equal to the pooling size

    X = Conv2D(16, (3, 3), strides=(1, 1))(X_input)  # 'Conv.Layer(1)'

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(2)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Conv.Layer(3)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(4)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(5)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(5) will be 82x82, we want 84x84

    X = MaxPooling2D((2, 2), strides=2)(X)  # MP Layer(6)

    X = Conv2D(64, (2, 2), strides=(1, 1))(X)  # Conv.Layer(7)

    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=(2, 2))(X)  # Output of convlayer(7) will be 40x40, we want 42x42

    X = MaxPooling2D((3, 3), strides=3)(X)  # MP Layer(8)

    X = Conv2D(32, (3, 3), strides=(1, 1))(X)  # Con.Layer(9)

    X = Activation('relu')(X)

    X = Flatten()(X)  # Convert it to FC

    X = Dense(256, activation='relu')(X)  # F.C. layer(10)

    X = Dense(128, activation='relu')(X)  # F.C. layer(11)

    X = Dense(4, activation='softmax')(X)

    # ------------------------------------------------------------------------------

    model = Model(inputs=X_input, outputs=X, name='Model')

    return model


def train(batch_size, epochs):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    model = defModel(X_train.shape[1:])

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    # Uncomment the below code and comment the lines with(<>), to implement the image augmentations.

    # datagen = keras.preprocessing.image.ImageDataGenerator(
    # zoom_range=0.2, # randomly zoom into images
    # rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    # horizontal_flip=False,  # randomly flip images
    # vertical_flip=False  # randomly flip images
    # )
    while True:
        try:
            model = load_model(modelSavePath)
        except:
            print("Training a new model")

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size) # <>

        # history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
        #                              epochs=epochs
        #                              # validation_data=(X_test, Y_test))
        #                              )
        # history.model.save('my_model3.h5')

        model.save(modelSavePath)

        preds = model.evaluate(X_test, Y_test_orig, batch_size=1, verbose=1, sample_weight=None)
        print(preds)

        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]) + "\n\n\n\n\n")
        ch = input("Do you wish to continue training? (y/n) ")
        if ch == 'y':
            epochs = int(input("How many epochs this time? : "))
            continue
        else:
            break

    return model


@tf.function(reduce_retracing=True)
def predict_once(model, x):
    return model(x, training=False)

def predict(img, savedModelPath, showImg=True):
    # Load model without optimizer configuration
    model = load_model(savedModelPath, compile=False)
    # if showImg:
    # Image.fromarray(np.array(img, np.float16), 'RGB').show()

    x = img
    if showImg:
        Image.fromarray(np.array(img, np.float16), 'RGB').show()
    x = np.expand_dims(x, axis=0)

    # Use the decorated function instead of direct predict call
    softMaxPred = predict_once(model, x).numpy()
    print("prediction from CNN: " + str(softMaxPred) + "\n")
    probs = softmaxToProbs(softMaxPred)

    # plot_model(model, to_file='Model.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    maxprob = 0
    maxI = 0
    for j in range(len(probs)):
        # print(str(j) + " : " + str(round(probs[j], 4)))
        if probs[j] > maxprob:
            maxprob = probs[j]
            maxI = j
    # print(softMaxPred)
    print("prediction index: " + str(maxI))
    return maxI, probs


def softmaxToProbs(soft):
    import math
    z_exp = [math.exp(i) for i in soft[0]]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]

# Creating a Flask Instance
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading Pre-trained Model ...")


def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to model.
    '''
    print('Processing Image ...')
    currImg_BGR = cv2.imread(path)
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
    currImg = currImg/255.0
    currImg = np.reshape(currImg, (1, 150, 150, 3))
    return currImg


def model_pred(image):

    print("Image_shape", image.shape)
    print("Image_dimension", image.ndim)
    # Returns Probability:
    # prediction = model.predict(image)[0]
    # Returns class:
    prediction = (model.predict(image) > 0.5).astype("int32")#model.predict_classes(image)[0]

    return (prediction)

    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('breastindex.html')

@app.route('/upload1', methods=['GET', 'POST'])
def upload_file1():
    # Checks if post request was submitted
    if request.method == 'POST':
        
        # check if the post request has the file part
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # check if filename is an empty string
        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # if file is uploaded
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")
            img_path=imgPath
            arrayImg=None
            printData=True
            crops = []
            if arrayImg == None:
                img = image.load_img(img_path)
                crops = np.array(getCropImgs(img, needRotations=False), np.float16)
                crops = np.divide(crops, 255.)
            Image.fromarray(np.array(crops[0]), "RGB").show()

            classes = []
            classes.append("Benign")
            classes.append("InSitu")
            classes.append("Invasive")
            classes.append("Lymphocytes")

            compProbs = []
            compProbs.append(0)
            compProbs.append(0)
            compProbs.append(0)
            compProbs.append(0)

            for i in range(len(crops)):
                if printData:
                    print("\n\nCrop " + str(i + 1) + " prediction:\n")

                ___, probs = predict(crops[i], modelSavePath, showImg=False)

                for j in range(len(classes)):
                    if printData:
                        print(str(classes[j]) + " : " + str(round(probs[j], 4)) + "%")
                    compProbs[j] += probs[j]

            if printData:
                print("\n\nAverage from all crops\n")

            for j in range(len(classes)):
                if printData:
                    print(str(classes[j]) + " : " + str(round(compProbs[j] / 12, 4)) + "%")
            oimg=cv2.imread(img_path)
            rimg=cv2.resize(oimg,[300, 300])
            cv2.imwrite('./static/rimg.png',rimg)
            rgbimg=cv2.cvtColor(rimg,cv2.COLOR_BGR2RGB)
            gbimg=cv2.GaussianBlur(rgbimg, (7,7), 0)
            mimg=cv2.medianBlur(rgbimg,5)
            cv2.imwrite('./static/mimg.png',mimg)

            dimg=cv2.fastNlMeansDenoisingColored(mimg,None,20,10,7,21)
            cv2.imwrite('./static/dimg.png',dimg)
            eimg=cv2.Canny(mimg,100,200)
            cv2.imwrite('./static/eimg.png',eimg)
            print(img_path)

            # Calculate percentage values for the progress bars
            benign_percent = round(compProbs[0] / 12, 4)
            insitu_percent = round(compProbs[1] / 12, 4)
            invasive_percent = round(compProbs[2] / 12, 4)
            lymphocytes_percent = round(compProbs[3] / 12, 4)
            
            return render_template('upload1.html', 
                name=filename,
                upimage='/static/uploads/'+filename, 
                bencla=str(classes[0])+" - "+str(benign_percent) + "%",
                InSitu=str(classes[1])+" - "+str(insitu_percent) + "%",
                inva=str(classes[2])+" - "+str(invasive_percent) + "%",
                norm=str(classes[3])+" - "+str(lymphocytes_percent) + "%",
                # Add these new variables for the progress bars - multiply by 100 for proper CSS percentages
                benign_value=benign_percent * 100,
                insitu_value=insitu_percent * 100,
                invasive_value=invasive_percent * 100,
                lymphocytes_value=lymphocytes_percent * 100
            )
    return redirect(url_for('home'))


app.config['UPLOAD_FOLDER'] = 'uploads'



def model_predict(img_path, model):
    
    img = Image.open(img_path).resize((224,224)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
   
    preds = model.predict(img)[0]
    prediction = sorted(
      [(class_dict[i], round(j*100, 2)) for i, j in enumerate(preds)],
      reverse=True,
      key=lambda x: x[1]
  )
    return prediction,img


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
from tensorflow.keras.utils import plot_model
# If you need model_to_dot specifically, use:
# from tensorflow.keras.utils import model_to_dot


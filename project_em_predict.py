import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import tkinter as tk
import tkinter.font as tkFont
from PIL import Image


# Loading in the model
# The model is trained on 19137 images from 3 different classes
# Refer to the ipynb file for more info
model = load_model("final_trained_model.h5")

emoji_dictionary = {"happy": ["\U0001f600",
                              "\U0001f603",
                              "\U0001f604",
                              "\U0001f601",
                              "\U0001f606",
                              "\U0001f923",
                              "\U0001f602",
                              "\U0001f642",
                              "\U0001f60A",
                              "\U0001f92A",
                              "\U0001f917",
                              "\U0001f60F",
                              "\U0001f600",
                              "\U0001f608",
                              "\U0001f921",
                              "\U0001f638",
                              "\U0001f9D1",
                              "\U0001f31E", ],
                    "sad":   ["\U0001f614",
                              "\U0001f61F",
                              "\U0001f641",
                              "\U00002639",
                              "\U0001f601",
                              "\U0001f97A",
                              "\U0001f627",
                              "\U0001f62D",
                              "\U0001f629",
                              "\U0001f62D"],
                    "neutral": ["\U0001f910",
                                "\U0001f610",
                                "\U0001f611",
                                "\U0001f636",
                                "\U0001f615",
                                "\U0001f971",
                                "\U0001f480",
                                "\U00002620",
                                "\U0001f482",
                                "\U0001f464",
                                "\U0001f439",
                                "\U0001f428"]}


def load_and_prepare_image(filename, image_shape=48):
    """
        Args:
            filename (str) : image path for reading and processing the image
            image_shape (str) : image shape to resize the image accordingly
        Returns:
            An image tensor of size (image_shape, image_shape)
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, 1)
    img = tf.image.resize(img, size=[image_shape, image_shape])
    img = img/255.
    return img


# The classes list, these are the available classes that our model will try to predict
classes = ["happy", "sad", "neutral"]


def predict_class(model, filename):
    """
        Args:
            model (tensorflow model object) : a model which will predict the class of the image
            filename (str) : image path for predicting the class
    """
    img = load_and_prepare_image(filename)

    predict_x = model.predict(tf.expand_dims(img, axis=0))
    class_name = classes[tf.argmax(predict_x[0])]
    emoji_class = random.choice(emoji_dictionary[class_name])

    # Opening and resizing the image to 250x250
    print(f"filename = {filename}")
    im1 = Image.open(filename)
    im1 = im1.resize((250, 250))
    im1.save(filename)  # Saving the image by same name

    # Creating a tkinter window
    root = tk.Tk()

    # Setting the font size and font style
    fontStyle = tkFont.Font(family="Lucida Grande", size=40)

    # Reading in the resized image
    logo = tk.PhotoImage(file=filename)

    # Displaying the image
    tk.Label(root, image=logo).pack(side="top")

    # Appending the predicted class and emoji into one
    pred_ = class_name + " " + \
        emoji_class

    # Displaying the pred_
    tk.Label(root,
             justify=tk.LEFT,
             padx=10,
             text=pred_,
             font=fontStyle).pack(side="bottom")
    root.mainloop()


cam = cv2.VideoCapture(0)
# cv2.resizeWindow(cam, 500, 500)

cv2.namedWindow("Image_capture")

img_counter = 0

while True:
    # Runs until the ESC is pressed
    # Fails to run if any error with camera
    ret, frame = cam.read()
    if not ret:
        print("Camera failed to start, please try again")
        break

    cv2.imshow("Image_capture", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:

        # ESC pressed
        print("ESC pressed, closing window")
        break

    elif k % 256 == 32:

        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)

        # Calling the function to predict the class
        predict_class(model=model, filename=img_name)

        # Increasing the counter for generating unique image name every time user clicks the space
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

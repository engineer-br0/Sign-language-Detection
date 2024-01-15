import os
print("Checkpoint 1")
import cv2
print("Checkpoint 2")
import numpy as np
print("Checkpoint 3")
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
from tensorflow.keras.utils import to_categorical  # Import to_categorical from tensorflow.keras.utils
from sklearn.model_selection import train_test_split
print("Checkpoint 4")

data_path_test = "./asl_alphabet_test/asl_alphabet_test"
data_path_train = "./asl_alphabet_train"

# Get the classes (letters) from the folder names
classes_train = sorted(os.listdir(data_path_train))
classes_test = sorted(os.listdir(data_path_test))
print("classes_train", classes_train)
print("classes_test",classes_test)

#print(os.listdir(data_path_train))

# Combine train and test classes (assuming they are the same)
classes = classes_train
#print("classes", enumerate(classes))

# Initialize lists to store images and labels
images = []
labels = []

# Load training data
#print(os.path.join(data_path_train, class_name))
for label, class_name in enumerate(classes):
    #print("label, class_name", label, class_name)
    class_path = os.path.join(data_path_train, class_name)
    #print(class_path)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
        # Resize the image
          image = cv2.resize(image, (64, 64))
          images.append(image)
          labels.append(label)
        else:
          print("Error: Empty or invalid image.")

# Convert lists to NumPy arrays
# print(images[0])
images_train = np.array(images) / 255.0
labels_train = to_categorical(labels, num_classes=len(classes))
print("labels_train", labels_train)
# Load test data
images = []  # Clear the list for test data
labels = []  # Clear the list for test data

for label, class_name in enumerate(classes):
    print(label, class_name)
    #class_path = os.path.join(data_path_test, class_name)
    class_path = os.path.join(data_path_test)
    print("class_path",class_path)
    #####
    image_name = classes_test[label]
    print("image_name", image_name)
    image_path = os.path.join(class_path, image_name)
    print("image_path", image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
    # Resize the image
     image = cv2.resize(image, (64, 64))
     images.append(image)
     labels.append(label)
     print("labelis", label)
    else:
      print("Error: Empty or invalid image.")
        #image = cv2.resize(image, (64, 64))
    # for image_name in os.listdir(class_path):
    #     image_path = os.path.join(class_path, image_name)
    #     print("image_path", image_path)
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     if image is not None:
    #     # Resize the image
    #       image = cv2.resize(image, (64, 64))
    #       images.append(image)
    #       labels.append(label)
    #     else:
    #       print("Error: Empty or invalid image.")
    #     #image = cv2.resize(image, (64, 64))


# Convert lists to NumPy arrays
images_test = np.array(images) / 255.0
labels_test = to_categorical(labels, num_classes=len(classes))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_train, labels_train, test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("model", model)
    return model

import numpy as np
from sklearn.model_selection import train_test_split
# from build_model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed data
#X_train, X_test, y_train, y_test = np.load("preprocessed_data.npy")

# Create and compile the model
num_classes = len(np.unique(y_train))
model = build_model(num_classes=len(np.unique(np.argmax(y_train, axis=1))))

# Train the model
model.fit(X_train.reshape(-1, 64, 64, 1), y_train, validation_data=(X_test.reshape(-1, 64, 64, 1), y_test), epochs=10, batch_size=100)
# Example output layer

# Save the model
model.save("sign_language_detector_model.h5")



#from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
from flask import Flask, request, jsonify
from flask_cors import CORS

# Function to capture a photo using Colab's webcam snippet
def take_photo(quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
          const div = document.createElement('div');
          const capture = document.createElement('button');
          capture.textContent = 'Capture';
          div.appendChild(capture);

          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });

          document.body.appendChild(div);
          div.appendChild(video);
          video.srcObject = stream;
          await video.play();

          // Resize the output to fit the video element.
          google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

          // Wait for Capture to be clicked.
          await new Promise((resolve) => capture.onclick = resolve);

          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getVideoTracks()[0].stop();
          div.remove();
          return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    image = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), -1)
    return image

# Load the trained model
model = load_model("sign_language_detector_model.h5")

# Create a list of class labels
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Display instructions
instructions = [
    "Welcome to Sign Language Detector!",
    "Instructions:",
    "1. Place your hand in front of the camera.",
    "2. Try different sign language gestures.",
    "3. Press 'q' to exit the application."
]

def predict2():
  for instruction in instructions:
    print(instruction)

#while True:
    # Capture a frame from the camera or a photo
# try:
#     frame = take_photo()
# except Exception as e:
#     print(f"Error capturing photo: {e}")

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["POST"])
def check():
   print("hello bhai working fine...")
   return jsonify("hello bhai working fine...")

@app.route("/predict", methods=["POST"])
def predict():
  print("hi")
  # Retrieve image from binary data
  image_data = request.data

  # Decode the image data
  frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), -1)
  #break

    # Preprocess the frame (resize, convert to grayscale, normalize)
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     if image is not None:
    #     # Resize the image
    #       image = cv2.resize(image, (64, 64))
    #       images.append(image)
    #       labels.append(label)
  frame = cv2.resize(frame, (64, 64))
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  normalized_frame = gray_frame / 255.0
  input_data = normalized_frame.reshape((-1, 64, 64, 1))

    # Make predictions using the model
  predictions = model.predict(input_data)
  predicted_class = classes[np.argmax(predictions)]

    # Print raw predictions
  print("Raw Predictions:", predictions)
  

    # Display the predicted class on the frame
  cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  print(f"Predicted: {predicted_class}")
    # Display instructions on the frame
  y = 80
  for instruction in instructions:
      cv2.putText(frame, instruction, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
      y += 20

    # Display the frame
  cv2_imshow(frame)

  result = {"predictions": predicted_class.tolist()}

  return jsonify(result)

    # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the camera and close all OpenCV windows
#cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run()
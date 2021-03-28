# Import os module in order to list directories and OpenCV-Python
# Make sure opencv-contrib-python is installed
# The command is pip install opencv-contrib-python
import os
import cv2
import numpy as np  # Numpy is an OpenCV dependecy, it is pre-installed
from PIL import Image  # Install pillow package: pip install pillow


def train_model() -> None:
    """Function that generates a trained model using all datasets from '/datasets/' directory."""

    # Initialize names and path to empty list
    names = []
    paths = []

    # Get the names
    for users in os.listdir("..\\03-datasets\\"):
        names.append(users)

    # Get the path to all the images
    for name in names:
        for image in os.listdir(f"..\\03-datasets\\{name}\\"):
            path_string = f"..\\03-datasets\\{name}\\{image}"
            # ../03-datasets/chris/1_1.jpg
            paths.append(path_string)

    faces = []
    ids = []

    # For each image create a numpy array and add it to faces list
    for img_path in paths:
        # Get image and convert it into B&W
        # Didn't use opencv to convert it to Gray this time
        image = Image.open(img_path).convert("L")

        # Convert the image to numpy array
        imgNp = np.array(image, "uint8")
        faces.append(imgNp)

        # Get the id of the person
        id = int(img_path.split("\\")[3].split("_")[0])
        ids.append(id)

    # Convert the ids list to numpy array
    ids = np.array(ids)

    # At this point, we have all the saved faces in faces with the corresponding persons' id in ids 

    # Call the recognizer (make sure contrib is installed)
    trainer = cv2.face.LBPHFaceRecognizer_create()

    # train function accespts two numpy arrays
    # Pass to it all the data (faces and ids numpy arrays)
    trainer.train(faces, ids)

    # Write the generated model to a .yml file
    trainer.write("trained-model.yml")
    print("Model trained and saved.")


if __name__ == "__main__":
    try:
        train_model()
    except:
        print("Error occured !")

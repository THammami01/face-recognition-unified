# # Import os module in order to list directories; OpenCV-Python; Path from module pathlib to use it to make directories
import os
import cv2
from pathlib import Path


def saveImage(image, userName: str, userId: str, imgId: str) -> None:
    """Function to save an image."""
    # Create a folder with the name as userName
    # Save the images inside the previously created folder
    Path(f"..\\03-datasets\\{userName}\\").mkdir(parents=True, exist_ok=True)
    # If parents is true, a missing parent won't raise FileNotFoundError.
    # If exist_ok is true, FileExistsError exceptions will be ignored.
    cv2.imwrite(f"..\\03-datasets\\{userName}\\{userId}_{imgId}.jpg", image)
    print(f"Image {imgId} has been saved in folder: {userName}")


def generate_dataset_from_imgsdir() -> None:
    """Function to generate a dataset from a directory that containes images of a person."""

    # Initialize the classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print("Enter the ID and name of the person..")
    userId: str = input("ID: ")
    userName: str = input("Name: ")

    # Initialize the number of saved images at 0
    count: int = 0

    # For each photo in the matching dir
    for imgName in os.listdir(f"..\\01-images\\{userName}\\"):
        # Get the image
        img = cv2.imread(f"..\\01-images\\{userName}\\{imgName}")

        # Get the gray version of img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_img)

        # Save all detected faces in each image
        for (x, y, w, h) in faces:
            count += 1
            coords = [x, y, w, h]
            roi_img = img[
                coords[1]: coords[1] + coords[3],
                coords[0]: coords[0] + coords[2]
            ]
            saveImage(roi_img, userName, userId, count)

    if count > 0:
        print(f"Dataset of {count} image(s) has been created for {userName}.")
    else:
        print("Did not create a dataset because no face is detected.")


if __name__ == "__main__":
    try:
        generate_dataset_from_imgsdir()
    except:
        print("Error occured !")

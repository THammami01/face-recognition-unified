# import OpenCV-Python and Path from module pathlib to use it in order to make directories
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


def generate_dataset_from_cam() -> None:
    """Function to generate a dataset from camera."""

    # Initialize the classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Start the video camera
    vc = cv2.VideoCapture(0)

    # Get the userId and userName
    print("Enter the ID and name of the person..")
    userId: str = input("ID: ")
    userName: str = input("Name: ")

    # Initialize the number of saved images at 0
    count: int = 0

    # For each frame
    while True:
        # Get the frame and save it in img
        _, img = vc.read()

        # Keep originalImg to crop for dataset, add rectangles to and show only img
        originalImg = img.copy()

        # Get the gray version of img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the coordinates of the location of the face in the gray image
        faces = faceCascade.detectMultiScale(
            gray_img,
            # scaleFactor=1.2,
            # minNeighbors=5,
            # minSize=(50, 50)
        )

        # Draw a rectangle around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coords = [x, y, w, h]

        # Show the image
        cv2.imshow("Live video camera with identified faces", img)

        # Wait for user to press a key
        key = cv2.waitKey(1) & 0xFF

        # If s is pressed  and count is less than 5, save the image
        if key == ord('s') and count < 5:
            count += 1
            # Crop the image before saving it (only last detected is saved)
            roi_img = originalImg[
                coords[1]: coords[1] + coords[3],
                coords[0]: coords[0] + coords[2]
            ]
            # Save the image after cropping it
            saveImage(roi_img, userName, userId, count)
        # If q is pressed or count equals 5, break out of the loop
        elif key == ord('q') or count == 5:
            break

    if count > 0:
        print(f"Dataset of {count} image(s) has been created for {userName}.")
    else:
        print("Did not create a dataset because no image is saved.")

    # Stop the video camera and close the window
    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        generate_dataset_from_cam()
    except:
        print("Error occured !")

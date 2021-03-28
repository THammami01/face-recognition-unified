# Import os module in order to list directories and OpenCV-Python
import os
import cv2


def recognize_faces_in_img(imageName: str) -> None:
    """Function used to recognize faces in an image using a trained model.
    @param imageName: a string representing the name of the image to recognize faces from.
    Image with the same name must be placed first in '/01-images/' directory.
    """

    # Initialize the classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Initialize the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Tell the recognizer to use the .yml file that contains the trained model to recognize faces
    recognizer.read("trained-model.yml")

    # Get the names corresponding to each id
    names = []
    for users in os.listdir("..\\03-datasets\\"):
        names.append(users)

    # Read the image to recognize faces from
    img = cv2.imread(f"..\\01-images\\{imageName}")

    # Convert it to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray image
    faces = faceCascade.detectMultiScale(
        gray_img,
        # scaleFactor=1.2,
        # minNeighbors=5,
        # minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        # Draw a green rectangle around each face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Tell the trained model to predict whose face's it is
        id, _ = recognizer.predict(gray_img[y: y+h, x: x+w])

        if id:  # If id is different than 0, then the model recognized the face
            cv2.putText(  # see putText function documentation
                img,
                names[id-1],  # if exists, show the according name
                (x-3, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
        else:  # Otherwise, it didn't recognize the face
            cv2.putText(
                img,
                "Unknown",  # otherwise, show Unknown
                (x-3, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    # Show the image after drawing rectangles around and putting text on each detected face
    if(len(faces)):
        print("Showing original image after drawing rectangles..")
        # Save the image after drawing
        imgLoc: str = f"..\\02-edited-images\\{imageName}"
        print(f"Image saved in {imgLoc}")
        cv2.imwrite(imgLoc, img)
    else:
        print("No face is detected.")

    # Show the original image after processing
    cv2.imshow(f"Final Image (Number of detected faces: {len(faces)})", img)
    cv2.waitKey()


if __name__ == "__main__":
    imageName: str = __import__("sys").argv[1]  # From Args
    try:
        recognize_faces_in_img(imageName)
    except:
        print("Error occured ! Image is most likely unavailable.")

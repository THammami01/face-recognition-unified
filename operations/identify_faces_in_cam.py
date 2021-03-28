# import OpenCV-Python package
import cv2


def identify_faces_in_cam() -> None:
    """Function that identifies faces from video camera."""

    # Initialize the classifier
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Start camera video
    vc = cv2.VideoCapture(0)

    # For each frame/image captured
    while True:
        # Get the frame and save it in img
        _, img = vc.read()

        # Convert img to Grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the gray image
        faces = faceCascade.detectMultiScale(
            gray_img,
            # scaleFactor=1.2,
            # minNeighbors=5,
            # minSize=(50, 50)
        )

        # Draw a red rectangle around each face
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Show the frame after drawing rectangles
        cv2.imshow("Identified Faces", img)

        keyPressed: int = cv2.waitKey(1) & 0xFF
        # Quit if user presses q
        if(keyPressed == ord("q")):
            break

    # Stop video camera and close the window
    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        identify_faces_in_cam()
    except:
        print("Error occured !")

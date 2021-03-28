# OpenCV-Python is a Python library designed to solve computer vision problems.
# pip install opencv-python and import the library
import cv2


def identify_faces_in_img(imageName: str) -> None:
    """Function that identifies faces in an image.
    @param imageName: a string representing the name of the image to process.
    Image with the same name must be placed first in '/01-images/' directory.
    """

    # Cascading classifiers are trained with several hundred "positive" sample views of a particular object and arbitrary "negative" images of the same size. After the classifier is trained it can be applied to an image and detect the object in question.
    # Haar Cascade is a machine learning object detection algorithm proposed by Paul Viola and Michael Jones
    # Create the haar cascade that detects frontal faces
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Read the image
    # imageName = "img.jpg" # Hardcoded
    # imageName = __import__("sys").argv[1] # From SysArgs
    image = cv2.imread(f"..\\01-images\\{imageName}\\")

    # Show the original image
    print("Showing original image..")
    cv2.imshow("Original Image", image)
    cv2.waitKey()

    # Convert the image to Grayscale (8-bit, shades of gray image)
    # The reason for this is that gray channel is easy to process and is computationally less intensive as it contains only 1-channel of black-white.
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Show the gray image
    print("Showing gray image..")
    cv2.imshow("Gray Image", grayImage)
    cv2.waitKey()

    # Detect faces in the gray image
    faces = faceCascade.detectMultiScale(grayImage)

    # Print the total number of detected faces and the coordinates of each
    print(f"Number of detected faces: {len(faces)}")
    print("Faces coordinates:")
    index: int = 0
    for face in faces:
        index += 1
        print(f"- Face {index}: {face}")

    # Draw a rectangle around each face (on the original image, not the gray one)
    for (x, y, w, h) in faces:
        # Function accepts img, pt1, pt2, color (in BGR not RGB), thickness
        # pt1 and pt2 are the corners of the rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Red color is (255, 0, 0) in RGB

    if(len(faces)):
        print("Showing original image after drawing rectangles..")
        # Save the image after drawing
        imgLoc: str = f"..\\02-edited-images\\{imageName}"
        print(f"Image saved in {imgLoc}")
        cv2.imwrite(imgLoc, image)
    else:
        print("No face is detected.")

    # Show the original image after processing
    cv2.imshow(f"Final Image (Number of detected faces: {len(faces)})", image)
    cv2.waitKey()


if __name__ == "__main__":
    imageName: str = __import__("sys").argv[1]  # From Args
    # try:
    identify_faces_in_img(imageName)
    # except:
        # print("Error occured ! Image is most likely unavailable.")

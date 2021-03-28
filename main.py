from .operations.identify_faces_in_img import identify_faces_in_img
from .operations.identify_faces_in_cam import identify_faces_in_cam
from .operations.generate_dataset_from_imgsdir import generate_dataset_from_imgsdir
from .operations.generate_dataset_from_cam import generate_dataset_from_cam
from .operations.recognize_faces_in_img import recognize_faces_in_img
from .operations.train_model import train_model


while True:
    try:
        option: int = int(input(
            "Choose option:\n" +
            "1. Identify faces in an image\n" +
            "2. Identify faces from camera\n" +
            "3. Generate dataset from a directory that containes images\n" +
            "4. Generate dataset from camera\n" +
            "5. Train a model\n" + 
            "6. Recognize faces in an image\n" +
            "0. Quit\n" +
            "> "
        ))

        if(option <= 0):
            print("Exiting.")
            break
        elif(option == 1):
            imageName: str = input("Enter image name: ")
            identify_faces_in_img(imageName)
        elif(option == 2):
            identify_faces_in_cam()
        elif(option == 3):
            generate_dataset_from_imgsdir()
        elif(option == 4):
            generate_dataset_from_cam()
        elif(option == 5):
            train_model()
        elif(option == 6):
            imageName: str = input("Enter image name: ")
            recognize_faces_in_img(imageName)

        print("=" * 40)

    except:
        print("Error occured.\nExiting.")
        break

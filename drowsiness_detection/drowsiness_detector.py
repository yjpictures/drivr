import cv2
import dlib
from scipy.spatial import distance

EAR_CUTOFF: float = 0.2
CAMERA_PORT: int = 0
DATABASE: str = "shape_predictor_68_face_landmarks.dat"


def calculate_EAR(eye):
    p2_p6 = distance.euclidean(eye[1], eye[5])                  # calculating distance between two points using scipy
    p3_p5 = distance.euclidean(eye[2], eye[4])
    p1_p4 = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (p2_p6 + p3_p5) / (2.0 * p1_p4)          # formula for EAR
    return eye_aspect_ratio


camera_capture = cv2.VideoCapture(CAMERA_PORT)                  # selecting the camera port
default_face_detector = dlib.get_frontal_face_detector()        # default face detector using dlib
dlib_68_point_detection = dlib.shape_predictor(DATABASE)        # maps out points on the face

while True:
    _, frame = camera_capture.read()
    grayscale_feed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                    # converts the feed to greyscale
    faces = default_face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))      # detecting face live on the feed

    for face in faces:

        face_landmarks = dlib_68_point_detection(grayscale_feed, face)          # getting the data for 68 points on face
        leftEye: list = []
        rightEye: list = []

        # function for highlighting the border around eyes
        def border_around_eye(x_coordinate, y_coordinate, instance):
            present_points = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
            next_points = [37, 38, 39, 40, 41, 36, 43, 44, 45, 46, 47, 42]
            next_point = next_points[present_points.index(instance)]
            # this draws a border around the eye
            cv2.line(frame, (x_coordinate, y_coordinate), (face_landmarks.part(next_point).x,
                                                           face_landmarks.part(next_point).y), (0, 255, 0), 1)

        for each_eye_point in range(36, 48):
            x = face_landmarks.part(each_eye_point).x
            y = face_landmarks.part(each_eye_point).y
            # points 36-41 are for the left eye and 42-47 are for the right eye
            rightEye.append((x, y)) if each_eye_point > 41 else leftEye.append((x, y))
            # border_around_eye(x, y, each_eye_point)
        EAR = round((calculate_EAR(leftEye) + calculate_EAR(rightEye)) / 2, 2)

        if EAR < EAR_CUTOFF:
            # interrupt detected
            # the line below writes text on the photo
            # cv2.putText(frame, "WAKE UP!!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
            print("driver is drowsy")
        # continuous EAR print
        # print(EAR)

    # the line below creates a window with camera feed
    # cv2.imshow("Drowsiness Detector", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
camera_capture.release()
cv2.destroyAllWindows()

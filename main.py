import cv2 as cv
import face_recognition

# Load the known image
known_image = face_recognition.load_image_file("pavan.jpg")
# print(known_image)
known_faces = face_recognition.face_encodings(known_image)[0]
# print(known_faces)

# Launch the live camera
cam = cv.VideoCapture(0)

# Check if the camera is opened
if not cam.isOpened():
    print("Camera not working")
    exit()

# Start reading frames from the camera
while True:
    # Capture the image frame-by-frame
    ret, frame = cam.read()

    # Check if the frame is successfully read
    if not ret:
        print("Can't receive the frame")
        break

    # Face detection in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = face_location
        # Draw a rectangle with blue line borders of thickness 2 px
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Compare with the known faces
        results = face_recognition.compare_faces([known_faces], face_encoding)

        if results[0]:
            cv.putText(frame, 'Pavan', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv.putText(frame, 'Unknown', (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)

    # End the streaming
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
cam.release()

# Close all OpenCV windows
cv.destroyAllWindows()

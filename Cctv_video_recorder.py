import cv2
import datetime

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

recording = False
video_writer = None
frames_left = 0

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1200:
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Start recording when motion is detected
    if motion_detected and not recording:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}.avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(
            filename, fourcc, 20.0,
            (frame1.shape[1], frame1.shape[0])
        )

        recording = True
        frames_left = 100  # record ~5 seconds
        print("Recording started:", filename)

    # Write frames if recording
    if recording:
        video_writer.write(frame1)
        frames_left -= 1

        if frames_left <= 0:
            recording = False
            video_writer.release()
            print("Recording saved")

    cv2.imshow("CCTV Surveillance", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

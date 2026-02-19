import cv2
import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

log = open("final_log.txt", "a")

recording = False
video_writer = None
frames_left = 0

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)

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
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Face detection
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame1, "Face", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame1, timestamp, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Start recording on motion
    if motion_detected and not recording:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_writer = cv2.VideoWriter(
            f"event_{ts}.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            20.0,
            (frame1.shape[1], frame1.shape[0])
        )
        cv2.imwrite(f"snapshot_{ts}.jpg", frame1)
        log.write(f"Motion detected at {timestamp}\n")
        log.flush()

        recording = True
        frames_left = 100  # ~5 seconds

    # Write video frames
    if recording:
        video_writer.write(frame1)
        frames_left -= 1
        if frames_left <= 0:
            recording = False
            video_writer.release()

    cv2.imshow("Integrated CCTV Surveillance System", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log.close()
cap.release()
cv2.destroyAllWindows()

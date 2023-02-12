import argparse
import cv2 as cv
import requests
import uuid


def main(path):
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        print("Cannot open capture")
        exit()

    frame_count = 0
    detection_rate = 5
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Cannot receive frame")
            break
        frame_count += 1

        if frame_count % detection_rate == 0:
            encoded = cv.imencode(".jpg", frame)[1]
            frame_id = str(uuid.uuid4())
            file = {'file': (f'{frame_id}.jpg', encoded.tobytes(), 'image/jpeg')}
            data = {"id": f"{frame_id}"}
            response = requests.post("http://127.0.0.1:8000/detection", files=file, data=data, timeout=5)
            print(response.json())
            bounding_box = 0

        # Operations on the frame here
        cv.rectangle(frame, (20, 20), (500, 200), (255, 0, 0), 3)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=0, help="path to video, defaults to using the camera")
    args = parser.parse_args()

    main(args.path)

import cv2 as cv


def main():
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open capture")
        exit()
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Cannot receive frame")
            break

        # Operations on the frame here
        cv.rectangle(frame, (20, 20), (500, 200), (255, 0, 0), 3)

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

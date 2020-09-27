import datetime
import cv2
import numpy as np
import time
import os
from utils.mtcnn import TrtMtcnn

BBOX_COLORS = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]

class Recorder():
    def __init__(self, cap, key, mtcnn):
        self.cap = cap
        self.mtcnn = mtcnn
        self.key = key
        self.filename_stack = []
        self.mouse_size = 224

    def recording(self):
        start = time.time()
        i = 0
        while True:
            ret, img = self.cap.read()
            if ret != True: break
            cv2.imshow("", img)
            k = cv2.waitKey(1)
            if k == 27: break
            elif k == ord("r"):
                failed = self.take_data()
                if failed: continue
                if i == 49:
                    break
                i += 1
            elif k == ord("q"):
                break
            elif k == ord("u"):
                if self.filename_stack != []:
                    filename = self.filename_stack.pop()
                    os.remove(filename + ".npy")
                    print(filename, " removed!")
            end = time.time()
            # print(int(1/(end - start)), "fps")
            start = end
        cv2.destroyAllWindows()

    def take_data(self):
        images = np.array([])
        start = time.time()
        for i in range(20):
            ret, img = self.cap.read()
            if ret != True: return True
            dets, landmarks = self.mtcnn.detect(img, minsize=75)
            if dets.shape[0] == 0: return True
            else:
                img = self.clip(img, dets, landmarks)
                if i == 0: images = img.reshape(1, self.mouse_size, self.mouse_size, 3)
                else: images = np.append(images, img.reshape(1, self.mouse_size, self.mouse_size, 3), axis=0)
                cv2.imshow("", img)
                k = cv2.waitKey(1)
                if i > 5 and k == ord("r") or i == 19:
                    if i < 19:
                        images = np.append(images, np.full((20 - images.shape[0], self.mouse_size, self.mouse_size, 3), -1), axis=0)
                    else:
                        cv2.waitKey(0)
                    now = datetime.datetime.now()
                    filename = "./dataset/" + self.key + "/" + now.strftime('%Y%m%d_%H%M%S')
                    np.save(filename, images)
                    self.filename_stack.append(filename)
                    print(filename)
                    return False
            end = time.time()
            # print(int(1/(end - start)), "fps")
            start = end

    def clip(self, img, boxes, landmarks):
        bb, ll = boxes[0], landmarks[0]
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        # cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLORS[0], 2)
        mw = int((ll[4] - ll[3]) * 1.25)
        mx = int(ll[3] - mw * 0.125)
        mh = int((ll[8] - ll[7]) * 1.5)
        my = int(ll[7] + mh * 0.25)

        img = img[my: my+mh, mx:mx+mw]
        img = cv2.resize(img, (self.mouse_size, self.mouse_size))
        return img


def loop_input(cap):
    mtcnn = TrtMtcnn()
    while True:
        key = input("select: 1 -> 上昇 2 -> 下降 3 -> 前進 4 -> 後退 q -> 終了: ")
        if key == "q":
            print("quit")
            break
        elif key == "1" or key == "2" or key == "3" or key == "4":
            recorder = Recorder(cap, key, mtcnn)
            recorder.recording()

def main():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        loop_input(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

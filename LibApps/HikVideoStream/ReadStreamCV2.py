
import cv2
import threading


class jyReadStreamCV2Base:
    def __init__(self, strVideoName):
        self._strVideoName = strVideoName
        self.__pCap = cv2.VideoCapture()
        self._strUserName = None
        self._strPass = None
        self._strUrl = None
        self._threadVideo = threading.Thread(target=self.captureVideo, args=(), name=strVideoName)
        self._listFrame = []

    def startCaptureVideo(self):
        self._threadVideo.start()

    def captureVideo(self):
        # self.__pCap.open('rtsp://admin:abcd1234@192.168.1.64:554/')
        self.__pCap.open('D:\wyc\Projects\TrainDataSet\Machine/MachineBelt2.mp4')
        if not self.__pCap.isOpened():
            self.__pCap.release()
            return
        ret, frame = self.__pCap.read()
        while ret:
            ret, frame = self.__pCap.read()
            # frame = cv2.resize(frame, (1280, 720))
            frame = frame[0: 1480, 340: 1820]
            frame = cv2.resize(frame, (224, 224))
            cv2.imshow('frame', frame)
            self._listFrame.append(frame)
            #timestamp = [cap.get(cv2.CAP_PROP_FPS)]
            #print(timestamp)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()
        self.__pCap.release()

    def getListFrame(self):
        return self._listFrame
# test = jyReadStreamCV2Base('Test')
# test.startCaptureVideo()

# cap.open('rtsp://admin:abcd1234@192.168.1.21:554/')

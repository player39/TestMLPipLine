
import socket
import threading
import configparser
import hashlib
import os


class jyReadStreamBase:
    def __init__(self, strVideoName):
        self.__strPath = os.getcwd()
        self._strVideoName = strVideoName

        self._pSocket = None
        self._pLinkThread = threading.Thread(target=self.startLink, args=(), name=self._strVideoName)
        self._bLoopFlag = False

        self._pConfig = configparser.ConfigParser()

        if not os.path.exists(self.__strPath + '/Config'):
            os.mkdir(self.__strPath + '/Config')
        self._pConfig.read(self.__strPath + '/Config/%s.ini' % self._strVideoName)

        if not self._pConfig.has_section(self._strVideoName):
            self._pConfig[self._strVideoName] = {
                'bufLen': 1024,
                'defaultServerIP': '127.0.0.1',
                'defaultServerPort': 8080,
                'defaultUrl': 'rtsp://127.0.0.1:8080/',
                'defaultUserAgent': 'python',
                'Username': 'admin',
                'Password': '123456'
            }

        strBufName = 'bufLen'
        if not self._pConfig.has_option(self._strVideoName, strBufName):
            self._pConfig[self._strVideoName][strBufName] = 1024
        self._iBufLen = int(self._pConfig[self._strVideoName][strBufName])

        strIPName = 'defaultServerIP'
        if not self._pConfig.has_option(self._strVideoName, strIPName):
            self._pConfig[self._strVideoName][strIPName] = '127.0.0.1'
        self._strIP = self._pConfig[self._strVideoName][strIPName]

        strPortName = 'defaultServerPort'
        if not self._pConfig.has_option(self._strVideoName, strPortName):
            self._pConfig[self._strVideoName][strPortName] = 8080
        self._iPort = int(self._pConfig[self._strVideoName][strPortName])

        strURLName = 'defaultURL'
        if not self._pConfig.has_option(self._strVideoName, strURLName):
            self._pConfig[self._strVideoName][strURLName] = 'rtsp://127.0.0.1:8080/'
        self._strURL = self._pConfig[self._strVideoName][strURLName]

        strAgent = 'defaultUserAgent'
        if not self._pConfig.has_option(self._strVideoName, strAgent):
            self._pConfig[self._strVideoName][strAgent] = 'python'
        self._strAgent = self._pConfig[self._strVideoName][strAgent]

        strUserName = 'Username'
        if not self._pConfig.has_option(self._strVideoName, strUserName):
            self._pConfig[self._strVideoName][strUserName] = 'admin'
        self._strUsername = self._pConfig[self._strVideoName][strUserName]

        strPass = 'Password'
        if not self._pConfig.has_option(self._strVideoName, strPass):
            self._pConfig[self._strVideoName][strPass] = '123456'
        self._strPass = self._pConfig[self._strVideoName][strPass]
        # self._dictVars = self._pConfig[self._strVideoName]

        with open(self.__strPath + '/Config/%s.ini' % self._strVideoName, 'w') as configfile:
            self._pConfig.write(configfile)

    def startLink(self):
        self._bLoopFlag = True
        while self._bLoopFlag:
            self._pSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._pSocket.settimeout(10)
            tupleIP = (self._strIP, self._iPort)
            iResult = self._pSocket.connect_ex(tupleIP)
            print(iResult)
            if iResult != 10056 and iResult != 0:
                continue

            iSeq = 1
            strOpMSG = self.genMSG_OPTIONS(iSeq)
            self._pSocket.send(str.encode(strOpMSG))
            strOpRes = self._pSocket.recv(self._iBufLen)
            print(strOpRes)
            iSeq += 1

            strDesMSG = self.genMSG_DESCRIBE(iSeq)
            self._pSocket.send(str.encode(strDesMSG))
            strDesRes = bytes.decode(self._pSocket.recv(self._iBufLen))
            print(strDesRes)
            iSeq += 1
            '''
            strRealm = None
            strNonce = None
            for param in strDesRes.split(','):
                iBIndex1 = param.find('realm')
                if iBIndex1 != -1:
                    strRealm = param.split('"')[1]
                iBIndex2 = param.find('nonce')
                if iBIndex2 != -1:
                    strNonce = param.split('"')[1]
            
            strRes1 = hashlib.md5(self._strPass.encode('utf-8')).hexdigest() # hashlib.md5(('%s:%s:%s' % (self._strUsername, strRealm, self._strPass)).encode('utf-8')).hexdigest() # ('DESCRIBE ' + self._strURL).encode('utf-8')
            test = hashlib.md5(('%s:%s:%s' % (self._strUsername, strRealm, self._strPass)).encode('utf-8')).hexdigest()
            # self._strPass = hashlib.md5(self._strPass.encode('utf-8')).hexdigest()
            strRes2 = '1304f42ad80475181d650713f7e727ee' # strNonce# .encode('utf-8')# ((self._strPass + ' ' + '1304f42ad80475181d650713f7e727ee' + ' ') + hashlib.md5(strRes1).hexdigest()).encode('utf-8')
            strRes3 = hashlib.md5(('SETUP:' + self._strURL).encode('utf-8')).hexdigest()
            strMD5 = hashlib.md5((test + ':' + strRes2 + ':' + strRes3).encode('utf-8')).hexdigest()
            '''
            # tupleGroup1 = self.getParamFromRes(strRes=strDesRes)
            tupleGroup = self.getParamFromRes(strDesRes)
            strRealm = tupleGroup[0]
            strNonce = tupleGroup[1]
            strRes = self.genAuthParam('DESCRIBE', strRealm, strNonce)
            strDesAuthor = self.genMSG_DESCRIBEAdmin(iSeq, strRealm, strNonce, strRes)
            self._pSocket.send(str.encode(strDesAuthor))
            strDesAuthRes = bytes.decode(self._pSocket.recv(self._iBufLen))
            print(strDesAuthRes)
            iSeq += 1

            strSetupRes = self.genAuthParam('SETUP', strRealm, strNonce)
            # trackID = 1 is Video
            strSetMSG = self.genMSG_SETUP(iSeq, strRealm, strNonce, strSetupRes, 'trackID=1')
            self._pSocket.send(str.encode(strSetMSG))
            strSetRes = bytes.decode(self._pSocket.recv(self._iBufLen))
            print(strSetRes)
            iSeq += 1

            strPlayRes = self.genAuthParam('PLAY', strRealm, strNonce)
            strSession = self.getParamFromRes(strSetRes)[2]
            strPlayMSG = self.genMSG_PLAY(iSeq, strRealm, strNonce, strPlayRes, strSession)
            self._pSocket.send(str.encode(strPlayMSG))
            while 1:
                strPlayRes = self._pSocket.recv(self._iBufLen)
            # iSeq += 1
                print(strPlayRes)

    # 0: Realm, 1: Nonce, 2: strResMD5
    def genAuthParam(self, strPublicFun, strRealm, strNonce):
        # tupleParam = self.getParamFromRes(strRes)
        # strRealm = tupleParam[0]
        # strNonce = tupleParam[1]
        # strSession = tupleParam[2]
        strRes1 = hashlib.md5(('%s:%s:%s' % (self._strUsername, strRealm, self._strPass)).encode('utf-8')).hexdigest()
        strRes2 = strNonce
        strRes3 = hashlib.md5(('%s:%s' % (strPublicFun, self._strURL)).encode('utf-8')).hexdigest()
        strResMD5 = hashlib.md5((strRes1 + ':' + strRes2 + ':' + strRes3).encode('utf-8')).hexdigest()
        return strResMD5

    def getParamFromRes(self, strRes):
        strRealm = None
        strNonce = None
        strSession = None
        for param in strRes.split(','):
            iBIndex1 = param.find('realm')
            if iBIndex1 != -1:
                strRealm = param.split('"')[1]
            iBIndex2 = param.find('nonce')
            if iBIndex2 != -1:
                strNonce = param.split('"')[1]

        t = strRes.split('\n')
        for param in strRes.split('\n'):
            iBIndex3 = param.find('Session')
            if iBIndex3 != -1:
                strSession = param.split(';')[0].split(':')[1]
        return strRealm, strNonce, strSession

    def genMSG_OPTIONS(self, seq):
        msgRet = 'OPTIONS ' + self._strURL + ' RTSP/1.0\r\n'
        msgRet += 'CSeq: ' + str(seq) + '\r\n'
        msgRet += 'User-Agent: ' + self._strAgent + '\r\n'
        msgRet += '\r\n'
        return msgRet

    def genMSG_DESCRIBE(self, seq):
        msgRet = 'DESCRIBE ' + self._strURL + ' RTSP/1.0\r\n'
        msgRet += 'CSeq: ' + str(seq) + '\r\n'
        msgRet += 'User-Agent: ' + self._strAgent + '\r\n'
        msgRet += 'Accept: application/sdp\r\n'
        msgRet += '\r\n'
        return msgRet

    def genMSG_DESCRIBEAdmin(self, seq, strRealm, strNonce, strRes):

        msgRet = 'DESCRIBE ' + self._strURL + ' RTSP/1.0\r\n'
        msgRet += 'CSeq: ' + str(seq) + '\r\n'
        msgRet += 'User-Agent: ' + self._strAgent + '\r\n'
        msgRet += 'Authorization: Digest username="%s", realm="%s", nonce="%s", uri="%s", response="%s"' % \
                  (self._strUsername, strRealm, strNonce, self._strURL, strRes)
        msgRet += 'Accept: application/sdp\r\n'

        msgRet += '\r\n'
        return msgRet

    def genMSG_SETUP(self, seq, strRealm, strNonce, strRes, strChannel):
        msgRet = 'SETUP %s%s  RTSP/1.0\r\n' % (self._strURL, strChannel)
        msgRet += 'CSeq: ' + str(seq) + '\r\n'
        msgRet += 'User-Agent: ' + self._strAgent
        msgRet += 'Authorization: Digest username="%s", realm="%s", nonce="%s", uri="%s", response="%s"' % \
                  (self._strUsername, strRealm, strNonce, self._strURL, strRes) + '\r\n'
        msgRet += 'Transport: RTP/AVP/TCP;unicast;interleaved=8080-8081\r\n'
        msgRet += '\r\n'
        return msgRet

    def genMSG_PLAY(self, seq, strRealm, strNonce, strRes, strSession):
        msgRet = 'PLAY ' + self._strURL + ' RTSP/1.0\r\n'
        msgRet += 'CSeq: ' + str(seq) + '\r\n'
        msgRet += 'User-Agent: ' + self._strAgent
        msgRet += 'Authorization: Digest username="%s", realm="%s", nonce="%s", uri="%s", response="%s"' % \
                  (self._strUsername, strRealm, strNonce, self._strURL, strRes)
        msgRet += 'Session: ' + strSession + '\r\n'
        msgRet += '\r\n'
        return msgRet

    def genMSG_TEARDOWN(self, seq, strSessionID):
        msgRet = 'TEARDOWN ' + self._strURL + ' RTSP/1.0\r\n'
        msgRet += 'CSeq: ' + str(seq) + '\r\n'
        msgRet += 'User-Agent: ' + self._strAgent
        msgRet += 'Session: ' + strSessionID + '\r\n'
        msgRet += '\r\n'
        return msgRet


test = jyReadStreamBase('test')
test.startLink()

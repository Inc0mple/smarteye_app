from locale import locale_encoding_alias
from streamlit_webrtc import VideoProcessorBase, VideoTransformerBase, webrtc_streamer, WebRtcMode
import streamlit_authenticator as stauth
import cv2
import av
import bcrypt
import streamlit as st
import openpifpaf
import torch
import argparse
import copy
import logging
#import torch.multiprocessing as mp
import csv
import re
import hashlib
import sqlite3
from default_params import *
from algorithms import *
from helpers import last_ip
import os
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import math
import time
import yagmail
from datetime import datetime, timedelta
import mysql.connector

LOCAL = False


onlineDb = mysql.connector.connect(user=st.secrets["db_username"],
    password=st.secrets["db_password"],
    host=st.secrets["db_host"],
    database=st.secrets["db_username"])


if "logged_in" not in st.session_state or "username" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = False

if not os.path.exists("fallImg"):
    os.mkdir("fallImg")
yag = yagmail.SMTP("smarteye012@gmail.com", oauth2_file="client_secret.json")

#yag.send(to="bryantanzy1@gmail.com",subject="test",contents="test")

def get_db_connection(dbPath,local=LOCAL):
    if local:
        print(f"Making local connection to {dbPath}")
        conn = sqlite3.connect(dbPath)
    else:
        print(f"Making online connection to {onlineDb}")
        conn=onlineDb
    return conn

def convertSeconds(seconds):
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02d:%02d:%02d" % (hour, minutes, seconds)

def loginPage():
    with st.form(key='login_form'):
        if "username" not in st.session_state:
            inputUsername = st.text_input("Username")
            inputPassword = st.text_input("Password", type="password")
        else:
            inputUsername = st.text_input("Username")
            inputPassword = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        result = find_user_info("data.db", inputUsername)
        if result:
            userId,username,email,phone,dbPassword = result[0]
            #if password == inputPassword:
            if bcrypt.checkpw(inputPassword.encode("utf-8"), dbPassword.encode("utf-8")):
                #st.success("Succesfully logged in! :tada:")
                st.session_state.logged_in = True
                st.session_state.username = inputUsername
                return True
            else:
                st.error("Invalid Password!")
                return False
        else:
            st.error("Invalid Username!")
            return False

def configPage():
    userConfig = get_config_info(st.session_state.username,"data.db")
    #print(userConfig)
    if not LOCAL:
        userConfig["routLogInterval"] = convertSeconds(userConfig["routLogInterval"].total_seconds())
        userConfig["emailInterval"] = convertSeconds(userConfig["emailInterval"].total_seconds())
    print(userConfig)
    routLogTimes = ["00:15:00","00:30:00","01:00:00","02:00:00","04:00:00","12:00:00","24:00:00","168:00:00","720:00:00"]
    routLogLabels = ["15 Minutes","30 Minutes","Hourly","2-hourly","4-hourly","12-hourly","Daily","Weekly","Monthly"]
    userInfo = find_user_info("data.db",st.session_state.username)[0]
    logTimeDict = dict(zip(routLogTimes,routLogLabels))
    with st.form(key="config_page"):
        logFalls = st.checkbox("Log Falls",value=bool(userConfig["logFalls"]))
        routLogInterval = st.select_slider("Routine Log Interval",format_func = lambda x:logTimeDict[x],
            value=userConfig["routLogInterval"],options=routLogTimes)
        emailLogs = st.checkbox("Send Logs to email",value=bool(userConfig["emailLogs"]))
        emailInterval = st.select_slider("Routine Email Interval",format_func = lambda x:logTimeDict[x],
            value=userConfig["emailInterval"],options=routLogTimes)
        alertEmailFalls = st.checkbox("Email Fall Alerts",value=bool(userConfig["alertEmailFalls"]))
        alertSMSFalls = st.checkbox("SMS Fall Alerts",value=bool(userConfig["alertSMSFalls"]))
        userEmail = st.text_input("Email",value=userInfo[2])
        userPhone = st.text_input("Phone",value=userInfo[3])
        submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        if valid_email(userEmail):
            configDict = {
                "userId":userInfo[0],
                "logFalls":int(logFalls),
                "routLogInterval":routLogInterval,
                "emailLogs":int(emailLogs),
                "emailInterval":emailInterval,
                "alertEmailFalls":int(alertEmailFalls),
                "alertSMSFalls":int(alertSMSFalls),
            }
            change_config("data.db",configDict)
            update_user_info("data.db",st.session_state.username,userEmail,userPhone)
            st.success("User settings successfully changed!")
        else:
            st.error("Invalid Email!")

def userLogPage():
    userInfo = find_user_info("data.db",st.session_state.username)
    userId,username,email,phone= userInfo[0][:-1]
    result = find_user_logs("data.db",userId)
    resultDf = pd.DataFrame(result,columns=table_columns("EventLog","data.db"))
    resultDf["eventId"] = resultDf["eventId"].apply(lambda x:read_event_id(x))
    st.dataframe(resultDf)


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False


def create_table(tableName, dbPath, local=LOCAL):
    if local:
        conn = get_db_connection(dbPath,local)
        cur = conn.cursor()
        sql_as_string = open(f"schemas/{tableName.lower()}Schema.sql").read()
        cur.executescript(sql_as_string)
        print(f"Created {tableName} table in sqllite db")
    else:
        conn = get_db_connection(onlineDb, local)
        print(conn)
        cur = conn.cursor()
        sql_as_string = open(f"schemas/{tableName.lower()}MySQL.sql").read()
        # cur.execute(sql_as_string, multi=True)
        sqlCommands = sql_as_string.split(';')
        for command in sqlCommands:
            if command.strip() != '':
                cur.execute(command)
                conn.commit()
        print(f"Created {tableName} table in MySQL db")

    
def readAll_table(tableName, dbPath,local=LOCAL):
    conn = get_db_connection(dbPath,local)
    cur = conn.cursor()
    sql = "SELECT * from %s;" % tableName
    cur.execute(sql)
    result = cur.fetchall()
    return result

def table_columns(tableName, dbPath):
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    sql = "SELECT * from %s WHERE 1=0;" % tableName
    cur.execute(sql)
    return [d[0] for d in cur.description]

def update_table(tableName, dbPath):
    pass

def get_config_info(inputUsername, dbPath, local=LOCAL):
    # Returns config as dictionary
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    if local:
        cur.execute(f"""
        SELECT c.userId, c.logFalls, c.routLogInterval, c.emailLogs, c.emailInterval, c.alertEmailFalls, c.alertSMSFalls
        FROM Config AS c
        INNER JOIN User AS u ON u.userId = c.userId
        WHERE u.username=?;
        """, [inputUsername])
    else:
        cur.execute(f"""
        SELECT c.userId, c.logFalls, c.routLogInterval, c.emailLogs, c.emailInterval, c.alertEmailFalls, c.alertSMSFalls
        FROM Config AS c
        INNER JOIN User AS u ON u.userId = c.userId
        WHERE u.username=%s;
        """, (inputUsername,))
    
    result = cur.fetchall()
    print(result)
    result = result[0]
    result = {
        "userId":result[0],
        "logFalls":result[1],
        "routLogInterval":result[2],
        "emailLogs":result[3],
        "emailInterval":result[4],
        "alertEmailFalls":result[5],
        "alertSMSFalls":result[6],
    }
    return result

def find_user_info(dbPath, inputUsername):
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    if LOCAL:
        cur.execute("SELECT * from User WHERE username=?;",[inputUsername])
    else:
        cur.execute("SELECT * from User WHERE username=%s;", (inputUsername,))
    result = cur.fetchall()
    return result

def find_user_logs(dbPath, inputId):
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    if LOCAL:
        cur.execute("SELECT * from EventLog WHERE userId=?;",[inputId])
    else:
        cur.execute("SELECT * from EventLog WHERE userId=%s;", (inputId,))
    result = cur.fetchall()
    return result


def update_user_info(dbPath, inputUsername,newEmail,newPhone):
    conn = get_db_connection(dbPath)
    conn.reconnect()
    cur = conn.cursor()
    if LOCAL:
        sql = f"""
        UPDATE User
        SET (email,phone) = ("{newEmail}","{newPhone}")
        WHERE username="{inputUsername}";
        """
    else:
        sql = f"""
        UPDATE User
        SET 
            email = "{newEmail}",
            phone = "{newPhone}"
        WHERE username="{inputUsername}";
        """
    cur.execute(sql)
    conn.commit()
    #conn.close()

def find_all_user_info(dbPath):
    conn = get_db_connection(dbPath,LOCAL)
    cur = conn.cursor()
    cur.execute("SELECT * from User;")
    result = cur.fetchall()
    return result


def delete_table(tableName, dbPath):
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    # Following statement is vulnerable to SQL injections; do not use for production!
    cur.execute(f"DROP TABLE IF EXISTS {tableName}")
    conn.commit()
    #conn.close()

def read_event_id(eventId):
    idDict = {
        0:"No fall",
        1:"Fall detected",
        2:"Device started",
        3:"Device running",
        4:"Device stopped"
    }
    return idDict[eventId]

def valid_email(inputEmailStr):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if(re.fullmatch(regex, inputEmailStr)):
        return True
    else:
        return False

def valid_sg_phone(inputPhone):
    regex= r'/\+65(6|8|9)\d{7}/g'
    if(re.fullmatch(regex, inputPhone)):
        return True
    else:
        return False
    
def alert_email(inputUsername,inputDeviceId,inputEmail,inputImg):
    fallImgName = f"fallImg/{inputUsername}_fall.jpg"
    cv2.imwrite(fallImgName,inputImg)
    yag.send(to=inputEmail,
        subject=f"Fall by SmartEye device!",
        contents=f"Fall detected by {inputUsername}'s device {inputDeviceId} on {datetime.now()}",
        attachments=fallImgName
    )
    for f in os.listdir("fallImg"):
        os.remove(os.path.join("fallImg",f))

def email_device_start(inputUsername,inputDeviceId,inputEmail,timestamp):
    yag.send(to=inputEmail,
        subject=f"SmartEye device started!",
        contents=f"{inputUsername}'s device {inputDeviceId} has started on {timestamp}",
    )

def email_device_stop(inputUsername,inputDeviceId,inputEmail,timestamp):
    yag.send(to=inputEmail,
        subject=f"SmartEye device stopped!",
        contents=f"{inputUsername}'s device {inputDeviceId} has stopped on {timestamp}",
    )

def email_logs(dbPath,inputId, inputEmail,startTime,endTime):
    # Emails all eventLogs between start and end time in tabular form
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    sql = f"SELECT * FROM eventLog WHERE (timestamp between '{startTime}' AND '{endTime}') AND (userId = '{inputId}');"
    # Following statement is vulnerable to SQL injections; do not use for production!
    cur.execute(sql)
    result = cur.fetchall()
    resultDf = pd.DataFrame(result,columns=table_columns("EventLog","data.db"))
    resultDf["eventId"] = resultDf["eventId"].apply(lambda x:read_event_id(x))
    resultsHTML = f"""
    <html>
    <h1>Logs since {startTime}</h1><br>
    <table>
    <tr>
        <th>Index</th>
        <th>userId</th>
        <th>deviceId</th>
        <th>timestamp</th>
        <th>event</th>
    </tr>
    """
    for idx, userId, deviceId, timestamp, eventId in resultDf.itertuples():
        resultsHTML += (f"""
            <tr>
                <th>{idx}</th>
                <th>{userId}</th>
                <th>{deviceId}</th>
                <th>{timestamp}</th>
                <th>{eventId}</th>
            </tr>
        """)
        print(idx, userId, deviceId, timestamp, eventId)
    resultsHTML += ("</table>")
    yag.send(to=inputEmail,
        subject=f"SmartEye device Routine Log",
        contents=resultsHTML
    )


def change_config(dbPath, configDict):
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    if LOCAL:
        sql = f"""
        UPDATE Config
        SET (logFalls,
        routLogInterval,
        emailLogs,
        emailInterval,
        alertEmailFalls,
        alertSMSFalls)
        = ({configDict["logFalls"]},
        "{configDict["routLogInterval"]}",
        {configDict["emailLogs"]},
        "{configDict["emailInterval"]}",
        {configDict["alertEmailFalls"]},
        {configDict["alertSMSFalls"]})
        WHERE userId={configDict["userId"]};
        """
    else:
        sql = f"""
        UPDATE Config
        SET 
            logFalls = {configDict["logFalls"]},
            routLogInterval = "{configDict["routLogInterval"]}",
            emailLogs = {configDict["emailLogs"]},
            emailInterval = "{configDict["emailInterval"]}",
            alertEmailFalls = {configDict["alertEmailFalls"]},
            alertSMSFalls = {configDict["alertSMSFalls"]}
        WHERE userId={configDict["userId"]};
        """
    cur.execute(sql)
    conn.commit()
    #conn.close()

def insert_eventLog(dbPath, inputUsername, deviceId, eventId):
    conn = get_db_connection(dbPath)
    conn.reconnect()
    cur = conn.cursor()
    if LOCAL:
        cur.execute(f"""
        INSERT INTO EventLog (userId,deviceId, eventId)
        VALUES (
            (SELECT userId FROM User
            WHERE User.username = '{inputUsername}')
            , {deviceId}, {eventId} )
        ;""")
    else:
        cur.execute(f"""
        INSERT INTO EventLog (userId,deviceId, eventId)
        VALUES (
            (SELECT userId FROM User
            WHERE User.username = '{inputUsername}')
            , {deviceId}, {eventId} )
        ;""")
    conn.commit()
    #conn.close()

def insert_user(dbPath, username, email, password, phone):
    conn = get_db_connection(dbPath)
    cur = conn.cursor()
    hashedPw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    if LOCAL:
        cur.execute("INSERT INTO User (username,email,phone,password) VALUES (?, ?, ?, ?);", (username,email,phone,hashedPw))
        cur.execute("""
        INSERT INTO Config (userId)
        SELECT userId from User
        WHERE User.username = ?
        ;""", [username])
    else:
        #cur.execute("SELECT * from User WHERE username=%s;", (inputUsername,))
        cur.execute("INSERT INTO User (username,email,phone,password) VALUES (%s, %s, %s, %s);", (username,email,phone,hashedPw))
        cur.execute("""
        INSERT INTO Config (userId)
        SELECT userId from User
        WHERE User.username = %s
        ;""", (username,))
    conn.commit()
    #conn.close()

class VideoProcessorFallLSTM(VideoProcessorBase):
    # https://github.com/taufeeque9/HumanFallDetection
    def __init__(self, username = st.session_state.username, deviceId = 0, t=DEFAULT_CONSEC_FRAMES):
        print('Starting fall detection LSTM...')
        self.consecutive_frames = t
        self.args = self.cli()
        #self.e = mp.Event()
        #self.queues = [mp.Queue()]
        #self.counter1 = mp.Value('i', 0)
        #self.counter2 = mp.Value('i', 0)
        self.argss = [copy.deepcopy(self.args)]
        self.argss[0].video = 0
        self.frame = 0
        self.t0 = time.time()
        self.model = LSTMModel(h_RNN=48, h_RNN_layers=2, drop_p=0.1, num_classes=7)
        self.model.load_state_dict(torch.load('model/lstm_weights.sav',map_location=self.argss[0].device))
        self.model.eval()
        self.output_videos = [None]
        self.ip_sets = [[] for _ in range(self.argss[0].num_cams)]
        self.lstm_sets = [[] for _ in range(self.argss[0].num_cams)]
        self.max_length_mat = 300
        self.num_matched = 0
        self.userDetails = find_user_info("data.db", username)
        self.userId,self.username,self.email,self.phone= self.userDetails[0][:-1]
        self.userConfig = get_config_info(self.username,"data.db")
        self.recent_fall = False
        self.count_fall_recency = False
        self.time_since_fall = 0
        self.deviceId = deviceId
        self.lastRoutLogTime = datetime.now()
        self.lastEmailLogTime = datetime.now()
        if LOCAL:
            self.logIntervalDt = datetime.strptime(self.userConfig["routLogInterval"],"%H:%M:%S")
            self.emailIntervalDt = datetime.strptime(self.userConfig["emailInterval"],"%H:%M:%S")
        else:
            self.logIntervalDt = datetime.strptime(convertSeconds(self.userConfig["routLogInterval"].total_seconds()),"%H:%M:%S")
            self.emailIntervalDt = datetime.strptime(convertSeconds(self.userConfig["emailInterval"].total_seconds()),"%H:%M:%S")
        self.routLogTimeThresh = timedelta(hours=self.logIntervalDt.hour,minutes=self.logIntervalDt.minute,seconds=self.logIntervalDt.second)
        self.routEmailTimeThresh = timedelta(hours=self.emailIntervalDt.hour,minutes=self.emailIntervalDt.minute,seconds=self.emailIntervalDt.second)
        insert_eventLog("data.db",self.username,self.deviceId,2)
        if self.userConfig["emailLogs"]:
            email_device_start(self.username,self.deviceId,self.email,datetime.now())
        #print(f"Init inputvar: {self.inputVar} ")
    
    def __del__(self):
        insert_eventLog("data.db",self.username,self.deviceId,4)
        if self.userConfig["emailLogs"]:
            email_device_stop(self.username,self.deviceId,self.email,datetime.now())

    def cli(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        print(parser)
        openpifpaf.network.Factory.cli(parser)
        openpifpaf.decoder.cli(parser)
        parser.add_argument('--resolution', default=0.3, type=float,
                            help=('Resolution prescale factor from 640x480. '
                                  'Will be rounded to multiples of 16.'))
        parser.add_argument('--resize', default=None, type=str,
                            help=('Force input image resize. '
                                  'Example WIDTHxHEIGHT.'))
        parser.add_argument('--num_cams', default=1, type=int,
                            help='Number of Cameras.')
        parser.add_argument('--video', default=None, type=str,
                            help='Path to the video file.\nFor single video fall detection(--num_cams=1), save your videos as abc.xyz and set --video=abc.xyz\nFor 2 video fall detection(--num_cams=2), save your videos as abc1.xyz & abc2.xyz and set --video=abc.xyz')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='debug messages and autoreload')
        parser.add_argument('--disable_cuda', default=False, action='store_true',
                            help='disables cuda support and runs from gpu')

        vis_args = parser.add_argument_group('Visualisation')
        vis_args.add_argument('--plot_graph', default=False, action='store_true',
                              help='Plot the graph of features extracted from keypoints of pose.')
        vis_args.add_argument('--joints', default=True, action='store_true',
                              help='Draw joint\'s keypoints on the output video.')
        vis_args.add_argument('--skeleton', default=True, action='store_true',
                              help='Draw skeleton on the output video.')
        vis_args.add_argument('--coco_points', default=False, action='store_true',
                              help='Visualises the COCO points of the human pose.')
        vis_args.add_argument('--save_output', default=False, action='store_true',
                              help='Save the result in a video file. Output videos are saved in the same directory as input videos with "out" appended at the start of the title')
        vis_args.add_argument('--fps', default=18, type=int,
                              help='FPS for the output video.')
        # vis_args.add_argument('--out-path', default='result.avi', type=str,
        #                       help='Save the output video at the path specified. .avi file format.')

        args = parser.parse_args()

        # Log
        logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

        args.force_complete_pose = True
        args.instance_threshold = 0.2
        args.seed_threshold = 0.5

        # Add args.device
        args.device = torch.device('cpu')
        args.pin_memory = False
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
            args.pin_memory = True

        if args.checkpoint is None:
            args.checkpoint = 'shufflenetv2k16'

        openpifpaf.decoder.configure(args)
        openpifpaf.network.Factory.configure(args)

        return args

    def recv(self, frame):
        #print(f"recv inputvar: {self.inputVar} ")
        img = frame.to_ndarray(format="bgr24")
        self.frame += 1
        width, height, width_height = resize(img, self.argss[0].resize, self.argss[0].resolution)
        processor_singleton = Processor(width_height, self.argss[0])
        curr_time = time.time()
        img = cv2.resize(img, (width, height))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
        assert bb_list is None or (type(bb_list) == list)
        if bb_list:
            assert type(bb_list[0]) == tuple
            assert type(bb_list[0][0]) == tuple
        # assume bb_list is a of the form [(x1,y1),(x2,y2)),etc.]

        anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
        ubboxes = [(np.asarray([width, height])*np.asarray(ann[1])).astype('int32')
                    for ann in anns]
        lbboxes = [(np.asarray([width, height])*np.asarray(ann[2])).astype('int32')
                    for ann in anns]
        bbox_list = [(np.asarray([width, height])*np.asarray(box)).astype('int32') for box in bb_list]
        uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
        lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
        keypoint_sets = [{"keypoints": keyp[0], "up_hist":uh, "lo_hist":lh, "time":curr_time, "box":box}
                            for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]

        cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
        cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
        for box in bbox_list:
            cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

        dict_frames = {"img": img, "keypoint_sets": keypoint_sets, "width": width, "height": height, "vis_keypoints": self.argss[0].joints,
                    "vis_skeleton": self.argss[0].skeleton, "CocoPointsOn": self.argss[0].coco_points,
                    "tagged_df": {"text": f"Avg FPS: {self.frame//(time.time()-self.t0)}, Frame: {self.frame}", "color": [0, 0, 0]}}
        kp_frames = dict_frames["keypoint_sets"]

        num_matched, new_num, indxs_unmatched = match_ip(self.ip_sets[0], kp_frames, self.lstm_sets[0], self.num_matched, self.max_length_mat)
        valid1_idxs, prediction = get_all_features(self.ip_sets[0], self.lstm_sets[0], self.model)
        dict_frames["tagged_df"]["text"] += f" Pred: {activity_dict[prediction+5]}"
        #print(activity_dict[prediction+5])
        fallStatus = activity_dict[prediction+5]
        timeSinceLastLog = datetime.now() - self.lastRoutLogTime
        timeSinceLastEmail = datetime.now() - self.lastEmailLogTime

        #self.routLogTimeThresh = timedelta(seconds=5)
        #self.routEmailTimeThresh = timedelta(seconds=15)

        if timeSinceLastLog > self.routLogTimeThresh:
            insert_eventLog("data.db",self.username,self.deviceId,3)
            self.lastRoutLogTime = datetime.now()
        if self.userConfig["emailLogs"]:
            if timeSinceLastEmail > self.routEmailTimeThresh:
                email_logs("data.db",self.userId,self.email,self.lastEmailLogTime,datetime.now())
                self.lastEmailLogTime = datetime.now()
        if fallStatus == "FALL":
            self.time_since_fall = 0
            if not self.recent_fall:
                if self.userConfig["logFalls"]:
                    # alert user
                    print("ALERTING USER")
                    #st.warning("Fall detected! Logging event!")
                    insert_eventLog("data.db",self.username,self.deviceId,1)
                    alert_email(self.username,self.deviceId,self.email,img)
                    #st.warning(f"Sending alert to {self.email}!")
                self.recent_fall = True
                self.count_fall_recency = True

        if fallStatus != "FALL" and self.count_fall_recency:
            self.time_since_fall += 1
            if self.time_since_fall >= 100:
                self.recent_fall = False
        #print(f"recent_fall:{self.recent_fall}, count_fall_recency: {self.count_fall_recency}, time_since_fall: {self.time_since_fall}")

        img, self.output_videos[0] = show_tracked_img(dict_frames, self.ip_sets[0], self.num_matched, self.output_videos[0], self.argss[0])

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class VideoProcessorMediapipePose(VideoProcessorBase):
    # https://google.github.io/mediapipe/getting_started/python
    def __init__(self, min_detection_confidence = 0.5, min_tracking_confidence=0.5):
        print('Starting MediapipePose...')
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        #print(f"Init inputvar: {self.inputVar} ")

    def recv(self, frame):
        #print(f"recv inputvar: {self.inputVar} ")
        img = frame.to_ndarray(format="bgr24")
        img.flags.writeable = False
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
            results = pose.process(image)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return av.VideoFrame.from_ndarray(cv2.flip(image, 1), format="bgr24")


def convertFrame(frame):
    r = 750.0 / frame.shape[1]
    dim = (750, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (31,31),0)
    return frame,gray

class VideoProcessorOpenCV(VideoProcessorBase):
    # https://github.com/ashwani227/humanBodyFallDetection/blob/master/Fall%20detection.py
    def __init__(self):
        print('Starting OpenCV Fall Detection...')
        self.count = 0
        self.count1 = 0
        self.slope = 0
        self.slope1 = 100
        self.minArea = 120*100
        self.radianToDegree = 57.324
        self.minimumLengthOfLine = 150.0
        self.minAngle = 18
        self.maxAngle = 72
        self.list_falls = []
        self.count_fall = 0
        self.firstFrame = None

    

    def recv(self, inputFrame):
        #print(f"recv inputvar: {self.inputVar} ")
        frame = inputFrame.to_ndarray(format="bgr24")

        #print("loop: "+ str(len(frame)))
        frame,gray = convertFrame(frame)

        #comparison Frame
        if self.firstFrame is None:
            self.firstFrame = gray

        
        #Frame difference between current and comparison frame
        frameDelta= cv2.absdiff(self. firstFrame,gray)
        #Thresholding
        thresh1 = cv2.threshold(frameDelta,20,255,cv2.THRESH_BINARY)[1]
        #Dilation of Pixels
        thresh = cv2.dilate(thresh1,None,iterations = 15)

        #Finding the Region of Interest with changes
        contour,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        currState = "no fall"
        printColor = [0,0,0]
        for con in contour:

            if len(con)>=5 and cv2.contourArea(con)>self.minArea:
                ellipse = cv2.fitEllipse(con)
                cv2.ellipse(frame,ellipse,(255,255,0),5)

                #Co-ordinates of extreme points
                extTop = tuple(con[con[:, :, 1].argmin()][0])
                extBot = tuple(con[con[:, :, 1].argmax()][0])
                extLeft = tuple(con[con[:, :, 0].argmin()][0])
                extRight = tuple(con[con[:, :, 0].argmax()][0])

                line1 = math.sqrt((extTop[0]-extBot[0])*(extTop[0]-extBot[0])+(extTop[1]-extBot[1])*(extTop[1]-extBot[1]))
                midPoint = [extTop[0]-int((extTop[0]-extBot[0])/2),extTop[1]-int((extTop[1]-extBot[1])/2)]
                if line1>self.minimumLengthOfLine:
                    #cv2.line(frame,(extBot[0],extBot[1]),(extTop[0],extTop[1]), (255, 0, 0), 5)
                    if (extTop[0]!=extBot[0]):
                        self.slope = abs(extTop[1]-extBot[1])/(extTop[0]-extBot[0])

                else:
                    #cv2.line(frame, (extLeft[0], extLeft[1]), (extRight[0], extRight[1]), (255, 0, 0), 5)
                    if (extRight[0] != extLeft[0]):
                        self.slope = abs(extRight[1]-extLeft[1])/(extRight[0]-extLeft[0])
                #print(slope)

                #cv2.line(frame, (midPoint[0], midPoint[1]), (midPoint[0] + 1, midPoint[1] + 100), (255, 255, 255), 5)
                #angle in Radians with perpendicular
                originalAngleP = np.arctan((self.slope1 - self.slope) / (1 + self.slope1 * self.slope))
                #angle with Horizontal
                originalAngleH = np.arctan(self.slope)
                #Angle in degrees
                originalAngleH = originalAngleH*self.radianToDegree
                originalAngleP=originalAngleP*self.radianToDegree
                #print(originalAngleP)
                if (abs(originalAngleP) > self.minAngle and abs(originalAngleH) < self.maxAngle and abs(originalAngleP)+abs(originalAngleH)>89 and abs(originalAngleP)+abs(originalAngleH)<91):
                    self.count += 1
                    if (self.count > 18):
                        self.count_fall+=1
                        #print("Fall detected")
                        self.list_falls.append((time.time()))
                        if(self.count_fall>1):
                            if(self.list_falls[len(self.list_falls)-1]-self.list_falls[len(self.list_falls)-2]<.5):
                                #print (list_falls[len(list_falls)-1]-list_falls[len(list_falls)-2])
                                print("Fall detected")
                                currState = "FALL"
                                printColor = [255,0,0]
                            else:
                                #currState = "no fall"
                                continue

                        self.count = 0
        frame = write_on_image(img=frame,text=f"Total Falls: {self.count_fall}| Current State: {currState}",color=printColor)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")


def tdConverterHHMMSS(x):
    hour = x.hour + (x.days * 24)
    hour = hour if hour > 9 else "0"+str(hour)
    minutes = x.minutes if x.minutes > 9 else "0"+str(x.minutes)
    seconds = x.seconds if x.minutes > 9 else "0"+str(x.seconds)
    return f"{hour}:{minutes}:{seconds}"



st.title("SmartEye")

menu = ["Login/Use Device","User Logs/Settings","Sign Up","Admin Home"]


if st.session_state.logged_in:
    logoutButton = st.button("Logout")
    if logoutButton:    
        st.session_state.logged_in = False
        st.session_state.username = False
        #time.sleep(1)
        st.success("Sucessfully logged out")

choice = st.sidebar.selectbox("Menu",menu)


if choice == "Admin Home":
    st.subheader("Admin Home")
    if st.session_state.username == "admin":
        dbType = st.radio("Choose DB Table to create", ("User","EventLog","Config"))
        if st.button("Create DB Table"):
            create_table(dbType,"data.db")
            st.success(f"Created {dbType} Table")
        if st.button("Delete DB Table"):
            delete_table(dbType,"data.db")
            st.success(f"Deleted {dbType} Table")
        if st.button("Find all"):
            result = readAll_table(dbType,"data.db")
            resultDf = pd.DataFrame(result,columns=table_columns(dbType,"data.db"))
            if dbType == "EventLog":
                resultDf["eventId"] = resultDf["eventId"].apply(lambda x:read_event_id(x))
                #print(resultDf)
            if dbType == "Config":
                if (type(resultDf["routLogInterval"][0]) == pd._libs.tslibs.timedeltas.Timedelta) and (not resultDf.empty):
                    resultDf["routLogInterval"] = resultDf["routLogInterval"].apply(lambda x:convertSeconds(x.total_seconds()))
                    resultDf["emailInterval"] = resultDf["emailInterval"].apply(lambda x:convertSeconds(x.total_seconds()))
            st.dataframe(resultDf)
    else:
        st.info("Log in as admin to access")


elif choice == "Login/Use Device":

    if not st.session_state.logged_in:
        if loginPage():
            st.success("Logged In as {}".format(st.session_state.username))
    if st.session_state.logged_in:
        st.write('Welcome *%s*' % (st.session_state.username))
        task = st.selectbox("Task",["SmartEye Fall Detection","MediaPipe Pose Estimation","OpenCV Fall Detection"])


        if task == "SmartEye Fall Detection":
            st.subheader("SmartEye Fall Detection")
            webrtc_ctx = webrtc_streamer(
                key="fall-detection",
                video_processor_factory=VideoProcessorFallLSTM, 
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            # Pass config file retrieved from database to the following variable
            myVar = "hi"
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.username = st.session_state.username
        elif task == "MediaPipe Pose Estimation":
            st.subheader("MediaPipe Pose Estimation")
            webrtc_ctx2 = webrtc_streamer(
                key="fall-detection2",
                video_processor_factory=VideoProcessorMediapipePose, 
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            detectionConf = st.slider(
                "min_detection_confidence", 0.0, 1.0, 0.5, 0.05
            )
            trackingConf = st.slider(
                "min_tracking_confidence", 0.0, 1.0, 0.5, 0.05
            )
            # Pass config file retrieved from database to the following variable
            if webrtc_ctx2.video_processor:
                    webrtc_ctx2.video_processor.min_detection_confidence = detectionConf
                    webrtc_ctx2.video_processor.min_tracking_confidence = trackingConf

        elif task == "OpenCV Fall Detection":
            st.subheader("OpenCV Fall Detection")
            webrtc_ctx3 = webrtc_streamer(
                key="fall-detection3",
                video_processor_factory=VideoProcessorOpenCV, 
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            # Pass config file retrieved from database to the following variable
            # myVar = "hi"
            #if webrtc_ctx3.video_processor:
            #        webrtc_ctx3.video_processor.inputVar = myVar

            #user_result = view_all_users()
            #clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
            #st.dataframe(clean_db)
        
    #username = st.sidebar.text_input("User Name")
    #password = st.sidebar.text_input("Password",type='password')
    

elif choice == "Sign Up":
    st.subheader("Create New Account")
    newUser = st.text_input("Username")
    newPassword = st.text_input("Password",type='password')
    newEmail = st.text_input("Email")
    newPhone = st.text_input("Phone")

    if st.button("Signup"):

        if not valid_email(newEmail):
            st.error("Invalid Email!")
        else:
        # TO-DO: Verify valid and non-duplicated username (using email verifier api)
            existingUser = find_user_info("data.db", newUser)
            if not existingUser:
            # create new table and upload to mysql database
                insert_user("data.db",newUser,newEmail,newPassword,newPhone)
                st.success("You have successfully created an account!")
                st.info("Go to Login Menu to login")
                print(find_all_user_info("data.db"))
            else:
                st.error("User already exists!")

if choice == "User Logs/Settings":
    st.subheader("User Setttings")
    if not st.session_state.logged_in or not st.session_state.username:
        st.info("Log in required to access configs")
    else:
        st.write('Welcome *%s*' % (st.session_state.username))
        configPage()
        st.subheader("Event Log")
        userLogPage()
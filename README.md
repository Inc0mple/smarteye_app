# SmartEye/HealthEye

AI web app for automatic fall detection with email notifiation using Python, Streamlit, SQL and OpenPifPaf. USES INTEGRATED CAMERA ON LAPTOPS.

https://user-images.githubusercontent.com/42108450/190897690-0dfd0d81-b11f-493a-853a-c4641d64ccc0.mp4

# Features

- Fall detection (SmartEye)
- MediaPipe Pose estimation
- Lightweight fall-detection using OpenCV
- Login and Signup (Local database)
- Email notification (currently not working due to expiration of free SQL database account)

# Setup (Tested on Windows 10)

1. Setup virtual environment `python -m venv env`
2. Activate virutal environment `.\env\Scripts\activate`
3. `pip install -m -r requirements.txt`
4. Go to `env\Lib\site-packages\openpifpaf\decoder\cifcaf.py`
5. Comment out line 119 (`assert not cls.force_complete`)
6. Run streamlit `streamlit run app.py`
7. Login with username "admin" and password "password"

# SmartEye/HealthEye

AI web app for automatic fall detection with email notifiation using Python, Streamlit, SQL and OpenPifPaf. USES INTEGRATED CAMERA ON LAPTOPS.

https://user-images.githubusercontent.com/42108450/190652415-55ec144c-7ee2-47aa-9fc0-da17a5cf6db9.mp4

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

from flask import Flask, render_template, request, jsonify, Response
import cv2
from ultralytics import YOLO
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import os
import threading
from PIL import Image, ImageDraw, ImageFont  # 한글 텍스트 추가용
import numpy as np

app = Flask(__name__)

stop_flag = False  # 작동 중지 플래그
camera_thread = None
video_capture = None


def get_gps_location():
    """GPS 기반 위치 정보를 가져옵니다 (GPS 모듈 필요)"""
    try:
        import gpsd
        gpsd.connect()
        packet = gpsd.get_current()
        if packet.mode >= 2:  # 2D 또는 3D 고정
            latitude = packet.lat
            longitude = packet.lon
            print(f"✅ GPS 위치 가져오기 성공: {latitude}, {longitude}")
            return latitude, longitude
        else:
            print("❌ GPS 신호를 찾을 수 없습니다.")
            return None, None
    except Exception as e:
        print(f"❌ GPS 데이터 가져오기 실패: {e}")
        return None, None

def get_ip_location():
    """IP 기반 위치 정보를 가져옵니다 (ipinfo.io 사용)"""
    try:
        response = requests.get("https://ipinfo.io/json", timeout=10)
        data = response.json()
        if "loc" in data:
            latitude, longitude = map(float, data["loc"].split(","))
            print(f"✅ IP 기반 위치 가져오기 성공: {latitude}, {longitude}")
            return latitude, longitude
        else:
            print("❌ IP 기반 위치 정보를 가져올 수 없습니다.")
            return None, None
    except Exception as e:
        print(f"❌ IP 위치 가져오기 실패: {e}")
        return None, None

def get_location_google(api_key):
    """Google Geolocation API를 사용하여 위치 정보를 가져옵니다."""
    url = "https://www.googleapis.com/geolocation/v1/geolocate"
    headers = {"Content-Type": "application/json"}
    payload = {"considerIp": True}

    try:
        response = requests.post(f"{url}?key={api_key}", json=payload)
        response_data = response.json()
        if "location" in response_data:
            latitude = response_data["location"]["lat"]
            longitude = response_data["location"]["lng"]
            print(f"✅ Google Geolocation API 위치 가져오기 성공: {latitude}, {longitude}")
            return latitude, longitude
        else:
            print(f"❌ Google API에서 위치를 가져올 수 없습니다: {response_data}")
            return None, None
    except Exception as e:
        print(f"❌ Google Geolocation API 호출 실패: {e}")
        return None, None

def get_location():
    """GPS, Google Geolocation, 또는 IP 기반 위치 정보를 가져옵니다."""
    latitude, longitude = get_gps_location()
    if latitude is None or longitude is None:
        google_api_key = "AIzaSyCV9CsNZvxpzNFn2-VKBG1RkCaZUF7a5WM"  # Google API 키 입력
        latitude, longitude = get_location_google(google_api_key)
    if latitude is None or longitude is None:
        latitude, longitude = get_ip_location()
    return latitude, longitude

def generate_google_maps_url(latitude, longitude):
    """Google Maps URL 생성"""
    if latitude and longitude:
        return f"https://www.google.com/maps?q={latitude},{longitude}"
    else:
        return "위치를 가져올 수 없습니다."

def draw_text_with_pillow(image, text, position, font_path="malgun.ttf", font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def send_email_with_location(sender_email, receiver_email, subject, body, app_password, attachment_path, latitude, longitude):
    """위치 정보를 포함하여 이메일 전송"""
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    location_url = generate_google_maps_url(latitude, longitude)
    if location_url != "위치를 가져올 수 없습니다.":
        body += f"\n\n위치 확인: {location_url}"
    else:
        body += "\n\n위치를 가져올 수 없습니다."

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as file:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(file.read())
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
        msg.attach(attachment)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"✅ 이메일 전송 완료: {receiver_email}")
    except Exception as e:
        print(f"❌ 이메일 전송 실패: {e}")
    finally:
        server.quit()

def detection_loop(model_path, camera_index, sender_email, receiver_email, subject, body, app_password):
    global stop_flag, video_capture
    model = YOLO(model_path)
    video_capture = cv2.VideoCapture(camera_index)

    if not video_capture.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("✅ 카메라 작동 중...")

    while not stop_flag:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ 프레임 읽기 실패.")
            break

        results = model.predict(source=frame, save=False, conf=0.25)
        detected = False

        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if cls == 0:
                detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"땅 손상 ({conf:.2f})"
                frame = draw_text_with_pillow(frame, label, (x1, y1 - 20))

        if detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"detected_pothole_{timestamp}.jpg"
            cv2.imwrite(file_name, frame)

            latitude, longitude = get_location()
            send_email_with_location(
                sender_email,
                receiver_email,
                subject,
                body,
                app_password,
                file_name,
                latitude,
                longitude
            )

    video_capture.release()
    print("⛔ 카메라 작동 중지.")

def start_camera_thread(*args):
    global camera_thread
    camera_thread = threading.Thread(target=detection_loop, args=args)
    camera_thread.daemon = True
    camera_thread.start()

def stop_camera():
    global stop_flag, video_capture
    stop_flag = True
    if video_capture:
        video_capture.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_camera():
    global stop_flag
    stop_flag = False
    model_path = "yolov8n.pt"
    sender_email = "wwwe7701@gmail.com"
    receiver_email = "24s620h0659@sonline20.sen.go.kr"
    app_password = "oshi hrrt cnjd qvrh"
    subject = "도로 이상 감지됨!"
    body = "도로 이상이 감지되었습니다. 확인 바랍니다."
    start_camera_thread(model_path, 0, sender_email, receiver_email, subject, body, app_password)
    return jsonify({"status": "Camera started"})

@app.route('/stop', methods=['POST'])
def stop_camera_route():
    stop_camera()
    if camera_thread:
        camera_thread.join()
    return jsonify({"status": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    def generate():
        global video_capture
        while True:
            if video_capture:
                ret, frame = video_capture.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

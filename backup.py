import cv2
from ultralytics import YOLO
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import requests
import os
import pygame  # 경고음 재생 라이브러리
from PIL import Image, ImageDraw, ImageFont  # 한글 텍스트 추가용
import numpy as np

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

def send_email_with_location(sender_email, receiver_email, subject, body, app_password, attachment_path, latitude, longitude):
    """위치 정보를 포함하여 이메일 전송"""
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = Header(subject, 'utf-8')

    # 위치 URL 생성
    location_url = generate_google_maps_url(latitude, longitude)
    if location_url != "위치를 가져올 수 없습니다.":
        body += f"\n\n위치 확인: {location_url}"
    else:
        body += "\n\n위치를 가져올 수 없습니다."

    # 본문 추가
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # 감지된 이미지 추가
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as file:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(file.read())
        encoders.encode_base64(attachment)
        attachment.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(attachment_path)}"
        )
        msg.attach(attachment)
    else:
        print(f"❌ 감지된 이미지 파일을 찾을 수 없습니다: {attachment_path}")

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"✅ 이메일이 성공적으로 전송되었습니다: {receiver_email}")
    except Exception as e:
        print(f"❌ 이메일 전송 중 오류가 발생했습니다: {e}")
    finally:
        server.quit()

def play_alert_sound(alert_sound):
    """경고음 재생"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play()
        print("🔊 경고음이 재생됩니다.")
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # 음악이 끝날 때까지 대기
    except Exception as e:
        print(f"❌ 경고음 재생 중 오류 발생: {e}")

def draw_text_with_pillow(image, text, position, font_path="malgun.ttf", font_size=20, color=(0, 255, 0)):
    """Pillow로 한글 텍스트를 이미지에 렌더링"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def capture_once_on_detection(model_path, camera_index, sender_email, receiver_email, subject, body, app_password, alert_sound):
    """감지 시 이메일 및 경고 기능 실행"""
    # YOLO 모델 로드
    model = YOLO(model_path)

    # USB 카메라 초기화
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ 카메라를 열 수 없습니다. 카메라 인덱스를 확인하세요: {camera_index}")
        return

    print(f"✅ USB 카메라 연결 성공 (인덱스: {camera_index}). '땅 손상'이 감지되면 캡처, 이미지 저장 및 이메일 전송 후 종료됩니다.")

    latitude, longitude = get_location()

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다. 연결 상태를 확인하세요.")
            break

        # YOLO 모델 예측
        results = model.predict(source=frame, save=False, conf=0.25)  # 신뢰도 0.25로 설정
        detected = False  # "땅 손상" 감지 여부

        # 감지된 프레임에 네모 박스와 라벨 추가
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 좌표를 정수로 변환

            if cls == 0:  # 'pothole' 클래스
                detected = True
                # 네모 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 한글 라벨 추가 (Pillow 사용)
                label = f"땅 손상 ({conf:.2f})"
                frame = draw_text_with_pillow(frame, label, (x1, y1 - 20), font_path="malgun.ttf", font_size=20, color=(0, 255, 0))

        if detected:
            # 경고음 재생
            play_alert_sound(alert_sound)

            # 감지된 이미지 저장
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"detected_pothole_{timestamp}.jpg"
            cv2.imwrite(file_name, frame)

            # 이메일 전송
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
            break

        # 감지되지 않아도 실시간으로 프레임 표시
        cv2.imshow("Live Feed", frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 설정
    sender_email = "wwwe7701@gmail.com"
    receiver_email = "24s620h0659@sonline20.sen.go.kr"
    app_password = "oshi hrrt cnjd qvrh"
    alert_sound = r"C:\Users\LG\Desktop\road\ca.wav"
    model_path = "yolov8n.pt"
    subject = "땅 손상 감지됨!"
    body = "AI 시스템이 땅 손상을 감지했습니다. 첨부된 이미지를 확인하세요."

    # 카메라 실행
    capture_once_on_detection(
        model_path,
        camera_index=0,  # USB 카메라 인덱스로 설정
        sender_email=sender_email,
        receiver_email=receiver_email,
        subject=subject,
        body=body,
        app_password=app_password,
        alert_sound=alert_sound
    )

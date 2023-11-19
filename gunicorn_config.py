# gunicorn_config.py

bind = "0.0.0.0:8000"  # Gunicorn이 수신할 IP 주소 및 포트
workers = 3  # Worker 프로세스 수, 적절한 값을 설정하세요
threads = 2  # 각 Worker에서 실행할 스레드 수
worker_class = "sync"  # Worker 클래스, 예: "sync", "eventlet", "gevent"
timeout = 240  # 요청을 처리하는 최대 시간 (초)

# 로깅 설정
accesslog = "/home/ubuntu/access.log"
errorlog = "/home/ubuntu/error.log"

# Daemon 모드로 실행할 경우 pid 파일 경로 설정
pidfile = "/home/ubuntu/gunicorn.pid"

# 프로젝트의 WSGI 애플리케이션 설정
# mydjango.wsgi:application은 프로젝트에 따라 변경되어야 합니다.
# 예: "myproject.wsgi:application"
pythonpath = "/home/ubuntu/songssam_ml"
wsgi_app = "mydjango.wsgi:application"


<img src="https://capsule-render.vercel.app/api?type=waving&color=037bfc&height=150&section=header" />

## 파란학기 SongSSam 프로젝트의 모델 서빙을 위한 서버
## 목차
[API 명세](#API-명세)

[전처리 순서도](#음성-파일-전처리-순서도)

## API 명세
### 전처리 요청
* Request URL: `/songssam/splitter`
* Method: `POST`
* RequestBody:
  ```
  {
    fileKey: "[s3에 저장된 wav파일의 key]",
    isUser: "[사용자의 목소리면 true 음원파일이면 false]",
    uuid: "[처리 완료된 후, s3에 저장될 파일 이름의 key값]"
  }
  ```
* Success Response: (HTTP Status 200)
  ```
  {
    "message": "[f0,f1,f2,...,f7]"
  }
  #사용자혹은 음원에서 노이즈를 제거한 음성파일의 주파수를 클러스터링하여 8가지 대표 주파수 반환 
  ```
  
* Error Response: (HTTP Status 404)Request Body 부재
  ```
  {
    "Error": "error"
  }
  ```
### DDSP-SVC 보컬곡 생성 요청
* Request URL: `/songssam/voiceChangeModel`
* Method: `POST`
* RequestBody:
  ```
  {
    "wav_path": "[splitter를 통해 분리한 음원의 배경 wav파일이 저장된 s3의 키값]",
    "fPtrPath": "[DDSP-SVC의 train 모듈에서 사용자의 목소리를 학습하여 생성된 ptr 파일이 저장된 s3의 키값"],
    "uuid": "[추론을 통해 생성된 파일이 저장될 이름]"
  }
  ```
* Success Response: (HTTP Status 200)
  ```
  {
    "uuid": "생성된 파일이 저장된 키 값"
  }
  ```
* Error Response: (HTTP Status 404)Request Body 부재
  ```
  {
    "Error": "error"
  }
  ```
## 음성 파일 전처리 순서도
![image](https://github.com/ajou20658/songssam_ml/assets/48721887/50d73898-0661-4b7b-a907-b4fb5b1d8ea8)
<img src="https://capsule-render.vercel.app/api?type=waving&color=037bfc&height=150&section=footer" />

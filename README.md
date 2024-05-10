<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=header" />

## 파란학기 SongSSam 프로젝트의 모델 서빙을 위한 서버
## 목차
[API 명세](#API-명세)

[전처리 순서도](#음성-파일-전처리-순서도)

## API 명세
### 전처리 요청
* Request URL: `/songssam/splitter?fileKey=[s3::wav_file_Url]&isUser=[boolean]&uuid=[uuid for_s3_zip_filename]`
* Method: `GET`
### DDSP-SVC 보컬곡 생성 요청
* Request URL: `/songssam/voiceChangeModel`
* Method: `POST`
* RequestBody: ``

## 음성 파일 전처리 순서도
![image](https://github.com/ajou20658/songssam_ml/assets/48721887/50d73898-0661-4b7b-a907-b4fb5b1d8ea8)
<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=footer" />

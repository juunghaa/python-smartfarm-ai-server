# SmartFarm AI Server

식물 이미지 기반 질병 판별을 위한 PyTorch 기반 AI 추론 서버입니다.  
업로드된 이미지를 분석하여 질병 여부를 판단하고 JSON 형태로 결과를 반환합니다.

---

## Overview

이 서버는 스마트팜 시스템의 AI 컴포넌트로, 다음과 같은 흐름으로 동작합니다:

이미지 업로드 → AI 서버 → 모델 추론 → 결과 반환

---

## Model

- Framework: PyTorch  
- Architecture: ResNet18  
- Task: Image Classification (Healthy vs Disease)

### Classes

disease  
healthy  

---

## Architecture

Frontend  
↓  
Node.js Backend  
↓ (HTTP 요청)  
FastAPI AI Server  
↓  
PyTorch Model (.pth)  
↓  
Prediction JSON Response  

---

## Project Structure

smartfarm-ai-server/  
├── main.py  
├── best_model.pth  
├── requirements.txt  
└── README.md  

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# SmartFarm AI Disease Detection Server

## Run Server

```bash
uvicorn main:app --reload --port 10000
```

서버 실행 후:

[http://localhost:10000](http://localhost:10000)

---

## API

### Health Check

```http
GET /health
```

Response:

```json
{
  "ok": true,
  "classes": ["disease", "healthy"]
}
```

---

### Image Prediction

```http
POST /predict
```

Request:

```bash
curl -X POST http://localhost:10000/predict \
-F "file=@test.jpg"
```

Response:

```json
{
  "result": "healthy",
  "label": "healthy",
  "confidence": 0.9823,
  "probabilities": {
    "disease": 0.0177,
    "healthy": 0.9823
  },
  "message": "현재 이미지에서는 뚜렷한 질병 징후가 보이지 않습니다."
}
```

---

## Features

* PyTorch 기반 이미지 분류 추론
* FastAPI 기반 REST API 서버
* multipart/form-data 이미지 업로드 지원
* 클래스 자동 매핑 (checkpoint 기반)
* Softmax 확률 반환
* 디버깅 로그 출력
* CORS 설정 지원

---

## Integration (Node Backend)

Node.js에서 호출 예시:

```js
const formData = new FormData();
formData.append("file", fs.createReadStream(imagePath));

const response = await axios.post(
  "http://localhost:8000/predict",
  formData,
  { headers: formData.getHeaders() }
);
```

---

## Deployment

Render 또는 HuggingFace Spaces 등에 배포 가능합니다.

### Render 설정 예시

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

## Purpose

이 서버는 스마트팜 시스템에서 다음 기능을 담당합니다:

* 식물 상태 자동 분석
* 질병 조기 감지
* 사용자 알림 및 리포트 생성

---

## Future Work

* 다중 질병 클래스 확장
* 객체 탐지 기반 병해 위치 분석
* 모델 성능 개선 및 경량화
* 클라우드 배포 최적화

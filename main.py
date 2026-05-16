from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, os, logging, gc, numpy as np
from pathlib import Path
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.onnx"
CLASSES_PATH = BASE_DIR / "classes.txt"  # 아래에서 설명
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 512  # 1024 → 512로 추가 축소
INFER_IMAGE_SIZE = 224

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartfarm-ai")

app = FastAPI(title="SmartFarm Disease Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONNX Session 옵션 (메모리 최소화)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
logger.info("✅ ONNX model loaded")

# classes.txt에서 로드 (없으면 기본값)
if CLASSES_PATH.exists():
    classes = CLASSES_PATH.read_text().strip().splitlines()
else:
    classes = ["disease", "healthy"]
logger.info("Classes: %s", classes)

# Normalize 상수 (NumPy용)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

def preprocess(image: Image.Image) -> np.ndarray:
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    image = image.resize((INFER_IMAGE_SIZE, INFER_IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0  # (H,W,3)
    arr = arr.transpose(2, 0, 1)                      # (3,H,W)
    arr = (arr - MEAN) / STD
    return arr[np.newaxis, :]                          # (1,3,H,W)

MESSAGES = {
    "healthy": "현재 이미지에서는 뚜렷한 질병 징후가 보이지 않습니다.",
    "disease": "질병이 의심됩니다. 잎의 상태를 확인해주세요.",
}

@app.get("/health")
def health():
    return {"ok": True, "classes": classes}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지 파일 크기가 너무 큽니다.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_arr = preprocess(image)

        outputs = session.run(None, {"input": input_arr})
        logits = outputs[0][0]                    # (num_classes,)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()                   # softmax

        pred_idx = int(np.argmax(probs))
        label = classes[pred_idx]
        conf = float(probs[pred_idx])

        logger.info("pred=%s conf=%.4f probs=%s", label, conf, probs.round(4).tolist())

        return {
            "result": label,
            "label": label,
            "class_index": pred_idx,
            "confidence": round(conf, 4),
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
            "message": MESSAGES.get(label, f"{label}로 분류되었습니다."),
        }
    finally:
        del image_bytes
        gc.collect()
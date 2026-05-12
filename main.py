from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parse_classes_from_checkpoint(checkpoint: dict) -> list[str]:
    classes = checkpoint.get("classes", ["disease", "healthy"])
    class_to_idx = checkpoint.get("class_to_idx")

    if isinstance(class_to_idx, dict) and class_to_idx:
        # class_to_idx 기준으로 인덱스->클래스 매핑을 고정해 label 뒤바뀜 방지
        classes_by_idx = [None] * (max(class_to_idx.values()) + 1)
        for name, idx in class_to_idx.items():
            classes_by_idx[idx] = name
        if all(c is not None for c in classes_by_idx):
            classes = classes_by_idx

    return classes


def load_model_and_classes(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    logger.info("Loaded checkpoint type: %s", type(checkpoint))

    model = None
    classes = ["disease", "healthy"]

    if isinstance(checkpoint, dict):
        classes = parse_classes_from_checkpoint(checkpoint)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info("Checkpoint format: dict with model_state_dict")
        else:
            state_dict = checkpoint
            logger.info("Checkpoint format: pure state_dict-like dict")

        model = build_resnet18(num_classes=len(classes))

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("State dict load summary | missing_keys=%d unexpected_keys=%d", len(missing), len(unexpected))
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)

        # 분류 헤드 shape이 맞는지 강제 검증
        fc_w = state_dict.get("fc.weight")
        fc_b = state_dict.get("fc.bias")
        if fc_w is not None and fc_w.shape[0] != len(classes):
            logger.warning(
                "fc.weight out_features(%s) != num_classes(%s)",
                fc_w.shape[0],
                len(classes),
            )
        if fc_b is not None and fc_b.shape[0] != len(classes):
            logger.warning(
                "fc.bias out_features(%s) != num_classes(%s)",
                fc_b.shape[0],
                len(classes),
            )

    else:
        # 사용자가 요청한 fallback: checkpoint 자체가 모델 객체인 경우
        model = checkpoint
        logger.info("Checkpoint format: serialized full model object")

        classes = getattr(model, "classes", classes)

    model.to(device)
    model.eval()
    logger.info("Model ready on %s | classes=%s", device, classes)
    return model, classes


model, classes = load_model_and_classes(MODEL_PATH, device)

# ResNet pretrained 학습이었다면 보통 ImageNet Normalize를 사용
# 학습 코드 부재 시 기본 True로 두고, 필요 시 환경변수로 비활성화 가능
use_imagenet_norm = os.getenv("USE_IMAGENET_NORMALIZE", "1") == "1"

transform_steps = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]
if use_imagenet_norm:
    transform_steps.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

transform = transforms.Compose(transform_steps)
logger.info("Transform configured | resize=224x224 to_tensor=True imagenet_normalize=%s", use_imagenet_norm)

MESSAGES = {
    "healthy": "현재 이미지에서는 뚜렷한 질병 징후가 보이지 않습니다.",
    "disease": "질병이 의심됩니다. 잎의 상태를 확인해주세요.",
}


@app.get("/health")
def health():
    return {
        "ok": True,
        "classes": classes,
        "use_imagenet_normalize": use_imagenet_norm,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_idx_int = int(pred_idx.item())
    if pred_idx_int < 0 or pred_idx_int >= len(classes):
        raise HTTPException(status_code=500, detail="예측 인덱스가 클래스 범위를 벗어났습니다.")

    label = classes[pred_idx_int]
    conf = float(confidence.item())

    # 디버깅 로그
    logger.info("Predict debug | logits=%s", logits.detach().cpu().numpy().round(6).tolist())
    logger.info("Predict debug | probs=%s", probs.detach().cpu().numpy().round(6).tolist())
    logger.info("Predict debug | pred_idx=%d class_name=%s confidence=%.6f", pred_idx_int, label, conf)

    return {
        "result": label,
        "label": label,
        "class_index": pred_idx_int,
        "confidence": round(conf, 4),
        "probabilities": {
            classes[i]: float(probs[0, i].item()) for i in range(len(classes))
        },
        "message": MESSAGES.get(label, f"{label}로 분류되었습니다."),
    }

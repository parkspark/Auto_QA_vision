# YOLO11 학습 (RTX 5090 32GB VRAM / 64GB RAM / i7-12700K 기준 설정)
from ultralytics import YOLO


def main():
    model = YOLO("C:/Users/park/Desktop/MINI_DATA_PROJECT/models/yolo11s.pt")
    model.train(
        data="C:/Users/park/Desktop/MINI_DATA_PROJECT/df_dataset.yaml",
        imgsz=1280,     # user_id(작은 닉네임 박스) 보존을 위한 최소 안전 크기
        batch=8,        # VRAM 제약값 (AutoBatch 실측, RAM 증설과 무관)
        epochs=80,
        patience=20,
        amp=True,
        cache="disk",   # Windows: cache="ram"은 spawn 시 캐시 직렬화로 워커 크래시(pickle truncated).
                        # disk 캐시(.npy)는 워커에 경로만 넘기고, 64GB RAM이 OS 페이지캐시로 흡수 → RAM 속도
        workers=12,     # i7-12700K(20스레드). 멀티프로세스 증강으로 5090 공급
        project="C:/Users/park/Desktop/MINI_DATA_PROJECT/runs",
        name="df_yolo11s_1280_v4",  # 영상 프레임 검수 반영(NPC 오탐 107개 제거)
    )


if __name__ == "__main__":
    main()

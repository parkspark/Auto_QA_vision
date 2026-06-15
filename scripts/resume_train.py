# 중단된 v3 학습을 last.pt에서 이어서 재개 (전원 크래시 복구용)
# 원래 train() 인자는 체크포인트에 저장돼 있어 resume=True만으로 동일 설정으로 이어진다.
from ultralytics import YOLO

LAST = "C:/Users/park/Desktop/MINI_DATA_PROJECT/runs/df_yolo11s_1280_v3/weights/last.pt"


def main():
    model = YOLO(LAST)
    model.train(resume=True)


if __name__ == "__main__":
    main()

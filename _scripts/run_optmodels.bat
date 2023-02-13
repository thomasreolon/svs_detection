
python main.py --exp_name comparison/models/_yolo8    --architecture opt_yolo8    --epochs 160 --framerate 2 --simulator static --debug --triggering
python main.py --exp_name comparison/models/_yolophi  --architecture opt_yolophi  --epochs 160 --framerate 2 --simulator static --debug
python main.py --exp_name comparison/models/_mlp2     --architecture opt_mlp2     --epochs 160 --framerate 2 --simulator static --debug

python main.py --exp_name comparison/models/f2s123   --architecture opt_yolo8  --epochs 50 --framerate 2 --simulator static --svs_close 1 --svs_open 2 --svs_hot 3 --pretrained comparison/models/_yolo8/model.pt  
python main.py --exp_name comparison/models/f2s135   --architecture opt_yolo8  --epochs 50 --framerate 2 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained comparison/models/_yolo8/model.pt
python main.py --exp_name comparison/models/f2s324   --architecture opt_yolo8  --epochs 50 --framerate 2 --simulator static --svs_close 3 --svs_open 2 --svs_hot 4 --pretrained comparison/models/_yolo8/model.pt

python main.py --exp_name comparison/models/f05s123  --architecture opt_yolo8  --epochs 50 --framerate .5 --simulator static --svs_close 1 --svs_open 2 --svs_hot 3 --pretrained comparison/models/_yolo8/model.pt
python main.py --exp_name comparison/models/f05s135  --architecture opt_yolo8  --epochs 50 --framerate .5 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained comparison/models/_yolo8/model.pt
python main.py --exp_name comparison/models/f05s324  --architecture opt_yolo8  --epochs 50 --framerate .5 --simulator static --svs_close 3 --svs_open 2 --svs_hot 4 --pretrained comparison/models/_yolo8/model.pt

python main.py --exp_name comparison/models/f10s123  --architecture opt_yolo8  --epochs 50 --framerate 10 --simulator static --svs_close 1 --svs_open 2 --svs_hot 3 --pretrained comparison/models/_yolo8/model.pt
python main.py --exp_name comparison/models/f10s135  --architecture opt_yolo8  --epochs 50 --framerate 10 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained comparison/models/_yolo8/model.pt
python main.py --exp_name comparison/models/f10s324  --architecture opt_yolo8  --epochs 50 --framerate 10 --simulator static --svs_close 3 --svs_open 2 --svs_hot 4 --pretrained comparison/models/_yolo8/model.pt
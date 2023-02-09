

python main.py --exp_name runs/baseline/fr2  --architecture yolophi --epochs 100 --framerate 2
python main.py --exp_name runs/baseline/fr10 --architecture yolophi --epochs 10 --framerate 10
python main.py --exp_name runs/baseline/fr20 --architecture yolophi --epochs 10 --framerate 20

python main.py --exp_name runs/grey/fr2  --architecture yolophi --epochs 10 --framerate 2 --simulator grey
python main.py --exp_name runs/grey/fr10 --architecture yolophi --epochs 10 --framerate 10 --simulator grey
python main.py --exp_name runs/grey/fr20 --architecture yolophi --epochs 10 --framerate 20 --simulator grey

python main.py --exp_name runs/mod/m_mlp2                      --epochs 100 --framerate 2
python main.py --exp_name runs/mod/m_yol5 --architecture yolo5 --epochs 100 --framerate 2
python main.py --exp_name runs/mod/m_yol8 --architecture yolo8 --epochs 100 --framerate 2
python main.py --exp_name runs/mod/m_mlp1 --architecture mlp   --epochs 100 --framerate 2




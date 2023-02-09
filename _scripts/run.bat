

@REM python main.py --exp_name runs/baseline/fr2  --architecture yolophi --epochs 100 --framerate 2 --skip_train
@REM python main.py --exp_name runs/baseline/fr10 --architecture yolophi --epochs 10 --framerate 10
@REM python main.py --exp_name runs/baseline/fr20 --architecture yolophi --epochs 10 --framerate 20

@REM python main.py --exp_name runs/grey/fr2  --architecture yolophi --epochs 10 --framerate 2 --simulator grey
@REM python main.py --exp_name runs/grey/fr10 --architecture yolophi --epochs 10 --framerate 10 --simulator grey
@REM python main.py --exp_name runs/grey/fr20 --architecture yolophi --epochs 10 --framerate 20 --simulator grey

@REM python main.py --exp_name runs/mod/m_mlp2                      --epochs 100 --framerate 2
@REM python main.py --exp_name runs/mod/m_yol5 --architecture yolo5 --epochs 100 --framerate 2
@REM python main.py --exp_name runs/mod/m_yol8 --architecture yolo8 --epochs 100 --framerate 2
@REM python main.py --exp_name runs/mod/m_mlp1 --architecture mlp1  --epochs 100 --framerate 2


python main.py --exp_name comparison/gry_phi  --architecture yolophi    --epochs 160 --framerate 2 --simulator grey
python main.py --exp_name comparison/gry_ml2  --architecture mlp2       --epochs 160 --framerate 2 --simulator grey
python main.py --exp_name comparison/sta_phi  --architecture yolophi    --epochs 160 --framerate 2 --simulator static
python main.py --exp_name comparison/sta_ml2  --architecture mlp2       --epochs 160 --framerate 2 --simulator static

python main.py --exp_name framerates/sta_phi10  --architecture yolophi    --epochs 160 --framerate 10 --simulator static
python main.py --exp_name framerates/sta_phi20  --architecture yolophi    --epochs 160 --framerate 20 --simulator static



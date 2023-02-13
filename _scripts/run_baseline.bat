

python main.py --exp_name comparison/baseline/trg  --architecture yolophi  --epochs 160 --framerate 2 --simulator static --triggering --debug

python main.py --exp_name comparison/baseline/f2s123   --architecture yolophi  --epochs 50 --framerate 2 --simulator static --svs_close 1 --svs_open 2 --svs_hot 3 --pretrained _outputs/comparison/baseline/trg/model.pt  
python main.py --exp_name comparison/baseline/f2s135   --architecture yolophi  --epochs 50 --framerate 2 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/baseline/trg/model.pt
python main.py --exp_name comparison/baseline/f2s324   --architecture yolophi  --epochs 50 --framerate 2 --simulator static --svs_close 3 --svs_open 2 --svs_hot 4 --pretrained _outputs/comparison/baseline/trg/model.pt

python main.py --exp_name comparison/baseline/f05s123  --architecture yolophi  --epochs 50 --framerate .5 --simulator static --svs_close 1 --svs_open 2 --svs_hot 3 --pretrained _outputs/comparison/baseline/trg/model.pt
python main.py --exp_name comparison/baseline/f05s135  --architecture yolophi  --epochs 50 --framerate .5 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/baseline/trg/model.pt
python main.py --exp_name comparison/baseline/f05s324  --architecture yolophi  --epochs 50 --framerate .5 --simulator static --svs_close 3 --svs_open 2 --svs_hot 4 --pretrained _outputs/comparison/baseline/trg/model.pt

python main.py --exp_name comparison/baseline/f10s123  --architecture yolophi  --epochs 50 --framerate 10 --simulator static --svs_close 1 --svs_open 2 --svs_hot 3 --pretrained _outputs/comparison/baseline/trg/model.pt
python main.py --exp_name comparison/baseline/f10s135  --architecture yolophi  --epochs 50 --framerate 10 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/baseline/trg/model.pt
python main.py --exp_name comparison/baseline/f10s324  --architecture yolophi  --epochs 50 --framerate 10 --simulator static --svs_close 3 --svs_open 2 --svs_hot 4 --pretrained _outputs/comparison/baseline/trg/model.pt
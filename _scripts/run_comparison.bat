
@REM triggering task: 
python main.py --exp_name comparison/trg/grey      --architecture yolophi --epochs 160 --framerate 2 --simulator grey    --triggering                                      
python main.py --exp_name comparison/trg/baseline  --architecture yolophi --epochs 160 --framerate 2 --simulator static  --triggering --svs_close 1 --svs_open 3 --svs_hot 5                                       
python main.py --exp_name comparison/trg/opt_sim   --architecture yolophi --epochs 160 --framerate 2 --simulator policy  --triggering --svs_close 1 --svs_open 3 --svs_hot 5 --dont_cache

@REM detection/counting task: 
python main.py --exp_name comparison/grey/f2    --architecture yolophi  --epochs 50 --framerate 2   --simulator grey --pretrained _outputs/comparison/trg/grey/model.pt  

python main.py --exp_name comparison/baseline/f2s135    --architecture yolophi  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/trg/baseline/model.pt  
python main.py --exp_name comparison/baseline/f2s248    --architecture yolophi  --epochs 50 --framerate 2   --simulator static --svs_close 2 --svs_open 4 --svs_hot 8 --pretrained _outputs/comparison/trg/baseline/model.pt
python main.py --exp_name comparison/baseline/f05s135   --architecture yolophi  --epochs 50 --framerate 0.5 --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/trg/baseline/model.pt
python main.py --exp_name comparison/baseline/f20s135   --architecture yolophi  --epochs 50 --framerate 20  --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/trg/baseline/model.pt

python main.py --exp_name comparison/opt_sim/f2s135    --architecture yolophi  --epochs 50  --framerate 2  --simulator policy --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/trg/opt_sim/model.pt  
python main.py --exp_name comparison/opt_sim/f2s248    --architecture yolophi  --epochs 50 --framerate 2   --simulator policy --svs_close 2 --svs_open 4 --svs_hot 8 --pretrained _outputs/comparison/trg/opt_sim/model.pt
python main.py --exp_name comparison/opt_sim/f05s135   --architecture yolophi  --epochs 50 --framerate 0.5 --simulator policy --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/trg/opt_sim/model.pt
python main.py --exp_name comparison/opt_sim/f20s135   --architecture yolophi  --epochs 50 --framerate 20  --simulator static --svs_close 1 --svs_open 3 --svs_hot 5 --pretrained _outputs/comparison/trg/opt_sim/model.pt



@REM baseline: 
python main.py --exp_name comp_synth/base/grey --architecture yolophi --framerate 4   --simulator grey
python main.py --exp_name comp_synth/base/4fr  --architecture yolophi --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/base/1fr  --architecture yolophi --framerate 1   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/base/15fr --architecture yolophi --framerate 15  --simulator static --svs_close 1 --svs_open 3 --svs_hot 5

@REM small models
python main.py --exp_name comp_synth/small/7k  --architecture opt_yolo7  --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/small/77k --architecture opt_yolo77 --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5

@REM other simulators
python main.py --exp_name comp_synth/other/base --architecture yolophi --framerate 4 --simulator mhi        --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/other/base --architecture yolophi --framerate 4 --simulator mhicatgrey --svs_close 1 --svs_open 3 --svs_hot 5

@REM learn-policy
python policy_learn.py --architecture yolophi    --pretrained comp_synth/base/4fr/model.pt  --policy nnpol
python policy_learn.py --architecture opt_yolo7  --pretrained comp_synth/small/7k/model.pt  --policy nnpol7
python policy_learn.py --architecture opt_yolo77 --pretrained comp_synth/small/77k/model.pt --policy nnpol77
@REM use-policy
python main.py --exp_name comp_synth/policy/base --architecture yolophi    --framerate 4   --simulator policy --policy nnpol   --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/policy/7k   --architecture opt_yolo7  --framerate 4   --simulator policy --policy nnpol7  --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/policy/77k  --architecture opt_yolo77 --framerate 4   --simulator policy --policy nnpol77 --svs_close 1 --svs_open 3 --svs_hot 5

@REM quantize
python main.py --exp_name comp_synth/quant/base --quantize 8bit --architecture yolophi    --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/quant/7k   --quantize 8bit --architecture opt_yolo7  --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comp_synth/quant/77k  --quantize 8bit --architecture opt_yolo77 --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5

@REM final POLICY + GREYMHI + 7K
@REM python main.py --exp_name comp_synth/quant/7k  --architecture opt_yolo7  --framerate 4   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5


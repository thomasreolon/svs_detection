
@REM @REM baseline: 
python main.py --exp_name comp_synth/base/grey  --architecture yolophi --framerate 4   --simulator grey   --dataset people 
python main.py --exp_name comp_synth/base/4fr_  --architecture yolophi --framerate 4   --simulator static --dataset people 
python main.py --exp_name comp_synth/base/4fr_sy  --architecture yolophi --framerate 4   --simulator static --dataset synth     --epochs 20 --pretrained comp_synth/base/4fr_/model.pt 
python main.py --exp_name comp_synth/base/4fr_st  --architecture yolophi --framerate 4   --simulator static --dataset streets23 --epochs 20 --pretrained comp_synth/base/4fr_/model.pt 

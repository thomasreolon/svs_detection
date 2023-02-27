
@REM @REM baseline: 
@REM python main.py --exp_name comp_synth/base/grey    --architecture yolophi --framerate 4   --simulator grey   --dataset people 
@REM python main.py --exp_name comp_synth/base/4fr_    --architecture yolophi --framerate 4   --simulator static --dataset people 
python main.py --exp_name comp_synth/base/4fr_sy  --architecture yolophi --framerate 4   --simulator static --dataset MOT     --epochs 20 --pretrained comp_synth/base/4fr_/model.pt 
@REM python main.py --exp_name comp_synth/base/4fr_st  --architecture yolophi --framerate 4   --simulator static --dataset streets23 --epochs 20 --pretrained comp_synth/base/4fr_/model.pt --triggering 

@REM @REM fps: 
@REM python main.py --exp_name comp_synth/base/15fr_    --architecture yolophi --framerate 15   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/base/1fr_    --architecture yolophi --framerate 1   --simulator static --dataset people 

@REM @REM models: 
@REM python main.py --exp_name comp_synth/small/77k  --architecture opt_yolo77 --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/small/7k  --architecture opt_yolo7   --framerate 4   --simulator static --dataset people 
python main.py --exp_name comp_synth/small/s1  --architecture mini       --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/small/s2  --architecture mini2       --framerate 4   --simulator static --dataset people 

@REM @REM policy
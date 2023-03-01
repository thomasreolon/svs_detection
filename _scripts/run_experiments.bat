python main.py --exp_name comp_synth/small/77k --architecture opt_yolo77 --framerate 4   --simulator static --dataset people 
python policy_learn2.py --architecture opt_yolo77 --pretrained comp_synth/small/77k/model.pt
python policy_learn2.py --architecture blob

@REM baseline: 
python main.py --exp_name comp_synth/base/grey    --architecture yolophi --framerate 4   --simulator grey   --dataset people 
python main.py --exp_name comp_synth/base/4fr_    --architecture yolophi --framerate 4   --simulator static --dataset people 
python mainblob.py --exp_name comp_synth/base/blob --architecture blob --framerate 4    --simulator static --dataset people 

@REM fps: 
python main.py --exp_name comp_synth/base/15fr_ --architecture yolophi --framerate 15   --simulator static --dataset people 
python main.py --exp_name comp_synth/base/1fr_  --architecture yolophi --framerate 1   --simulator static --dataset people 

@REM models: 
@REM python main.py --exp_name comp_synth/small/77k --architecture opt_yolo77 --framerate 4   --simulator static --dataset people 
python main.py --exp_name comp_synth/small/7k  --architecture opt_yolo7  --framerate 4   --simulator static --dataset people 
python main.py --exp_name comp_synth/small/s1  --architecture mini       --framerate 4   --simulator static --dataset people 
python main.py --exp_name comp_synth/small/s2  --architecture mini2      --framerate 4   --simulator static --dataset people 

@REM @REM policy NN
@REM python policy_learn2.py --architecture opt_yolo77 --pretrained comp_synth/small/77k/model.pt
@REM python main.py --exp_name comp_synth/policy/77k_lin --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_lin.pt  --dataset people
@REM python main.py --exp_name comp_synth/policy/77k_nn  --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_nn.pt   --dataset people
@REM python main.py --exp_name comp_synth/policy/77k_fix --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_fix.pt  --dataset people

@REM policy default boy
@REM python policy_learn2.py --architecture blob
@REM python mainblob.py --exp_name comp_synth/policy/blob_lin --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_lin.pt  --dataset people
@REM python mainblob.py --exp_name comp_synth/policy/blob_nn  --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_nn.pt   --dataset people
@REM python mainblob.py --exp_name comp_synth/policy/blob_fix --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_fix.pt  --dataset people


@REM python main.py --exp_name comp_synth/base/4fr_st  --architecture yolophi --framerate 4   --simulator static --dataset streets23 --epochs 20 --pretrained comp_synth/base/4fr_/model.pt --triggering 
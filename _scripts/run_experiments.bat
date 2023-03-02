@REM baseline: 
@REM python main.py --exp_name comp_synth/base/grey    --architecture yolophi --framerate 4   --simulator grey   --dataset people 
@REM python main.py --exp_name comp_synth/base/4fr_    --architecture yolophi --framerate 4   --simulator static --dataset people 
@REM python mainblob.py --exp_name comp_synth/base/blob --architecture blob --framerate 4    --simulator static --dataset people 

@REM @REM fps: 
@REM python main.py --exp_name comp_synth/base/15fr_ --architecture yolophi --framerate 15   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/base/1fr_  --architecture yolophi --framerate 1   --simulator static --dataset people 

@REM models
@REM python main.py --exp_name comp_synth/small/77k --architecture opt_yolo77 --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/small/7k  --architecture opt_yolo7  --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/small/s1  --architecture mini       --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comp_synth/small/s2  --architecture mini2      --framerate 4   --simulator static --dataset people 

@REM @REM policy NN
python policy_learn2.py --architecture opt_yolo77 --pretrained comp_synth/small/77k/model.pt
python main.py --exp_name comp_synth/policy/77k_f1 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_f1.pt  --dataset people
python main.py --exp_name comp_synth/policy/77k_f2 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people
python main.py --exp_name comp_synth/policy/77k_f3 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_f3.pt  --dataset people
python main.py --exp_name comp_synth/policy/77k_n1 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_n1.pt  --dataset people
python main.py --exp_name comp_synth/policy/77k_n2 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_n2.pt  --dataset people
python main.py --exp_name comp_synth/policy/77k_sv --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_sv.pt  --dataset people

@REM policy default boy
@REM python policy_learn2.py --architecture blob
python mainblob.py --exp_name comp_synth/policy/blob_f1 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_f1.pt  --dataset people
python mainblob.py --exp_name comp_synth/policy/blob_f2 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_f2.pt  --dataset people
python mainblob.py --exp_name comp_synth/policy/blob_f3 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_f3.pt  --dataset people
python mainblob.py --exp_name comp_synth/policy/blob_n1 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_n1.pt  --dataset people
python mainblob.py --exp_name comp_synth/policy/blob_n2 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_n2.pt  --dataset people
python mainblob.py --exp_name comp_synth/policy/blob_sv --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_sv.pt  --dataset people


@REM python main.py --exp_name comp_synth/base/4fr_st  --architecture yolophi --framerate 4   --simulator static --dataset streets23 --epochs 20 --pretrained comp_synth/base/4fr_/model.pt --triggering 

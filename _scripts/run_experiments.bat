@REM @REM baseline: 
@REM python main.py      --exp_name comparison/base/grey --architecture yolophi --framerate 4  --simulator grey   --dataset people 
@REM python main.py      --exp_name comparison/base/4fr_ --architecture yolophi --framerate 4  --simulator static --dataset people 
@REM python mainblob.py  --exp_name comparison/base/blob --architecture blob    --framerate 4  --simulator static --dataset people 
@REM python main.py      --exp_name comparison/base/mhi --architecture yolophi --framerate 4  --simulator mhi --dataset people 
@REM python main.py      --exp_name comparison/base/grmhi --architecture yolophi --framerate 4  --simulator mhicatgrey --dataset people 

@REM @REM better init: 
@REM python main.py      --exp_name comparison/stat/1232 --architecture yolophi --framerate 4  --simulator mhicatgrey --dataset people --svs_close 1 --svs_open 2 --svs_hot 3 --svs_ker 2
@REM python main.py      --exp_name comparison/stat/1355 --architecture yolophi --framerate 4  --simulator mhicatgrey --dataset people --svs_close 1 --svs_open 3 --svs_hot 5 --svs_ker 5

@REM @REM quantize: 
@REM python main.py --exp_name comparison/quant/8bit --architecture opt_yolo7 --framerate 4 --quantize 8bit
@REM python main.py --exp_name comparison/quant/1bit --architecture opt_yolo7 --framerate 4 --quantize 1bit
@REM python main.py --exp_name comparison/quant/2bit --architecture opt_yolo7 --framerate 4 --quantize 2bit
@REM python main.py --exp_name comparison/quant/4bit --architecture opt_yolo7 --framerate 4 --quantize 4bit
@REM python main.py --exp_name comparison/quant/by7  --architecture opt_yolo7 --framerate 4 --quantize binary --epochs 20
@REM python main.py --exp_name comparison/quant/bys  --architecture mini      --framerate 4 --quantize binary --epochs 20
@REM python main.py --exp_name comparison/quant/bym  --architecture mlp2      --framerate 4 --quantize binary --epochs 20
@REM python main.py --exp_name comparison/quant/by8  --architecture yolo8     --framerate 4 --quantize binary --epochs 20


@REM @REM fps: 
@REM python main.py --exp_name comparison/base/15fr_ --architecture yolophi --framerate 15   --simulator static --dataset people 
@REM python main.py --exp_name comparison/base/1fr_  --architecture yolophi --framerate 1   --simulator static --dataset people 

@REM @REM models
@REM python main.py --exp_name comparison/small/77k --architecture opt_yolo77 --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comparison/small/7k  --architecture opt_yolo7  --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comparison/small/s1  --architecture mini       --framerate 4   --simulator static --dataset people 
@REM python main.py --exp_name comparison/small/s2  --architecture mini2      --framerate 4   --simulator static --dataset people
@REM python main.py --exp_name comparison/small/s3  --architecture mini3      --framerate 4   --simulator static --dataset people
@REM python main.py --exp_name comparison/small/mlp --architecture mlp2       --framerate 4   --simulator static --dataset people
@REM python main.py --exp_name comparison/small/v8  --architecture yolo8      --framerate 4   --simulator static --dataset people

@REM @REM policy
@REM python policy_learn2.py --architecture blob --dataset MOT
@REM python mainblob.py --exp_name comparison/policy/blob_f1 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_f1.pt  --dataset people
@REM python mainblob.py --exp_name comparison/policy/blob_f2 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_f2.pt  --dataset people
@REM python mainblob.py --exp_name comparison/policy/blob_n1 --dont_cache --architecture blob --framerate 4   --simulator policy --policy plogs/blob_n1.pt  --dataset people

@REM @REM policy NN
@REM python policy_learn2.py --architecture opt_yolo77 --pretrained comparison/small/77k/model.pt --dataset people
@REM python main.py --exp_name comparison/policy/77k_f1 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_f1.pt  --dataset people
@REM python main.py --exp_name comparison/policy/77k_f2 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people
@REM python main.py --exp_name comparison/policy/77k_n1 --dont_cache --architecture opt_yolo77 --framerate 4   --simulator policy --policy plogs/opt_yolo77_n1.pt  --dataset people


@REM @REM quantize + policy
@REM python main.py --exp_name comparison/final/77k --dont_cache --architecture opt_yolo77 --quantize 8bit --framerate 4  --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people 
@REM python main.py --exp_name comparison/final/7k  --dont_cache --architecture opt_yolo7  --quantize 8bit --framerate 4  --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people 
@REM python main.py --exp_name comparison/final/s1  --dont_cache --architecture mini       --quantize 8bit --framerate 4  --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people 
@REM python main.py --exp_name comparison/final/s2  --dont_cache --architecture mini2      --quantize 8bit --framerate 4  --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people
@REM python main.py --exp_name comparison/final/mlp --dont_cache --architecture mlp2       --quantize binary --framerate 4  --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people

@REM @REM policy NN fr 15
@REM python policy_learn2.py --architecture opt_yolo77 --pretrained comparison/small/77k/model.pt --dataset people --framerate 15 --reset
@REM python main.py --exp_name comparison/policy_15fps2/77k_f1 --dont_cache --architecture opt_yolo77 --framerate 15   --simulator policy --policy plogs/opt_yolo77_f1.pt  --dataset people
@REM python main.py --exp_name comparison/policy_15fps2/77k_f2 --dont_cache --architecture opt_yolo77 --framerate 15   --simulator policy --policy plogs/opt_yolo77_f2.pt  --dataset people
@REM python main.py --exp_name comparison/policy_15fps2/77k_n2 --dont_cache --architecture opt_yolo77 --framerate 15   --simulator policy --policy plogs/opt_yolo77_n2.pt  --dataset people


@REM python policy_learn.py --architecture blob --dataset MOT --n_iter 10
python mainblob.py --exp_name comparison/policy_new/77k --dont_cache --architecture opt_yolo77 --pretrained plogs2/model.pt --framerate 4  --simulator policy --policy plogs2/opt_yolo77_fix.pt  --dataset people

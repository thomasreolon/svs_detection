@REM baseline: 
python main.py      --exp_name comparison/base/grey --architecture phiyolo --framerate 4  --simulator grey   --dataset people 
python main.py      --exp_name comparison/base/4fr_ --architecture phiyolo --framerate 4  --simulator static --dataset people 
python mainblob.py  --exp_name comparison/base/blob --architecture blob    --framerate 4  --simulator static --dataset people 
python main.py      --exp_name comparison/base/mhi --architecture phiyolo --framerate 4  --simulator mhi     --dataset people 
python main.py      --exp_name comparison/base/grmhi --architecture phiyolo --framerate 4  --simulator mhicatgrey --dataset people 

@REM baseline-triggering
python main.py      --exp_name comparison/base/4fr_tr --architecture phiyolo --framerate 4  --simulator static --dataset streets23 --triggering --pretrained  comparison/base/4fr_/model.pt --epochs 20

@REM fps: 
python main.py --exp_name comparison/base/15fr_ --architecture phiyolo --framerate 15   --simulator static --dataset people 
python main.py --exp_name comparison/base/1fr_  --architecture phiyolo --framerate 1   --simulator static --dataset people 

@REM models
python main.py --exp_name comparison/small/77k --architecture phiyolo77K --framerate 4   --simulator static --dataset people 
python main.py --exp_name comparison/small/7k  --architecture phiyolo7K  --framerate 4   --simulator static --dataset people 
python main.py --exp_name comparison/small/s1  --architecture phisimple1 --framerate 4   --simulator static --dataset people 
python main.py --exp_name comparison/small/s2  --architecture phisimple2 --framerate 4   --simulator static --dataset people
python main.py --exp_name comparison/small/s3  --architecture phisimple3 --framerate 4   --simulator static --dataset people
python main.py --exp_name comparison/small/mlp --architecture mlp2       --framerate 4   --simulator static --dataset people
python main.py --exp_name comparison/small/v8  --architecture yolo8      --framerate 4   --simulator static --dataset people

@REM policy
python policy_learn.py --architecture phiyolo7K --n_iter 15  --pretrained comparison/small/7k/model.pt  --dataset MOT --framerate 4
python main.py --exp_name comparison/policy/f1   --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fix1.pt  --dataset people 
python main.py --exp_name comparison/policy/f3   --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fix3.pt  --dataset people 
python main.py --exp_name comparison/policy/f8   --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fix8.pt  --dataset people 
python main.py --exp_name comparison/policy/fbl  --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fixwb.pt --dataset people
python main.py --exp_name comparison/policy/nnxs --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnxs.pt  --dataset people 
python main.py --exp_name comparison/policy/nns  --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nns.pt  --dataset people 
python main.py --exp_name comparison/policy/nnm  --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnm.pt  --dataset people 
python main.py --exp_name comparison/policy/nnl  --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnl.pt  --dataset people 
python main.py --exp_name comparison/policy/nnxl --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnxl.pt  --dataset people 
python main.py --exp_name comparison/policy/nnex --dont_cache --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnex.pt  --dataset people 

@REM policy + quantization
python policy_learn.py --architecture phiyolo7K --n_iter 15  --pretrained comparison/small/7k/model.pt --reset --dataset people --framerate 4
python main.py --exp_name comparison/final/f8_     --dont_cache --quantize 8bit --architecture phiyolo    --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fix8.pt  --dataset people 
python main.py --exp_name comparison/final/nnex_   --dont_cache --quantize 8bit --architecture phiyolo    --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnex.pt  --dataset people 
python main.py --exp_name comparison/final/f8_77   --dont_cache --quantize 8bit --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fix8.pt  --dataset people 
python main.py --exp_name comparison/final/nnex_77 --dont_cache --quantize 8bit --architecture phiyolo77K --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnex.pt  --dataset people 
python main.py --exp_name comparison/final/f8_7    --dont_cache --quantize 8bit --architecture phiyolo7K  --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_fix8.pt  --dataset people 
python main.py --exp_name comparison/final/nnex_7  --dont_cache --quantize 8bit --architecture phiyolo7K  --framerate 4  --simulator policy --policy plogs3_fr4/phiyolo7_nnex.pt  --dataset people 





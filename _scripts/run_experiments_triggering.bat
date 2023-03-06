@REM triggering baseline: 
python main.py      --exp_name comparison/triggering/grey  --architecture yolophi --framerate 4  --epochs 20 --simulator grey   --dataset streets23 --triggering --pretrained comparison/base/grey/model.pt
python main.py      --exp_name comparison/triggering/4fr_  --architecture yolophi --framerate 4  --epochs 20 --simulator static --dataset streets23 --triggering --pretrained comparison/base/4fr_/model.pt
python mainblob.py  --exp_name comparison/triggering/blob  --architecture blob    --framerate 4  --epochs 20 --simulator static --dataset streets23 --triggering
python main.py      --exp_name comparison/triggering/15fr_ --architecture yolophi --framerate 15 --epochs 20 --simulator static --dataset streets23 --triggering --pretrained comparison/base/15fr_/model.pt 
python main.py      --exp_name comparison/triggering/1fr_  --architecture yolophi --framerate 1  --epochs 20 --simulator static --dataset streets23 --triggering --pretrained comparison/base/1fr_/model.pt 

@REM triggering [quantize + policy]
python main.py --exp_name comparison/triggering_f/77k --dont_cache --architecture opt_yolo77 --pretrained comparison/final/77k/model.pt --quantize 8bit    --simulator policy --policy plogs_old/opt_yolo77_f2.pt  --dataset streets23 --epochs 20 --triggering 
python main.py --exp_name comparison/triggering_f/7k  --dont_cache --architecture opt_yolo7  --pretrained comparison/final/7k/model.pt  --quantize 8bit    --simulator policy --policy plogs_old/opt_yolo77_f2.pt  --dataset streets23 --epochs 20 --triggering 
python main.py --exp_name comparison/triggering_f/s1  --dont_cache --architecture mini       --pretrained comparison/final/s1/model.pt  --quantize 8bit    --simulator policy --policy plogs_old/opt_yolo77_f2.pt  --dataset streets23 --epochs 20 --triggering 
python main.py --exp_name comparison/triggering_f/s2  --dont_cache --architecture mini2      --pretrained comparison/final/s2/model.pt  --quantize 8bit    --simulator policy --policy plogs_old/opt_yolo77_f2.pt  --dataset streets23 --epochs 20 --triggering
python main.py --exp_name comparison/triggering_f/mlp --dont_cache --architecture mlp2       --pretrained comparison/final/mlp/model.pt --quantize binary  --simulator policy --policy plogs_old/opt_yolo77_f2.pt  --dataset streets23 --epochs 20 --triggering

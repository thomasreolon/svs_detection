
@REM policy
python policy_learn.py --architecture yolophi    --pretrained comp_synth/base/4fr/model.pt  --policy nnpol.pt
python main.py --exp_name comp_synth/policy/base --dont_cache --architecture yolophi    --framerate 4   --simulator policy --policy E:/dataset/_outputs/nnpol.pt   --svs_close 1 --svs_open 3 --svs_hot 5

@REM policy
python policy_learn.py --architecture opt_yolo7    --pretrained comp_synth/small/7k/model.pt  --policy nnpol7.pt
python main.py --exp_name comp_synth/policy/7k  --dont_cache   --architecture opt_yolo7    --framerate 4   --simulator policy --policy E:/dataset/_outputs/nnpol7.pt   --svs_close 1 --svs_open 3 --svs_hot 5

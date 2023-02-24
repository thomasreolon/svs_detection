
@REM policy
python policy_learn.py --architecture yolophi    --pretrained comp_synth/base/4fr/model.pt  --policy nnpol.pt
python main.py --exp_name comp_synth/policy/base --architecture yolophi    --framerate 4   --simulator policy --policy E:/dataset/_outputs/nnpol.pt   --svs_close 1 --svs_open 3 --svs_hot 5

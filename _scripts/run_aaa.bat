@REM learn-policy
python policy_learn2.py --architecture opt_yolo77 --pretrained C:/Users/Tom/Desktop/svs_detection/_outputs/comparison/model/phi1_77/model.pt
python policy_learn2.py --architecture yolophi    --pretrained C:/Users/Tom/Desktop/svs_detection/_outputs/comparison/baseline/f2s135/model.pt
python policy_learn2.py --architecture opt_yolo7  --pretrained C:/Users/Tom/Desktop/svs_detection/_outputs/comparison/opt_sim7/f2s135/model.pt


@REM quantization experiment: 
python main.py --exp_name comparison/quant/8bit_104 --quantize 8bit --architecture yolophi  --epochs 160 --framerate 2    --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/quant/8bit_7 --quantize 8bit  --architecture opt_yolo7  --epochs 160 --framerate 2  --simulator static --svs_close 1 --svs_open 3 --svs_hot 5

python main.py --exp_name comparison/quant/binary_77 --quantize binary --architecture opt_yolo77  --epochs 160 --framerate 2  --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/quant/binary_104 --quantize binary --architecture yolophi  --epochs 160 --framerate 2    --simulator static --svs_close 1 --svs_open 3 --svs_hot 5

python main.py --exp_name comparison/quant/policy8bit_7 --quantize 8bit  --architecture opt_yolo7  --epochs 160 --framerate 2  --simulator policy --policy policy_7.pt --svs_close 1 --svs_open 3 --svs_hot 5

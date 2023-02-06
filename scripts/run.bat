

python main.py --exp_name f1c1 --batch_size 128 --epochs 100 --framerate 1
python main.py --exp_name f16c1 --batch_size 128 --epochs 100 --framerate 16

python main.py --svs_close 2 --svs_open 3 --svs_hot 10 --exp_name f1c2 --batch_size 128 --epochs 100 --framerate 1
python main.py --svs_close 2 --svs_open 3 --svs_hot 10  --exp_name f16c2 --batch_size 128 --epochs 100 --framerate 16
python main.py --svs_close 2 --svs_open 3 --svs_hot 10  --exp_name f4c2 --batch_size 128 --epochs 100

python main.py --svs_close 4 --svs_open 1 --svs_hot 5  --exp_name f1c4 --batch_size 128 --epochs 100 --framerate 1
python main.py --svs_close 4 --svs_open 1 --svs_hot 5  --exp_name f16c4 --batch_size 128 --epochs 100 --framerate 16
python main.py --svs_close 4 --svs_open 1 --svs_hot 5  --exp_name f4c4 --batch_size 128 --epochs 100

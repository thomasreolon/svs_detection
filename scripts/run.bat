

@REM python main.py --exp_name runs/grey/fr1 --batch_size 256 --epochs 200 --simulator grey --framerate 1
@REM python main.py --exp_name runs/grey/fr4 --batch_size 256 --epochs 100 --simulator grey --framerate 4
@REM python main.py --exp_name runs/grey/fr16 --batch_size 256 --epochs 40 --simulator grey --framerate 16
python main.py --exp_name runs/tmp/fr1 --batch_size 256 --epochs 5 --framerate 1
python main.py --exp_name runs/tmp/fr4 --batch_size 256 --epochs 5 --framerate 4
python main.py --exp_name runs/tmp/fr16 --batch_size 256 --epochs 5 --framerate 16

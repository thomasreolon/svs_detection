
@REM @REM triggering task: 
python main.py --exp_name comparison/model/y5_127 --architecture opt_y5  --epochs 160 --framerate 2   --simulator static  --triggering --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/phi1_77 --architecture opt_phi1  --epochs 160 --framerate 2   --simulator static  --triggering --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/phi2_102 --architecture opt_phi2  --epochs 160 --framerate 2   --simulator static  --triggering --svs_close 1 --svs_open 3 --svs_hot 5


@REM other models for plotting 
python main.py --exp_name comparison/model/stats/0 --architecture 0  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/1 --architecture 1  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/2 --architecture 2  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/3 --architecture 3  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/4 --architecture 4  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/5 --architecture 5  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/6 --architecture 6  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/7 --architecture 7  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/8 --architecture 8  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/9 --architecture 9  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/10 --architecture 10  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/11 --architecture 11  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/12 --architecture 12  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5

python main.py --exp_name comparison/model/stats/y5 --architecture opt_y5  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/phi1 --architecture opt_phi1  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5
python main.py --exp_name comparison/model/stats/phi2 --architecture opt_phi2  --epochs 50 --framerate 2   --simulator static --svs_close 1 --svs_open 3 --svs_hot 5



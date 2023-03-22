@REM @REM baseline: 
@REM python main.py --exp_name carscmp/base/grey --architecture phiyolo --framerate 15  --simulator grey   --dataset cars 
@REM python main.py --exp_name carscmp/base/15fr_ --architecture phiyolo --framerate 15   --simulator static --dataset cars 

@REM @REM models
@REM python main.py --exp_name carscmp/small/77k --architecture phiyolo77K --framerate 15   --simulator static --dataset cars 
@REM python main.py --exp_name carscmp/small/7k  --architecture phiyolo7K  --framerate 15   --simulator static --dataset cars 

@REM @REM policy
@REM python policy_learn.py --architecture phiyolo7K --n_iter 30  --pretrained carscmp/small/7k/model.pt  --dataset cars --framerate 15 --reset
@REM python main.py --exp_name carscmp/policy/f8   --dont_cache --architecture phiyolo77K --framerate 15  --simulator policy --policy plogs3_fr15/phiyolo7K_fix8.pt  --dataset cars 
python main.py --exp_name carscmp/policy/nnex --dont_cache --architecture phiyolo77K --framerate 15  --simulator policy --policy plogs3_fr15/phiyolo7K_nnex.pt  --dataset cars 

@REM policy + quantization
python main.py --exp_name carscmp/final/f8_77   --dont_cache --quantize 8bit --architecture phiyolo77K --framerate 15  --simulator policy --policy plogs3_fr15/phiyolo7K_fix8.pt  --dataset cars 
python main.py --exp_name carscmp/final/nnex_77 --dont_cache --quantize 8bit --architecture phiyolo77K --framerate 15  --simulator policy --policy plogs3_fr15/phiyolo7K_nnex.pt  --dataset cars 

python main.py --exp_name carscmp/final/f8_7   --dont_cache --quantize 8bit --architecture phiyolo7K --framerate 15  --simulator policy --policy plogs3_fr15/phiyolo7K_fix8.pt  --dataset cars 
python main.py --exp_name carscmp/final/nnex_7 --dont_cache --quantize 8bit --architecture phiyolo7K --framerate 15  --simulator policy --policy plogs3_fr15/phiyolo7K_nnex.pt  --dataset cars 

python main.py --exp_name carscmp/final/base77 --quantize 8bit --architecture phiyolo77K --framerate 15  --dataset cars 
python main.py --exp_name carscmp/final/grey77 --quantize 8bit --architecture phiyolo77K --framerate 15  --simulator grey  --dataset cars 




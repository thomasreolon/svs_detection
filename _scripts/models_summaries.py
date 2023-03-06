import sys, pathlib ; sys.path.append(pathlib.Path(__file__).parent.resolve().__str__() + '/..') # allows import from home folder
from torchinfo import summary
import torch
import gc
import threading
from time import time

from main import build_model


max_ = [None]
def thread_function(start):
    max_[0] = 0
    while time() -start < 3:
        max_[0] = max(max_[0], torch.cuda.memory_allocated(0))



models = ['mlp2', 'yolo5', 'yolo8', 'yolophi', 'mini', 'mini2', 'mini3', 'opt_yolo7', 'opt_yolo77']
for m_name in models:
    model = build_model(m_name)
    model.eval()
    
    # use torch info to compute MACC
    print('------------->',m_name)
    print(summary(model, input_size=(1, 1, 128, 160)))
    
    # NOTE: try not to use programs that run on the GPU while running this script
    with torch.no_grad():
        # move to cuda
        model = model.cuda()

        MB_memory = []
        for _ in range(10):
            gc.collect() ; torch.cuda.empty_cache()

            # check GPU usage
            x = threading.Thread(target=thread_function, args=(time(),))
            base = torch.cuda.memory_allocated(0)
            x.start()
            model(torch.ones(1, 1, 128, 160).cuda())
            x.join()
            
            # max memory used
            MB_memory.append((max_[0] - base)/1e6)
        
        print(m_name, '(MB): Memory Inference: ', sorted(MB_memory))


    input()
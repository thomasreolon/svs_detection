import pickle
import os

from configs.defaults import get_args_parser
from datasets.mot_dataset_svs import MOTDataset

args = get_args_parser().parse_args()

try:
    with open('outputs/dataset.pk', 'rb') as fin:
        backup_data = pickle.load(fin)
        assert isinstance(backup_data, dict)
        print('LOADED N KEYS:', len(backup_data))
except:
    print('NEW')
    backup_data = {}

for col in [{'brightness':0.5, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.5}, {'brightness':0.1, 'contrast':0.5, 'saturation':0.5, 'sharpness':0.5, 'hue':0.5, 'gamma':0.2}]:
    for fr in [1,3,10,20]:
        for cl,op,ho in [(1,3,5), (4,1,5), (2,3,10)]:
            for vid in os.listdir(args.mot_path+'/train'):
                key = f'{cl}-{op}-{ho}-{vid}-{fr}-False-True-{list(col.values())}'
                if key in backup_data:continue
                print('working on', key)
                try:
                    ds = MOTDataset(args.mot_path, cl,op,ho, vid, fr, False, True, col)

                    backup_data[key] = ds.data
                except KeyboardInterrupt:
                    print(backup_data.keys())
                    with open('outputs/dataset.pk', 'wb') as fout:
                        pickle.dump(backup_data, fout)
                    exit(0)
                except:
                    print('FAILES', vid)

            with open('outputs/dataset.pk', 'wb') as fout:
                pickle.dump(backup_data, fout)

print(backup_data.keys())



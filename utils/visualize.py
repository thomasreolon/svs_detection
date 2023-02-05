from PIL import Image
from datasets.transforms import augment_color, get_mask

def visualize_prediction():
    pass




def load_original(args, info):
    info = info.split(';')
    vid = info[0]
    frame = info[1]
    folder = 'train' if eval(info[2]) else 'test'
    aug_color = eval(info[3])
    frame_path = f'{args.mot_path}/{folder}/{vid}/img1/{frame}.jpg'

    # mask
    tmp = Image.open(f'{args.mot_path}/{folder}/{vid}/img1/000001.jpg')
    mask, sigma = get_mask(tmp, aug_color['noise'])
    aug_color.update({'gnoise_mask':mask, 'gnoise_sigma':sigma})

    # augment color
    img = Image.open(frame_path)
    img = augment_color(img, **aug_color)

    return img

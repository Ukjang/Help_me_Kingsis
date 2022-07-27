from model.mmaction2 import mmaction
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import argparse
import os.path as osp
import decord
import webcolors
from mmcv import Config, DictAction
from model.mmaction2.mmaction.apis import inference_recognizer, init_recognizer

config_path = './model/mmactions2/configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py'
checkpoint_path = './model/mmactions2/checkpoints/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'
label_path = './model/mmactions2/tools/data/kinetics/label_map_k400.txt' 
def TimeSformer(video, config=config_path, checkpoint=checkpoint_path, label=label_path):
    
    device = torch.device('cuda:0')

    cfg = Config.fromfile(config)
    cfg.merge_from_dict({})

    # build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, checkpoint, device=device)

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # test a single video or rawframes of a single video
    if output_layer_names:
        results, returned_feature = inference_recognizer(
            model, video, outputs=output_layer_names)
    else:
        results = inference_recognizer(model, video)

    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    results = [(labels[k[0]], k[1]) for k in results]
    
    # 수정한 부분
    csv_path = str(video).split('.')[-2] + '.csv'
    df = pd.DataFrame(results, columns=['label', 'per'])
    df.index = df.label

    del df['label']
    df_results = df.index[df.per.argmax()]
#    print('The top-5 labels with corresponding scores are:')
#    for result in results:
#        print(f'{result[0]}: ', result[1])
    return df_results
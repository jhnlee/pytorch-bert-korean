from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import optimization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import Datasets
from model import BertForEmotionClassification
from tqdm import tqdm
from sklearn.metrics import classification_report
import pickle
import config
import utils
import argparse
import os
import torch
import torch.nn as nn

#logger = get_logger('Binary Classification')
# logger.setLevel(logging.INFO)


def train(args):
    # Set device
    if args.device == 'cuda':
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    # Load pretrained model and model configuration
    pretrained_path = os.path.join('./pretrained_model/', args.pretrained_type)
    pretrained = torch.load(pretrained_path + 'pytorch_model.bin')
    if args.pretrained_type == 'skt':
        # skt model의 파라미터 이름이 달라 수정
        new_keys_ = ['bert.' + k for k in pretrained.keys()]
        old_values_ = pretrained.values()
        pretrained = {k: v for k, v in zip(new_keys_, old_values_)}
    bert_config = BertConfig(pretrained_path + 'bert_config.json')
    model = BertForEmotionClassification(bert_config)
    model.load_state_dict(pretrained, strict=False)

    label_list = ['공포', '놀람', '분노', '슬픔', '중립', '행복',
                  '혐오'] if args.num_label == 'multi' else ['긍정', '부정']
    tr_set = Datasets(file_path=args.train_data_path, label_list=label_list,)


parser = argparse.ArgumentParser()
# Pretrained model Parameters
parser.add_argument("--pretrained_type", default=None, type=str, required=True,
                    help="type of pretrained model (skt, etri)")

# Data Parameters
parser.add_argument("--num_label", default=None, type=int, required=True,
                    help="Number of labels in datastes")

parser.add_argument("--device", default=None, type=str, required=True,
                    help="Whether to use cuda or cpu")
args = parser.parse_args()

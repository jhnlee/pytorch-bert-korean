from flask import Flask, jsonify, request, render_template
from model import BertForEmotionClassification
from datasets import get_pretrained_model, Datasets
from pytorch_transformers.modeling_bert import BertConfig
import numpy as np
import torch

pretrained_model_path = './best_model/best_model.bin'
config_path = './best_model/bert_config.json'

pretrained = torch.load(pretrained_model_path)
bert_config = BertConfig(config_path)
bert_config.num_labels = 7

model = BertForEmotionClassification(bert_config)
model.load_state_dict(pretrained, strict=False)
model.eval()
softmax = torch.nn.Softmax(dim=1)

tokenizer, vocab = get_pretrained_model('etri')

label_list=np.array(['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오'])

def get_prediction(sentence):
    sentence = Datasets.normalize_string(sentence)
    sentence = tokenizer.tokenize(sentence)
    sentence = tokenizer.convert_tokens_to_ids(sentence)    
    sentence = [vocab['[CLS]']] + sentence + [vocab['[SEP]']]
    
    output = model(torch.tensor(sentence).unsqueeze(0))
    output_softmax = softmax(output)[0]
    max_out = label_list[output_softmax.argmax()]
    argidx = output_softmax.argsort(descending=True)
    result = {label_list[i]: round(output_softmax[i].item(), 3) for i in range(len(label_list))}
    sorted_result = {label_list[i]: round(output_softmax[i].item(), 3) for i in argidx}
    return max_out, result, sorted_result

app = Flask(__name__)
app._static_folder = './static'

@app.route('/api', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.json['sentence']
        max_out, result, sorted_result = get_prediction(sentence)
        return jsonify({'input': sentence,
                        'emotion': max_out,
                        'output': result})

@app.route('/')
def index():
    if request.args:
        sentence = request.args['sentence']
        max_out, result, sorted_result = get_prediction(sentence)
        return render_template('index.html', sentence=sentence, result=result, sorted_result=sorted_result) 
    else:
        return render_template('index.html', result={}, sorted_result={})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    

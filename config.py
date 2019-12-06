class classification_config:
    pretrained_model_name = 'etri'
    mode = 'classification'  # 'MLM', 'classification', 'both'
    # in-domain w/o classification
    #use_further_pretrained = '/In-domain_pretraining/without_classification/param_etri_epoch 3, lr 3e-5, batch 64, warmup 10000, accu 1, len 64/' \
    #                         'best_modeletri_pretrained_epoch3, lr3e-05, batch64, warmup10000, accu1.bin'
    # in-domain w/ classification
    use_further_pretrained = '/In-domain_pretraining/with_classification/param_etri_epoch 3, lr 2e-05, batch 64, warmup 1000, accu 2, len 64/' \
                             'best_model_etri_pretrained_epoch3, lr2e-05, batch64, warmup1000, accu2.bin'
    # in-task
    #use_further_pretrained = '/In-task_pretraining/param_etri_epoch 12, lr 3e-5, batch 64, warmup 0, accu 1, len 64/' \
    #                         'best_modeletri_pretrained_epoch12, lr3e-05, batch64, warmup0, accu1.bin'
    # finetuning
    #use_further_pretrained = None
    layerwise_decay = None

    max_length = 64
    lr = 2e-5  # 2e-5
    max_grad_norm = 1  # 10
    num_warmup_steps = 20  # 100
    summary_step = 1000

    batch_size = 64
    epoch = 3
    gradient_accumulation_steps = 1  # for lager batch size


    trainpath = './data/korean_single_train.csv'
    devpath = './data/korean_single_dev.csv'
    model_dir = './model/finetuning/masked_pretrained'
    num_label = 7

class pretraining_config:
    pretrained_model_name = 'etri'
    mode = 'MLM'  # 'MLM', 'classification', 'MLM_with_classification'
    use_further_pretrained = None
    layerwise_decay = None

    max_length = 64
    lr = 3e-5
    max_grad_norm = 1
    num_warmup_steps = 10000
    summary_step = 10000

    batch_size = 64
    epoch = 20
    gradient_accumulation_steps = 1  # for lager batch size

    trainpath = './data/korean_crawled_train.csv'
    devpath = './data/korean_crawled_dev.csv'
    model_dir = './model/In-domain_pretraining/without_classification'
    num_label = 2

class MLM_with_classification_config:
    pretrained_model_name = 'etri'
    mode = 'MLM_with_classification'  # 'MLM', 'classification', 'MLM_with_classification'
    use_further_pretrained = None
    layerwise_decay = None

    max_length = 64
    lr = 2e-5
    max_grad_norm = 1
    num_warmup_steps = 1000
    summary_step = 1000

    batch_size = 64
    epoch = 3
    gradient_accumulation_steps = 2  # for lager batch size

    trainpath = './data/korean_crawled_train.csv'
    devpath = './data/korean_crawled_dev.csv'
    model_dir = './model/In-domain_pretraining/with_classification'
    num_label = 2


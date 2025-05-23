from llama_zp import *


if __name__ == "__main__":
    data_name = 'pdtb3'
    data_path = '/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_distilled/pdtb3.llama3.subtext_base.csv'
    instruction = '''
Argument 1:
{arg1}

Argument 2:
{arg2}

{subtext}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison.Concession
B. Comparison.Contrast
C. Comparison.Similarity
D. Contingency.Cause
E. Contingency.Condition
F. Contingency.Purpose
G. Expansion.Conjunction
H. Expansion.Equivalence
I. Expansion.Instantiation
J. Expansion.Level-of-detail
K. Expansion.Manner
L. Expansion.Substitution
M. Temporal.Asynchronous
N. Temporal.Synchronous

'''.strip()
    
    data_name = 'pdtb2'
    data_path = '/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_distilled/pdtb2.llama3.subtext_base.csv'
    instruction = '''
Argument 1:
{arg1}

Argument 2:
{arg2}

{subtext}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison.Concession
B. Comparison.Contrast
C. Contingency.Cause
D. Contingency.Pragmatic cause
E. Expansion.Alternative
F. Expansion.Conjunction
G. Expansion.Instantiation
H. Expansion.List
I. Expansion.Restatement
J. Temporal.Asynchronous
K. Temporal.Synchrony

'''.strip()

    dfs = IDRRDataFrames(
        data_name=data_name,
        data_level='second',
        data_relation='Implicit',
        data_path=data_path,
    )
    trainset_config = IDRRDatasetConfig(
        data_split='train',
        prompt={
            "instruction": instruction,
            "input": '',
            "output": '{label11}',
            "system": "",
            "history": [],
        },
        desc='subtext_distilled',
        **dfs.arg_dic,
    )

    # model_path = path('/public/home/hongy/pretrained_models/Llama-3-8B-Instruct').resolve()
    model_path = '/public/home/hongy/pretrained_models/Llama-3.2-1B-Instruct'
    # model_path = '/public/home/hongy/pretrained_models/Meta-Llama-3-8B-Instruct'
    model_path = path(model_path).resolve()
    # print(model_path)
    # print(model_path.exists())
    trainer_config = LLaMALoraSFTConfig(
        model_name_or_path=model_path,
        # adapter_name_or_path=

        do_train=True,
        predict_with_generate=False,
        lora_rank=8,
        lora_alpha=16,

        template='llama3',
        cutoff_len=2048,
        # max_samples=25,  # ===
        overwrite_cache=True,
        preprocessing_num_workers=16,

        logging_steps=100,  # ===
        save_steps=1000,  # ===
        plot_loss=True,
        overwrite_output_dir=True,

        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=10,
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,

        eval_steps=10**10,  # ===
    )
    
    extra_setting = ExtraSetting(
        rest_mem_mb=10000,
        wait_before_start=3,
        output_scores=False,
        do_dev=False,
    )

    cuda_id = CUDAUtils.set_cuda_visible(
        target_mem_mb=20000,
        cuda_cnt=1,
        device_range=None,
    )

    ROOT_DIR = path('/public/home/hongy/zpwang/IDRR_Subtext')
    main = LLaMA_zp(
        trainset_config=trainset_config,
        testset_config=OneShotDatasetConfig(),
        trainer_config=trainer_config,
        extra_setting=extra_setting,
        output_dir=ROOT_DIR/'exp_space'/'Inbox',
        desc='subtext_distilled',
        cuda_id=cuda_id,
    )
    main._version_info_list = [
        Datetime_().format_str(2), 
        f'{dfs.data_name}_{dfs.data_level}',
        main.desc, 
        f'bs{main.trainer_config.per_device_train_batch_size}-{main.trainer_config.gradient_accumulation_steps}_lr{main.trainer_config.learning_rate}_ep{main.trainer_config.num_train_epochs}.train'
    ]
    
    main.start()

from llama_zp import *


if __name__ == "__main__":
    data_name = 'pdtb2'
    data_path = '/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_llm/pdtb2.gpt-3.5-turbo.subtext_base/pdtb2.gpt-3.5-turbo.subtext_base.csv'
    dfs = IDRRDataFrames(
        data_name=data_name,
        data_level='top',
        data_relation='Implicit',
        data_path=data_path,
    )
    trainset_config = IDRRDatasetConfig(
        data_split='train',
        prompt={
            "instruction": '''
Argument 1:
{arg1}

Argument 2:
{arg2}

What's the implicit meaning between the arguments?
'''.strip(),
            "input": '',
            "output": '{subtext}',
            "system": "",
            "history": [],
        },
        desc='gpt-3.5_distill_llama',
        **dfs.arg_dic,
    )

    # model_path = path('/public/home/hongy/pretrained_models/Llama-3-8B-Instruct').resolve()
    model_path = '/public/home/hongy/pretrained_models/Llama-3.2-1B-Instruct'
    model_path = '/public/home/hongy/pretrained_models/Meta-Llama-3-8B-Instruct'
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
        learning_rate=1e-4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,

        eval_steps=10**9,  # ===
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
        desc='gpt3.5_distill_llama',
        cuda_id=cuda_id,
    )
    main._version_info_list = [
        Datetime_().format_str(2), data_name, main.desc, 
        f'bs{main.trainer_config.per_device_train_batch_size}-{main.trainer_config.gradient_accumulation_steps}_lr{main.trainer_config.learning_rate}_ep{main.trainer_config.num_train_epochs}.train'
    ]
    
    main.start()


        
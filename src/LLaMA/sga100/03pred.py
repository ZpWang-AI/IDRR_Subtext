from llama_zp import *


if __name__ == "__main__":
    data_split = 'dev'
    data_split = 'test'
    data_split = 'train'
    dfs = IDRRDataFrames(
        data_name='pdtb2',
        data_level='top',
        data_relation='Implicit',
        data_path='/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_llm/pdtb2.gpt-3.5-turbo.subtext_base/pdtb2.gpt-3.5-turbo.subtext_base.csv'
    )
    testset_config = IDRRDatasetConfig(
        data_split=data_split,
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

    model_path = '/public/home/hongy/pretrained_models/Llama-3.2-1B-Instruct'
    model_path = '/public/home/hongy/pretrained_models/Meta-Llama-3-8B-Instruct'
    model_path = path(model_path).resolve()
    # ckpt_path = path('/public/home/hongy/zpwang/LLaMA-Factory_zp/exp_space/Inbox/2024-12-18_07-28-07._local_test.bs1-8_lr5e-05_ep5.succeed/src_output/checkpoint-16').resolve()
    # print(model_path)
    # print(model_path.exists())
    trainer_config = LLaMALoraSFTConfig(
        model_name_or_path=model_path,
        # adapter_name_or_path=ckpt_path,

        do_train=False,
        do_eval=False,
        do_predict=True,
        predict_with_generate=True,
        lora_rank=8,
        lora_alpha=16,

        template='llama3',
        cutoff_len=2048,
        # max_samples=32, # ===
        overwrite_cache=True,
        preprocessing_num_workers=16,

        logging_steps=100,
        save_steps=1000,
        plot_loss=True,
        overwrite_output_dir=True,

        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,

        eval_steps=10**10,
    )
    
    extra_setting = ExtraSetting(
        rest_mem_mb=10000,
        wait_before_start=3,
        output_scores=True,
        do_dev=False,
    )

    target_mem_mb = 20000
    cuda_id = CUDAUtils.set_cuda_visible(
        target_mem_mb=target_mem_mb,
        cuda_cnt=1,
        device_range=None,
    )

    def predict(ckpt_path, ckpt_num):
        trainer_config.adapter_name_or_path = ckpt_path

        testset_config.set_create_time()
        trainer_config.set_create_time()
        extra_setting.set_create_time()

        CUDAUtils.get_free_cudas(
            target_mem_mb=target_mem_mb,
            cuda_cnt=1,
            device_range=[int(cuda_id)],
        )
        main = LLaMA_zp(
            trainset_config=OneShotDatasetConfig(),
            testset_config=testset_config,
            trainer_config=trainer_config,
            extra_setting=extra_setting,
            # output_dir=ROOT_DIR/'exp_space'/'Inbox',
            output_dir=ckpt_dir.parent,
            desc=f'gpt3.5_distill_llama_{data_split}',
            cuda_id=cuda_id,
        )
        main._version_info_list = [
            Datetime_().format_str(2), 
            dfs.data_name,
            main.desc, 
            # f'bs{main.trainer_config.per_device_train_batch_size}-{main.trainer_config.gradient_accumulation_steps}_lr{main.trainer_config.learning_rate}_ep{main.trainer_config.num_train_epochs}.pred.ckpt-{ckpt_num}'
            f'pred.ckpt-{ckpt_num}',
        ]
        
        main.start()
        # time.sleep(10)
        # exit()

    ckpt_dir = '/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/2025-01-10_11-02-40._gpt3.5_distill_llama.bs1-8_lr0.0001_ep5.train'
    ckpt_dir = path(ckpt_dir) / 'src_output'

    # to_predict_list = []
    # for p in sorted(listdir_full_path(ckpt_dir)):
    #     if p.stem.startswith('checkpoint-'):
    #         to_predict_list.append((p, p.stem.split('-')[-1]))

    # to_predict_list.sort(key=lambda x:int(x[1]))
    # for a,b in to_predict_list:
    #     predict(a,b)
    predict(ckpt_dir, 'final')
        
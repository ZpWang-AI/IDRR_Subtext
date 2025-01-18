from utils_zp import *
from IDRR_data import *


def build_csv_distilled(
        dfs:IDRRDataFrames,
        train_dir,
        dev_dir,
        test_dir,    
        output_path,
    ):
    def build_part(split, _dir):
        df = dfs.get_dataframe(split)
        target_jsonl = path(_dir)/'src_output'/'generated_predictions.jsonl'
        preds = auto_load(target_jsonl)
        distilled_subtext = []
        assert df.shape[0] == len(preds)
        for (p, row), pred in tqdm.tqdm(zip(df.iterrows(), preds)):
            distilled_subtext.append(pred['predict'])
        df['subtext'] = distilled_subtext
        return df
    
    df_list = [
        build_part('train', train_dir),
        build_part('dev', dev_dir),
        build_part('test', test_dir),
    ]
    final_df = pd.concat(df_list, axis=0)
    final_df = IDRRDataFrames.del_new_columns(final_df)
    auto_dump(final_df, output_path)
    return final_df


if __name__ == '__main__':
    build_csv_distilled(
        dfs=IDRRDataFrames(
            'pdtb3',
            'raw',
            'Implicit',
            '/public/home/hongy/zpwang/IDRR_Subtext/data/subtext_llm/pdtb3.gpt-3.5-turbo.subtext_base/pdtb3.gpt-3.5-turbo.subtext_base.csv'
        ),
        train_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/2025-01-10_11-02-40.pdtb3_gpt3.5_distill_llama.bs1-8_lr0.0001_ep5.train/2025-01-16_06-31-13.gpt3.5_distill_llama_train.pred.ckpt-final',
        dev_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/2025-01-10_11-02-40.pdtb3_gpt3.5_distill_llama.bs1-8_lr0.0001_ep5.train/2025-01-11_11-43-00.gpt3.5_distill_llama_dev.pred.ckpt-final',
        test_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/2025-01-10_11-02-40.pdtb3_gpt3.5_distill_llama.bs1-8_lr0.0001_ep5.train/2025-01-11_12-48-08.gpt3.5_distill_llama_test.pred.ckpt-final',
        output_path=path(
            '/public/home/hongy/zpwang/IDRR_Subtext/data',
            'subtext_distilled', 'pdtb3.llama3.subtext_base.csv'
        )
    )

        
from IDRR_data import *
from llm_api import *
from utils_zp import *


@config_args
@dataclass
class SubtextGenerator:
    prompt:str = ''
    llm_name:str = ''
    desc:str = ''
    IDRR_dataframes:dict = field(
        default_factory=lambda: {}
    )
    data_split:str = ''
    # n_reasoning_per_sample:int = 1
    max_sample:int = 1000000000
    
    @property
    def version(self):
        return f'{self.IDRR_dataframes["data_name"]}.{self.llm_name}.{self.desc}'

    def start(self):
        dfs = IDRRDataFrames(
            **self.IDRR_dataframes
        )
        src_dir = path(__file__).parent.parent
        root_dir = src_dir.parent
        target_dir = root_dir/'data'/'subtext_llm' / self.version
        print('>', target_dir)
        auto_dump(self, target_dir/'args.json')
        df = dfs.get_dataframe(self.data_split)
        # df = df[pd.notna(df['subtext'])]
        _cnt = self.max_sample
        for p, row in df.iterrows():
            if pd.isna(row['subtext']):
                query = PromptFiller.fill_prompt(row, self.prompt)
                response:Messages = llm_api(query, self.llm_name, print_messages=True)
                df.loc[p, 'subtext'] = response.value[-1]['content']
                _cnt -= 1
                if _cnt == 0:
                    break
        df = IDRRDataFrames.del_new_columns(df)
        auto_dump(df, target_dir / f'{self.version}.csv')


if __name__ == '__main__':
#     SubtextGenerator(
#         prompt='''
# Argument 1:
# {arg1}

# Argument 2:
# {arg2}

# What's the implicit meaning between the arguments?
#         '''.strip(),
#         llm_name='gpt-3.5-turbo',
#         desc='subtext_base',
#         IDRR_dataframes=IDRRDataFrames(
#             'pdtb3', 'top', 'Implicit', 
#             r'D:\ZpWang\Projects\02.05-IDRR_Subtext\IDRR_Subtext\data\used_subtext\pdtb3_top_implicit.subtext2.csv'
#         ).arg_dic,
#         data_split='All',
#         max_sample=10**10,
#     ).start()

    SubtextGenerator(
        prompt='''
Argument 1:
{arg1}

Argument 2:
{arg2}

What's the implicit meaning between the arguments?
        '''.strip(),
        llm_name='gpt-3.5-turbo',
        desc='subtext_base',
        IDRR_dataframes=IDRRDataFrames(
            'pdtb2', 'top', 'Implicit', 
            '/public/home/hongy/zpwang/IDRR_Subtext/data/used_subtext/pdtb2_top_implicit.subtext.csv'
        ).arg_dic,
        data_split='All',
        max_sample=10**10,
        # max_sample=3,
    ).start()
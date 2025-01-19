from utils_zp import *
from IDRR_data import *

from sklearn.metrics import f1_score


class Analyser:
    @classmethod
    def analyse_result(cls, target_pred_dir):
        pred_results = auto_load(
            path(target_pred_dir, 'src_output', 'generated_scores.jsonl')
        )
        preds, labels = [], []
        for line in pred_results:
            preds.append(line['output_str'][0])
            labels.append(line['label_str'][0])
        metrics = cls._cal_metric(preds, labels)

        auto_dump(
            metrics,
            path(target_pred_dir, 'metrics.json'),
        )
        return metrics

    @classmethod
    def analyse_result_with_confidence_score(cls, target_pred_dir, source_pred_dir):
        pass

    @classmethod
    def _cal_metric(cls, preds:List[str], labels:List[str]):
        tot = len(preds)
        assert len(preds)==len(labels)
        # print(labels)
        label_list = list(set(labels))
        wrong_outputs = [p for p in preds if p not in label_list]
        def to_lid(_s:str):
            for lid, label in enumerate(label_list):
                # if label_s.startswith(label):
                # if label in label_s.split('\n'):
                if _s == label:
                # if label in _s or _s in label:
                    return lid
            return -1
        preds = np.array(list(map(to_lid, preds)))
        labels = np.array(list(map(to_lid, labels)))

        acc = (preds==labels).mean()
        f1 = [f1_score(labels==i, preds==i)for i in range(len(label_list))]
        macro_f1 = np.average(f1)
        return {
            'tot': tot,
            'acc': float(f'{acc*100:.3f}'),
            'macro-f1': float(f'{macro_f1*100:.3f}'),
            'f1': f1,
            'labels': label_list,
            'wrong': wrong_outputs,
        }
    
    @classmethod
    def analyse_all_pred_results(cls, root_dir):
        xs, ys = [], []
        wrong_outputs = []
        for _dir in listdir_full_path(root_dir):
            # print(_dir)
            # print(_dir.stem)
            if '.pred.' in _dir.name and 'ckpt' in _dir.name:
                metrics = cls.analyse_result(_dir)
                ckpt_num = _dir.name.split('-')[-1]
                try:
                    ckpt_num = int(ckpt_num)
                except:
                    continue
                xs.append(ckpt_num)
                ys.append(metrics['macro-f1'])
                wrong_outputs.extend(metrics['wrong'])

        auto_dump(
            wrong_outputs,
            path(root_dir, 'wrong_outputs.json')
        )
        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(xs, ys)
        plt_utils.mark_extremum(
            xs, ys, mark_max=True,
            format_y_func=lambda y: f'{y:.3f}'
        )
        plt.xlabel('step')
        plt.ylabel('macro-f1')
        plt.title('test_macro-f1')
        img_path = path(root_dir, 'test_macro_f1.png')
        plt.savefig(img_path)
        plt.close()
        print(img_path, 'saved')


if __name__ == '__main__':
    Analyser.analyse_all_pred_results(
        # '/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/2025-01-07_11-59-07.pdtb3_second._baseline.bs1-8_lr0.0001_ep5.train'
        '/public/home/hongy/zpwang/IDRR_Subtext/exp_space/Inbox/2025-01-17_13-38-43.pdtb3_second.subtext_distilled.bs1-8_lr0.0001_ep5.train'
    )
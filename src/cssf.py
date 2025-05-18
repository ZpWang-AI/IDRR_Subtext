from utils_zp import *

from analyse import Analyser


class CSSF:
    @staticmethod
    def get_threshold(scores, source_preds, target_preds, labels, type_):
        source_preds = [p==type_ and l==type_ for p, l in zip(source_preds, labels)]
        target_preds = [p==type_ and l==type_ for  p, l in zip(target_preds, labels)]
        lst = list(zip(scores, source_preds, target_preds))
        lst.sort(key=lambda x: -x[0])
        tot = len(lst)
        correct = sum(target_preds)
        # res, max_acc = -1, -1
        acc, score_ = [], []
        for score, sy, ty in lst:
            correct += sy
            correct -= ty
            acc.append(correct/tot)
            score_.append(score)

        threshold = score_[acc.index(max(acc))]
        return threshold
    
    @staticmethod
    def get_final_by_threshold(
        thresholds,
        source_scores,
        source_preds,
        # target_score,
        target_preds,
    ):
        final_pred = []
        for source_s, source_p, target_p in zip(source_scores, source_preds, target_preds):
            if source_s >= thresholds[source_p]:
                final_pred.append(source_p)
            else:
                final_pred.append(target_p)
        return final_pred
    
    @classmethod
    def main(
        cls,
        source_threshold_dir,
        target_threshold_dir,
        source_dir,
        target_dir, 
        final_res_path,
    ):
        def get_label_list(_dir):
            output = auto_load(_dir)
            label_list = []
            for line in output:
                label_list.append(line['label_str'][0])
            return sorted(set(label_list))
        label_list = get_label_list(path(source_threshold_dir, 'src_output', 'generated_scores.jsonl'))

        def to_label_id(label_str):
            return label_list.index(label_str) if label_str in label_list else len(label_list)

        def get_output(output_dir):
            output = auto_load(output_dir)
            score, pred, label = [], [], []
            for line in output:
                cur_score = line['output_scores']
                cur_score = np.array(cur_score).mean()
                score.append(cur_score)
                pred.append(to_label_id(line['output_str'][0]))
                label.append(to_label_id(line['label_str'][0]))
            return {
                'score': score,
                'pred': pred,
                'label': label,
            }

        source_threshold_output = get_output(
            path(source_threshold_dir, 'src_output', 'generated_scores.jsonl')
        )
        target_threshold_output = get_output(
            path(target_threshold_dir, 'src_output', 'generated_scores.jsonl')
        )
        source_output = get_output(
            path(source_dir, 'src_output', 'generated_scores.jsonl')
        )
        target_output = get_output(
            path(target_dir, 'src_output', 'generated_scores.jsonl')
        )
        assert len(source_threshold_output['score'])==len(target_threshold_output['score']) \
            and len(source_output['score'])==len(target_output['score']), \
            (
                len(source_threshold_output['score']), 
                len(target_threshold_output['score']), 
                len(source_output['score']),
                len(target_output['score'])
            )
        # if len(source_threshold_output['score']) != len(target_threshold_output['score']):
        #     source_threshold_dir['score']

        thresholds = [
            cls.get_threshold(
                scores=source_threshold_output['score'],
                source_preds=source_threshold_output['pred'],
                target_preds=target_threshold_output['pred'],
                labels=source_threshold_output['label'],
                type_=i,
            )
            for i in range(len(label_list))
        ]
        thresholds.append(100000)
        for thr, label in zip(thresholds, label_list):
            print(label, thr)

        final_pred = cls.get_final_by_threshold(
            thresholds=thresholds,
            source_scores=source_output['score'],
            source_preds=source_output['pred'],
            target_preds=target_output['pred'],
        )

        final_metrics = Analyser.cal_metric(
            preds=final_pred,
            labels=source_output['label'],
        )
        
        print(final_metrics)
        final_metrics['thresholds'] = thresholds

        auto_dump(final_metrics, final_res_path)
        return final_metrics


if __name__ == '__main__':
    CSSF.main(
        # source_threshold_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb3_sec/cssf/2025-02-02_13-26-56._baseline.pred.ckpt-10000.train',
        source_threshold_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb3_sec/cssf/2025-02-13_13-15-44._baseline.pred.ckpt-10000.train',
        target_threshold_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb3_sec/cssf/2025-02-02_10-25-55.subtext_distilled.pred.ckpt-8000.train',
        source_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb3_sec/cssf/2025-02-13_13-17-01._baseline.pred.ckpt-10000',
        target_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb3_sec/cssf/2025-01-23_06-07-15.subtext_distilled.pred.ckpt-8000.test',
        final_res_path='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb3_sec/cssf/cssf_metrics_test.json',
    )

    # CSSF.main(
    #     source_threshold_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb2_sec/2025-02-03_06-46-27.pdtb2_second._baseline.bs1-8_lr0.0001_ep5.train/2025-02-05_22-18-03._baseline.pred.ckpt-7000.train',
    #     target_threshold_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb2_sec/2025-01-19_12-07-41.pdtb2_second.subtext_distilled.bs1-8_lr0.0001_ep10.train/2025-02-06_09-17-57.subtext_distilled.pred.ckpt-4000.train',
    #     source_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb2_sec/cssf/2025-02-03_15-37-17._baseline.pred.ckpt-7000',
    #     target_dir='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb2_sec/cssf/2025-01-21_10-37-01.subtext_distilled.pred.ckpt-4000',
    #     final_res_path='/public/home/hongy/zpwang/IDRR_Subtext/exp_space/result/pdtb2_sec/cssf/cssf_metrics.json'
    # )
import numpy as np
from scipy import stats

# 1. 模型的准确率数据
# 使用表格中的准确率数据
text_rnn_accuracies = [0.8442, 0.8430, 0.8465]
fasttext_accuracies = [0.8405, 0.8420, 0.8390]
textcnn_accuracies = [0.8609, 0.8615, 0.8602]
bert_rnn_accuracies = [0.8707, 0.8725, 0.8690]
bert_accuracies = [0.8736, 0.8740, 0.8720]
bert_cnn_accuracies = [0.8854, 0.8845, 0.8850]
tann_accuracies = [0.8895, 0.8900, 0.8890]


# 2. 对比TANN与基线模型进行t检验和计算置信区间

def perform_ttest_and_ci(model_accuracies, model_name):
    # 计算t检验
    t_stat, p_value = stats.ttest_ind(tann_accuracies, model_accuracies)

    print(f"\nTANN vs {model_name}:")
    print(f"T-statistic: {t_stat}, p-value: {p_value}")
    if p_value < 0.05:
        print(f"TANN和{model_name}之间的差异具有显著性 (p < 0.05)")
    else:
        print(f"TANN和{model_name}之间的差异不显著 (p >= 0.05)")

    # 计算95%置信区间
    mean_tann = np.mean(tann_accuracies)
    std_tann = np.std(tann_accuracies, ddof=1)
    n_tann = len(tann_accuracies)

    confidence_interval_tann = stats.t.interval(0.95, n_tann - 1, loc=mean_tann, scale=std_tann / np.sqrt(n_tann))

    mean_model = np.mean(model_accuracies)
    std_model = np.std(model_accuracies, ddof=1)
    n_model = len(model_accuracies)

    confidence_interval_model = stats.t.interval(0.95, n_model - 1, loc=mean_model, scale=std_model / np.sqrt(n_model))

    print(f"TANN的95%置信区间: {confidence_interval_tann}")
    print(f"{model_name}的95%置信区间: {confidence_interval_model}")


# 对比TANN与所有基线模型
perform_ttest_and_ci(text_rnn_accuracies, "TextRNN")
perform_ttest_and_ci(fasttext_accuracies, "FastText")
perform_ttest_and_ci(textcnn_accuracies, "TextCNN")
perform_ttest_and_ci(bert_rnn_accuracies, "BERT_RNN")
perform_ttest_and_ci(bert_accuracies, "BERT")
perform_ttest_and_ci(bert_cnn_accuracies, "BERT_CNN")

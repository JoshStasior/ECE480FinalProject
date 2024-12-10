import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import multivariate_normal



NUM_DIGITS = 10
NUM_CEPSTRA = 13
UTTERANCES_PER_DIGIT_TRAIN = 660
UTTERANCES_PER_DIGIT_TEST = 220

phoneme_clusters = {
    0: 4, 1: 3, 2: 5, 3: 5, 4: 4,
    5: 5, 6: 5, 7: 3, 8: 5, 9: 3
}



def read_mfcc_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    blocks = []
    current_block = []
    for line in lines:
        line=line.strip()
        if line == '':
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            float_values = [float(x) for x in line.split()]
            current_block.append(float_values)
    if current_block:
        blocks.append(current_block)
    return blocks

def extract_digit_blocks(blocks, digit, utterances_per_digit):
    start_index = digit * utterances_per_digit
    end_index = start_index + utterances_per_digit
    return blocks[start_index:end_index]


def fit_em_gmm_tied(digit_blocks, digit):
    """
    Fit a GaussianMixture model with tied covariance for each digit.
    The number of components is determined by phoneme_clusters[digit].
    """
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', 
                          random_state=0, n_init=1, reg_covar=1e-6)
    gmm.fit(data)
    return gmm



def classify_token_em(token_data, em_gmms):
    """
    Classify a single token by choosing the model that gives the highest log-likelihood.
    """
    best_digit = None
    best_ll = -np.inf
    for d, model in em_gmms.items():
        # log probability of all samples in token_data under model
        ll = model.score(token_data)*len(token_data)
        if ll > best_ll:
            best_ll = ll
            best_digit = d
    return best_digit

def classify_all_tokens_em(blocks, em_gmms, train=False):
    """
    Classify all tokens in the dataset (train or test) using the EM GMM models.
    """
    predictions = []
    true_labels = []
    per_digit_utterances = UTTERANCES_PER_DIGIT_TRAIN if train else UTTERANCES_PER_DIGIT_TEST
    for digit in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(blocks, digit, per_digit_utterances)
        for token in digit_blocks:
            token_data = np.array(token)
            pred = classify_token_em(token_data, em_gmms)
            predictions.append(pred)
            true_labels.append(digit)
    return np.array(true_labels), np.array(predictions)



def evaluate_performance(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    return cm, acc

def plot_confusion_matrix(cm, acc, title='EM Tied Covariance GMM Confusion Matrix'):
    cm_prob = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8,6))
    df_cm = pd.DataFrame(cm_prob, index=range(NUM_DIGITS), columns=range(NUM_DIGITS))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f"{title}\nAccuracy: {acc:.4f}")
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    plt.show()



def main():
    # Load training and test data
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')

    # Fit EM-based Tied Covariance GMM for each digit
    em_gmms = {}
    for d in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
        em_gmms[d] = fit_em_gmm_tied(digit_blocks, d)

    # Classify the test set
    y_true_em, y_pred_em = classify_all_tokens_em(test_blocks, em_gmms, train=False)

    # Evaluate performance
    cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
    print("EM Tied Covariance GMM Accuracy:", acc_em)
    plot_confusion_matrix(cm_em, acc_em)

if __name__ == "__main__":
    main()

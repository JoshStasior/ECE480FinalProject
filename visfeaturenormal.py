import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
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
    """Read MFCC data and organize into blocks (utterances)."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    blocks = []
    current_block = []
    for line in lines:
        line = line.strip()
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


def compute_normalization_params(blocks):
    """
    Compute the mean and std for each MFCC coefficient across all frames in all training tokens.
    """
    all_frames = []
    for block in blocks:
        for frame in block:
            all_frames.append(frame)
    all_frames = np.array(all_frames)  # shape: (total_frames, NUM_CEPSTRA)
    mean_vec = np.mean(all_frames, axis=0)
    std_vec = np.std(all_frames, axis=0) + 1e-12  # Avoid division by zero
    return mean_vec, std_vec

def normalize_blocks(blocks, mean_vec, std_vec):
    """
    Apply normalization (Z-score) to blocks using given mean_vec and std_vec.
    """
    norm_blocks = []
    for block in blocks:
        block_arr = np.array(block)
        norm_block = (block_arr - mean_vec) / std_vec
        norm_blocks.append(norm_block.tolist())
    return norm_blocks


def kmeans_gmm_params(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(data)
    labels = kmeans.labels_

    means = kmeans.cluster_centers_
    covariances = []
    weights = []

    for c in range(n_clusters):
        cluster_data = data[labels == c]
        weights.append(len(cluster_data) / len(data))
        cov = np.cov(cluster_data, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        covariances.append(cov)

    covariances = np.array(covariances)
    weights = np.array(weights)
    return means, covariances, weights

def fit_kmeans_gmm(digit_blocks, digit):
    if len(digit_blocks) == 0:
        return None, None, None
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    means, covs, weights = kmeans_gmm_params(data, n_clusters)
    return means, covs, weights


def fit_gmm_em(digit_blocks, digit, covariance_type='full', n_init=1):
    if len(digit_blocks) == 0:
        return None
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                          random_state=0, n_init=n_init, reg_covar=1e-6)
    gmm.fit(data)
    return gmm



def gmm_likelihood(data, means, covariances, weights):
    N = data.shape[0]
    K = len(weights)
    pdf_vals = np.zeros((N, K))
    for k in range(K):
        pdf_vals[:, k] = weights[k]*multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])
    mixture = np.sum(pdf_vals, axis=1)
    return np.sum(np.log(mixture + 1e-12))

def em_gmm_likelihood(data, gmm: GaussianMixture):
    return gmm.score(data)*len(data)

def classify_token(token_data, kmeans_gmms=None, em_gmms=None, use_em=False):
    if use_em:
        best_digit = None
        best_ll = -np.inf
        for d, model in em_gmms.items():
            if model is None:
                continue
            ll = em_gmm_likelihood(token_data, model)
            if ll > best_ll:
                best_ll = ll
                best_digit = d
        if best_digit is None:
            return 0
        return best_digit
    else:
        best_digit = None
        best_ll = -np.inf
        for d, params in kmeans_gmms.items():
            if params[0] is None:
                continue
            means, covs, weights = params
            ll = gmm_likelihood(token_data, means, covs, weights)
            if ll > best_ll:
                best_ll = ll
                best_digit = d
        if best_digit is None:
            return 0
        return best_digit

def classify_all_tokens(blocks, kmeans_gmms=None, em_gmms=None, use_em=False, train=False, mean_vec=None, std_vec=None):
    predictions = []
    true_labels = []
    per_digit_utterances = UTTERANCES_PER_DIGIT_TRAIN if train else UTTERANCES_PER_DIGIT_TEST
    for digit in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(blocks, digit, per_digit_utterances)
        for token in digit_blocks:
            token_arr = np.array(token)
            # Apply normalization if requested
            if mean_vec is not None and std_vec is not None:
                token_arr = (token_arr - mean_vec) / std_vec
            pred = classify_token(token_arr, kmeans_gmms=kmeans_gmms, em_gmms=em_gmms, use_em=use_em)
            predictions.append(pred)
            true_labels.append(digit)
    return np.array(true_labels), np.array(predictions)



def evaluate_performance(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    cm_prob = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    return cm_prob, acc

def plot_confusion_matrix(cm, title='Confusion Matrix (Percentages)'):
    plt.figure(figsize=(8,6))
    df_cm = pd.DataFrame(cm, index=range(NUM_DIGITS), columns=range(NUM_DIGITS))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()



def main():
    # Load data
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')


    scenarios = [
        ("No Normalization", False),
        ("Normalization", True)
    ]

    results = []

    for scenario_name, do_normalize in scenarios:
        print(f"\n=== Evaluating Scenario: {scenario_name} ===")

        digit_train_data = {}
        # Just separate data by digit for training
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
            digit_train_data[d] = digit_blocks

        # Compute normalization parameters if needed
        mean_vec, std_vec = None, None
        if do_normalize:
            # Compute global mean/std from all training data
            all_train_blocks = []
            for d in range(NUM_DIGITS):
                all_train_blocks.extend(digit_train_data[d])
            mean_vec, std_vec = compute_normalization_params(all_train_blocks)

            # Normalize training data
            for d in range(NUM_DIGITS):
                digit_train_data[d] = normalize_blocks(digit_train_data[d], mean_vec, std_vec)

        # Train K-means GMM
        start_time_kmeans = time.time()
        kmeans_gmms = {}
        for d in range(NUM_DIGITS):
            means, covs, weights = fit_kmeans_gmm(digit_train_data[d], d)
            kmeans_gmms[d] = (means, covs, weights)
        end_time_kmeans = time.time()
        print(f"Time taken to train K-means GMMs: {end_time_kmeans - start_time_kmeans:.2f} seconds")

        # Evaluate K-Means GMM
        y_true_km, y_pred_km = classify_all_tokens(test_blocks, kmeans_gmms=kmeans_gmms, use_em=False,
                                                   train=False, mean_vec=mean_vec, std_vec=std_vec)
        cm_km, acc_km = evaluate_performance(y_true_km, y_pred_km)
        print("K-means GMM Accuracy:", acc_km)
        plot_confusion_matrix(cm_km, title=f'K-means GMM Confusion Matrix ({scenario_name})')

        # Train EM GMM
        start_time_em = time.time()
        em_gmms = {}
        for d in range(NUM_DIGITS):
            em_gmms[d] = fit_gmm_em(digit_train_data[d], d, covariance_type='full')
        end_time_em = time.time()
        print(f"Time taken to train EM GMMs: {end_time_em - start_time_em:.2f} seconds")

        # Evaluate EM GMM
        y_true_em, y_pred_em = classify_all_tokens(test_blocks, em_gmms=em_gmms, use_em=True,
                                                   train=False, mean_vec=mean_vec, std_vec=std_vec)
        cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
        print("EM GMM Accuracy:", acc_em)
        plot_confusion_matrix(cm_em, title=f'EM GMM Confusion Matrix ({scenario_name})')

        results.append((scenario_name, acc_km, acc_em))

    # Plot comparison of accuracies
    labels = [r[0] for r in results]
    km_accuracies = [r[1] for r in results]
    em_accuracies = [r[2] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10,6))
    kmeans_bars = plt.bar(x - width/2, np.array(km_accuracies)*100, width, label='K-Means GMM', alpha=0.7)
    em_bars = plt.bar(x + width/2, np.array(em_accuracies)*100, width, label='EM GMM', alpha=0.7)

    for bar, acc in zip(kmeans_bars, km_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f"{acc*100:.2f}%", ha='center', va='bottom', fontsize=10)
    for bar, acc in zip(em_bars, em_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f"{acc*100:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.xticks(x, labels)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracies for Feature Normalization Constraint')
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

    # Print summary of results
    print("\nSummary of Accuracies:")
    for scenario_name, km_acc, em_acc in results:
        print(f"Scenario={scenario_name}: K-Means Acc={km_acc:.4f}, EM Acc={em_acc:.4f}")

if __name__ == "__main__":
    main()

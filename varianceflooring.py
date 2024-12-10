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



def apply_variance_floor(covariances, floor):
    """
    Enforce a variance floor on the diagonal elements of covariance matrices.
    """
    if covariances.ndim == 2:  
        # Single covariance matrix (tied)
        diag_indices = np.arange(covariances.shape[0])
        covariances[diag_indices, diag_indices] = np.maximum(
            covariances[diag_indices, diag_indices], floor
        )
        return covariances
    else:
        # Multiple covariance matrices
        for c in range(covariances.shape[0]):
            diag_indices = np.arange(covariances.shape[1])
            covariances[c, diag_indices, diag_indices] = np.maximum(
                covariances[c, diag_indices, diag_indices], floor
            )
        return covariances

def compute_precisions_cholesky(covariances):
    """
    Compute precisions_cholesky for 'full' covariance type.
    precisions_cholesky[k] = inv(cholesky(covariances[k]))^T
    """
    n_components, n_features, _ = covariances.shape
    precisions_cholesky = np.zeros_like(covariances)
    for k in range(n_components):
        cov_chol = np.linalg.cholesky(covariances[k])
        precisions_cholesky[k] = np.linalg.inv(cov_chol).T
    return precisions_cholesky


def kmeans_gmm_params(data, n_clusters, floor):
    """Run k-means on data and derive GMM parameters: means, covariances, weights."""
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
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularization
        cov = apply_variance_floor(cov, floor)
        covariances.append(cov)

    covariances = np.array(covariances)
    weights = np.array(weights)
    return means, covariances, weights

def fit_kmeans_gmm(digit_blocks, digit, floor):
    if len(digit_blocks) == 0:
        return None, None, None
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    means, covs, weights = kmeans_gmm_params(data, n_clusters, floor)
    return means, covs, weights



def fit_gmm_em(digit_blocks, digit, floor, covariance_type='full', n_init=1):
    if len(digit_blocks) == 0:
        return None
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                          random_state=0, n_init=n_init, reg_covar=1e-6)
    gmm.fit(data)

    # Apply variance floor if 'full' covariance
    if covariance_type == 'full':
        gmm.covariances_ = apply_variance_floor(gmm.covariances_, floor)
        # Recompute precisions_cholesky manually
        gmm.precisions_cholesky_ = compute_precisions_cholesky(gmm.covariances_)

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

def classify_all_tokens(blocks, kmeans_gmms=None, em_gmms=None, use_em=False, train=False):
    predictions = []
    true_labels = []
    per_digit_utterances = UTTERANCES_PER_DIGIT_TRAIN if train else UTTERANCES_PER_DIGIT_TEST
    for digit in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(blocks, digit, per_digit_utterances)
        for token in digit_blocks:
            token_data = np.array(token)
            pred = classify_token(token_data, kmeans_gmms=kmeans_gmms, em_gmms=em_gmms, use_em=use_em)
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
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    plt.show()



def main():
    # Load data
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')

    # Try different variance floors
    variance_floors = [0,.1,.25,.5]

    km_accuracies = []
    em_accuracies = []
    labels = []

    for floor in variance_floors:
        print(f"\n=== Evaluating Variance Floor = {floor} ===")

        # Train K-means GMM
        start_time_kmeans = time.time()
        kmeans_gmms = {}
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
            means, covs, weights = fit_kmeans_gmm(digit_blocks, d, floor)
            kmeans_gmms[d] = (means, covs, weights)
        end_time_kmeans = time.time()
        print(f"Time taken to train K-means GMMs: {end_time_kmeans - start_time_kmeans:.2f} seconds")

        # Evaluate K-Means GMM
        y_true_km, y_pred_km = classify_all_tokens(test_blocks, kmeans_gmms=kmeans_gmms, use_em=False, train=False)
        cm_km, acc_km = evaluate_performance(y_true_km, y_pred_km)
        print("K-means GMM Accuracy:", acc_km)
        plot_confusion_matrix(cm_km, title=f'K-means GMM Confusion Matrix (Floor={floor})')

        # EM-based GMM
        start_time_em = time.time()
        em_gmms = {}
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
            em_gmms[d] = fit_gmm_em(digit_blocks, d, floor, covariance_type='full')
        end_time_em = time.time()
        print(f"Time taken to train EM GMMs: {end_time_em - start_time_em:.2f} seconds")

        # Evaluate EM GMM
        y_true_em, y_pred_em = classify_all_tokens(test_blocks, em_gmms=em_gmms, use_em=True, train=False)
        cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
        print("EM GMM Accuracy:", acc_em)
        plot_confusion_matrix(cm_em, title=f'EM GMM Confusion Matrix (Floor={floor})')

        km_accuracies.append(acc_km)
        em_accuracies.append(acc_em)
        labels.append(f"{floor}")

    # Plot comparison of accuracies for different floors
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
    plt.title('Accuracies for Different Variance Floors')
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

    # Print summary of results
    print("\nSummary of Accuracies for Different Variance Floors:")
    for lbl, km_acc, em_acc in zip(labels, km_accuracies, em_accuracies):
        print(f"Floor={lbl}: K-Means Acc={km_acc:.4f}, EM Acc={em_acc:.4f}")

if __name__ == "__main__":
    main()

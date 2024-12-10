import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal



NUM_DIGITS = 10
NUM_CEPSTRA = 13
UTTERANCES_PER_DIGIT_TRAIN = 660
UTTERANCES_PER_DIGIT_TEST = 220

# Mixture components per digit
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
        if line == '':  # Blank line indicates new utterance
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            float_values = [float(x) for x in line.split()]
            current_block.append(float_values)

    if current_block:  # Add last block if not empty
        blocks.append(current_block)
    return blocks

def extract_digit_blocks(blocks, digit, utterances_per_digit):
    """Extract blocks for a given digit."""
    start_index = digit * utterances_per_digit
    end_index = start_index + utterances_per_digit
    return blocks[start_index:end_index]


def kmeans_gmm_params(data, n_clusters):
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
        covariances.append(cov)
    
    covariances = np.array(covariances)
    weights = np.array(weights)
    return means, covariances, weights

def apply_covariance_constraint_to_kmeans(means, covs, weights, cov_type):
    """Apply the given covariance structure constraint to the K-means derived parameters."""
    if cov_type == 'full':
        # No change needed
        pass
    elif cov_type == 'tied':
        # Average all covariances
        avg_cov = np.mean(covs, axis=0)
        covs = np.array([avg_cov for _ in range(covs.shape[0])])
    elif cov_type == 'diag':
        # Zero out off-diagonal elements
        for c in range(covs.shape[0]):
            covs[c] = np.diag(np.diag(covs[c]))
    elif cov_type == 'spherical':
        # Convert to diagonal first
        for c in range(covs.shape[0]):
            covs[c] = np.diag(np.diag(covs[c]))
        # Now average diagonals for each component
        for c in range(covs.shape[0]):
            avg_var = np.mean(np.diag(covs[c]))
            covs[c] = np.eye(covs.shape[1]) * avg_var
    return means, covs, weights

def fit_kmeans_gmm(digit_blocks, digit):
    """Fit a GMM to a digit's data using k-means-derived parameters (no EM)."""
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    means, covs, weights = kmeans_gmm_params(data, n_clusters)
    return means, covs, weights


def fit_gmm_em(digit_blocks, digit, covariance_type='full', n_init=1):
    """Fit a GMM using EM algorithm for a specific digit."""
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, 
                          random_state=0, n_init=n_init)
    gmm.fit(data)
    return gmm


def gmm_likelihood(data, means, covariances, weights):
    """Compute log-likelihood of data under a GMM defined by (means, covariances, weights)."""
    N = data.shape[0]
    K = len(weights)
    pdf_vals = np.zeros((N, K))
    for k in range(K):
        pdf_vals[:,k] = weights[k]*multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])
    mixture = np.sum(pdf_vals, axis=1)
    return np.sum(np.log(mixture + 1e-12))

def classify_token(token_data, kmeans_gmms=None, em_gmms=None, use_em=False):
    """Classify a single token using max likelihood over digits."""
    if use_em:
        best_digit = None
        best_ll = -np.inf
        for d, model in em_gmms.items():
            ll = model.score(token_data) * len(token_data)
            if ll > best_ll:
                best_ll = ll
                best_digit = d
        return best_digit
    else:
        best_digit = None
        best_ll = -np.inf
        for d, params in kmeans_gmms.items():
            means, covs, weights = params
            ll = gmm_likelihood(token_data, means, covs, weights)
            if ll > best_ll:
                best_ll = ll
                best_digit = d
        return best_digit

def classify_all_tokens(blocks, kmeans_gmms=None, em_gmms=None, use_em=False, train=False):
    """Classify all tokens in a dataset."""
    predictions = []
    true_labels = []
    per_digit_utterances = UTTERANCES_PER_DIGIT_TRAIN if train else UTTERANCES_PER_DIGIT_TEST
    for digit in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(blocks, digit, utterances_per_digit=per_digit_utterances)
        for token in digit_blocks:
            token_data = np.array(token)
            pred = classify_token(token_data, kmeans_gmms, em_gmms, use_em=use_em)
            predictions.append(pred)
            true_labels.append(digit)
    return np.array(true_labels), np.array(predictions)


def evaluate_performance(true_labels, predicted_labels):
    acc = accuracy_score(true_labels, predicted_labels)
    return acc



def main():
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')

    covariance_types = [
        ('full',     'Full'),
        ('tied',     'Tied'),
        ('diag',     'Diagonal'),
        ('spherical','Spherical')
    ]

    kmeans_accuracies = []
    em_accuracies = []
    kmeans_times = []
    em_times = []
    labels = []

    for cov_type, cov_label in covariance_types:
        print(f"Evaluating with covariance type: {cov_label}")

        # K-means GMM
        start_time_kmeans = time.time()
        kmeans_gmms = {}
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks, d, utterances_per_digit=UTTERANCES_PER_DIGIT_TRAIN)
            means, covs, weights = fit_kmeans_gmm(digit_blocks, d)
            means, covs, weights = apply_covariance_constraint_to_kmeans(means, covs, weights, cov_type)
            kmeans_gmms[d] = (means, covs, weights)
        end_time_kmeans = time.time()
        kmeans_times.append(end_time_kmeans - start_time_kmeans)

        # Evaluate K-means GMM
        y_true_km, y_pred_km = classify_all_tokens(test_blocks, kmeans_gmms=kmeans_gmms, use_em=False, train=False)
        acc_km = evaluate_performance(y_true_km, y_pred_km)
        kmeans_accuracies.append(acc_km)

        # EM-based GMM
        start_time_em = time.time()
        em_gmms = {}
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
            em_gmms[d] = fit_gmm_em(digit_blocks, d, covariance_type=cov_type)
        end_time_em = time.time()
        em_times.append(end_time_em - start_time_em)

        # Evaluate EM GMM
        y_true_em, y_pred_em = classify_all_tokens(test_blocks, em_gmms=em_gmms, use_em=True, train=False)
        acc_em = evaluate_performance(y_true_em, y_pred_em)
        em_accuracies.append(acc_em)

        labels.append(cov_label)

        print(f"{cov_label}: K-means Acc = {acc_km:.2f}, EM Acc = {acc_em:.2f}")

    # Plot the comparison of accuracies and times
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10,6))
    kmeans_bars = plt.bar(x - width/2, np.array(kmeans_accuracies)*100, width, label='K-Means Accuracy', alpha=0.7)
    em_bars = plt.bar(x + width/2, np.array(em_accuracies)*100, width, label='EM Accuracy', alpha=0.7)

    plt.xticks(x, labels)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracies for Different Covariance Structures')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    kmeans_time_bars = plt.bar(x - width/2, kmeans_times, width, label='K-Means Time', alpha=0.7)
    em_time_bars = plt.bar(x + width/2, em_times, width, label='EM Time', alpha=0.7)

    plt.xticks(x, labels)
    plt.ylabel('Time (seconds)')
    plt.title('Training Times for Different Covariance Structures')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

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
        if line == '':
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



def is_outlier_token(token, threshold):
    """
    Determine if a token (frames x MFCCs) is an outlier.
    If any MFCC coefficient in any frame exceeds 'threshold' in absolute value,
    the token is considered an outlier.
    """
    if threshold is None:
        return False
    token_arr = np.array(token)
    # Check if any coefficient exceeds threshold in absolute value
    return np.any(np.abs(token_arr) > threshold)

def remove_outliers(digit_blocks, threshold):
    """
    Remove tokens that are outliers based on the given threshold.
    Returns a new list of digit blocks without outliers.
    """
    if threshold is None:
        return digit_blocks
    filtered = [token for token in digit_blocks if not is_outlier_token(token, threshold)]
    return filtered



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
        cov += np.eye(cov.shape[0]) * 1e-6  # Regularization
        covariances.append(cov)
    
    covariances = np.array(covariances)
    weights = np.array(weights)
    return means, covariances, weights

def fit_kmeans_gmm(digit_blocks, digit):
    """Fit a GMM to a digit's data using k-means-derived parameters (no EM)."""
    if len(digit_blocks) == 0:
        return None, None, None  # No data for this digit
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    means, covs, weights = kmeans_gmm_params(data, n_clusters)
    return means, covs, weights



def fit_gmm_em(digit_blocks, digit, covariance_type='full', n_init=1):
    """Fit a GMM using EM algorithm for a specific digit."""
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
            # If no model was found or all are None,
            # return a default digit label, e.g., 0
            return 0
        return best_digit
    else:
        best_digit = None
        best_ll = -np.inf
        for d, params in kmeans_gmms.items():
            if params[0] is None:  # Means is None if no data for that digit
                continue
            means, covs, weights = params
            ll = gmm_likelihood(token_data, means, covs, weights)
            if ll > best_ll:
                best_ll = ll
                best_digit = d
        if best_digit is None:
            # If no valid GMM was found, return a default digit, e.g., 0
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

    # Different outlier thresholds to try
    outlier_thresholds = [None,10, 9,8,7,6]

    # Store results for final comparison
    km_accuracies = []
    em_accuracies = []

    for threshold in outlier_thresholds:
        if threshold is None:
            print("\n=== Evaluating with No Outlier Removal ===")
        else:
            print(f"\n=== Evaluating with Outlier Threshold = {threshold} ===")

        # Remove outliers from training data (digit by digit)
        digit_train_data = {}
        total_removed = 0
        total_initial = 0
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
            total_initial += len(digit_blocks)
            digit_blocks_clean = remove_outliers(digit_blocks, threshold)
            total_removed += (len(digit_blocks) - len(digit_blocks_clean))
            digit_train_data[d] = digit_blocks_clean

        if threshold is not None:
            print(f"Out of {total_initial} total training tokens, {total_removed} were removed as outliers.")

        # Train K-means GMM
        start_time_kmeans = time.time()
        kmeans_gmms = {}
        for d in range(NUM_DIGITS):
            means, covs, weights = fit_kmeans_gmm(digit_train_data[d], d)
            kmeans_gmms[d] = (means, covs, weights)
        end_time_kmeans = time.time()
        print(f"Time taken to train K-means GMMs: {end_time_kmeans - start_time_kmeans:.2f} seconds")

        # Evaluate K-Means GMM
        y_true_km, y_pred_km = classify_all_tokens(test_blocks, kmeans_gmms=kmeans_gmms, use_em=False, train=False)
        cm_km, acc_km = evaluate_performance(y_true_km, y_pred_km)
        print("K-means GMM Accuracy:", acc_km)
        plot_confusion_matrix(cm_km, title=f'K-means GMM Confusion Matrix (Threshold={threshold})')

        # Train EM GMM
        start_time_em = time.time()
        em_gmms = {}
        for d in range(NUM_DIGITS):
            em_gmms[d] = fit_gmm_em(digit_train_data[d], d, covariance_type='full')
        end_time_em = time.time()
        print(f"Time taken to train EM GMMs: {end_time_em - start_time_em:.2f} seconds")

        # Evaluate EM GMM
        y_true_em, y_pred_em = classify_all_tokens(test_blocks, em_gmms=em_gmms, use_em=True, train=False)
        cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
        print("EM GMM Accuracy:", acc_em)
        plot_confusion_matrix(cm_em, title=f'EM GMM Confusion Matrix (Threshold={threshold})')

        # Store results
        km_accuracies.append(acc_km)
        em_accuracies.append(acc_em)

    # Final comparison plot
    labels = [str(t) if t is not None else "None" for t in outlier_thresholds]
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
    plt.title('Accuracies for Different Outlier Removal Thresholds')
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

    # Print summary of all results
    print("\nSummary of Accuracies:")
    for t, km_acc, em_acc in zip(labels, km_accuracies, em_accuracies):
        print(f"Threshold={t}: K-Means Acc={km_acc:.4f}, EM Acc={em_acc:.4f}")

if __name__ == "__main__":
    main()

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

def select_cepstral_coefficients(blocks, coef_indices):
    """
    Given a list of blocks (utterances) and a list of coefficient indices,
    return a new set of blocks with only the selected coefficients.
    """
    new_blocks = []
    for block in blocks:

        new_block = [ [frame[i] for i in coef_indices] for frame in block ]
        new_blocks.append(new_block)
    return new_blocks



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

def em_gmm_likelihood(data, gmm: GaussianMixture):
    """Compute log-likelihood of data under an EM-fitted GMM."""
    return gmm.score(data)*len(data)

def classify_token(token_data, kmeans_gmms=None, em_gmms=None, use_em=False):
    """Classify a single token using max likelihood over digits."""
    if use_em:
        best_digit = None
        best_ll = -np.inf
        for d, model in em_gmms.items():
            ll = em_gmm_likelihood(token_data, model)
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
    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    # Normalize confusion matrix and convert to percentage
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
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')

    cepstral_subsets = {
        'All_13': list(range(13)),
        'First_6': list(range(6)),
        'Last_7': list(range(6,13)),
        'Subset_0_4_7_8': [0, 4, 7, 8]
    }

    results = []

    for subset_name, coef_indices in cepstral_subsets.items():
        print(f"Evaluating subset: {subset_name}, Coefficients: {coef_indices}")

        # Select the given subset of coefficients for training and test data
        train_blocks_subset = select_cepstral_coefficients(train_blocks, coef_indices)
        test_blocks_subset = select_cepstral_coefficients(test_blocks, coef_indices)

        # Time the training of K-means GMM
        start_time_kmeans = time.time()
        kmeans_gmms = {}
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks_subset, d, utterances_per_digit=UTTERANCES_PER_DIGIT_TRAIN)
            means, covs, weights = fit_kmeans_gmm(digit_blocks, d)
            kmeans_gmms[d] = (means, covs, weights)
        end_time_kmeans = time.time()
        print(f"Time taken to train K-means GMMs ({subset_name}): {end_time_kmeans - start_time_kmeans:.2f} seconds")

        # Confusion Matrix for K-means GMM
        y_true_km, y_pred_km = classify_all_tokens(test_blocks_subset, kmeans_gmms=kmeans_gmms, use_em=False, train=False)
        cm_km, acc_km = evaluate_performance(y_true_km, y_pred_km)
        print(f"K-means GMM Accuracy ({subset_name}):", acc_km)
        plot_confusion_matrix(cm_km, title=f'K-means GMM Confusion Matrix ({subset_name})')

        # Time the training of EM GMM
        start_time_em = time.time()
        em_gmms = {}
        for d in range(NUM_DIGITS):
            digit_blocks = extract_digit_blocks(train_blocks_subset, d, utterances_per_digit=UTTERANCES_PER_DIGIT_TRAIN)
            em_gmms[d] = fit_gmm_em(digit_blocks, d, covariance_type='full')
        end_time_em = time.time()
        print(f"Time taken to train EM GMMs ({subset_name}): {end_time_em - start_time_em:.2f} seconds")

        # Confusion Matrix for EM GMM
        y_true_em, y_pred_em = classify_all_tokens(test_blocks_subset, em_gmms=em_gmms, use_em=True, train=False)
        cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
        print(f"EM GMM Accuracy ({subset_name}):", acc_em)
        plot_confusion_matrix(cm_em, title=f'EM GMM Confusion Matrix ({subset_name})')

        # Store results for comparison
        results.append((subset_name, acc_km, acc_em))

    # Plot a comparison of accuracies across subsets
    subset_names = [r[0] for r in results]
    kmeans_accuracies = [r[1] for r in results]
    em_accuracies = [r[2] for r in results]

    x = np.arange(len(subset_names))
    width = 0.35

    plt.figure(figsize=(10,6))
    kmeans_bars = plt.bar(x - width/2, kmeans_accuracies, width, label='K-means GMM', alpha=0.7)
    em_bars = plt.bar(x + width/2, em_accuracies, width, label='EM GMM', alpha=0.7)

    # Annotate each bar with accuracy values
    for bar, acc in zip(kmeans_bars, kmeans_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

    for bar, acc in zip(em_bars, em_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

    plt.xticks(x, subset_names)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracies for Different Cepstral Coefficient Subsets')
    plt.legend()
    plt.ylim(0, 1)  # Accuracy in percentage
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

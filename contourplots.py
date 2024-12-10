import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import multivariate_normal



NUM_DIGITS = 10
NUM_CEPSTRA = 13
UTTERANCES_PER_DIGIT_TRAIN = 660
UTTERANCES_PER_DIGIT_TEST = 220
COVARIANCE_TYPE = 'tied'  # Tied covariance
DIGIT_TO_PLOT = 0  # Digit to visualize contours

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



def apply_variance_floor(cov, floor):
    if floor is not None and floor > 0:
        diag = np.diag(cov)
        diag = np.maximum(diag, floor)
        np.fill_diagonal(cov, diag)
    return cov



def kmeans_gmm_params(data, n_clusters, floor=None):
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

def fit_kmeans_gmm_tied(digit_blocks, digit, floor=None):
    if len(digit_blocks) == 0:
        return None, None, None
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    means, covs, weights = kmeans_gmm_params(data, n_clusters, floor)

    # Compute tied covariance by weighted average
    tied_cov = np.zeros_like(covs[0])
    for c in range(n_clusters):
        tied_cov += weights[c]*covs[c]

    # Replace all component covariances with tied covariance
    covs = np.array([tied_cov for _ in range(n_clusters)])
    return means, covs, weights



def fit_em_gmm_tied(digit_blocks, digit):
    data = np.vstack(digit_blocks)
    n_clusters = phoneme_clusters[digit]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', 
                          random_state=0, n_init=1, reg_covar=1e-6)
    gmm.fit(data)
    return gmm



def gmm_likelihood(data, means, covariances, weights):
    N = data.shape[0]
    K = len(weights)
    pdf_vals = np.zeros((N, K))
    for k in range(K):
        pdf_vals[:,k] = weights[k]*multivariate_normal.pdf(data, mean=means[k], cov=covariances[k])
    mixture = np.sum(pdf_vals, axis=1)
    return np.sum(np.log(mixture + 1e-12))

def classify_token_kmeans(token_data, kmeans_params):
    best_digit = None
    best_ll = -np.inf
    for d, params in kmeans_params.items():
        means, covs, weights = params
        ll = gmm_likelihood(token_data, means, covs, weights)
        if ll > best_ll:
            best_ll = ll
            best_digit = d
    return best_digit

def classify_all_tokens_kmeans(blocks, kmeans_params, train=False):
    predictions = []
    true_labels = []
    per_digit_utterances = UTTERANCES_PER_DIGIT_TRAIN if train else UTTERANCES_PER_DIGIT_TEST
    for digit in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(blocks, digit, per_digit_utterances)
        for token in digit_blocks:
            token_data = np.array(token)
            pred = classify_token_kmeans(token_data, kmeans_params)
            predictions.append(pred)
            true_labels.append(digit)
    return np.array(true_labels), np.array(predictions)


def classify_token_em(token_data, em_gmms):
    best_digit = None
    best_ll = -np.inf
    for d, model in em_gmms.items():
        ll = model.score(token_data)*len(token_data)
        if ll > best_ll:
            best_ll = ll
            best_digit = d
    return best_digit

def classify_all_tokens_em(blocks, em_gmms, train=False):
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

def plot_confusion_matrix(cm, acc, title='Confusion Matrix'):
    cm_prob = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8,6))
    df_cm = pd.DataFrame(cm_prob, index=range(NUM_DIGITS), columns=range(NUM_DIGITS))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f"{title}\nAccuracy: {acc:.4f}")
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    plt.show()



def plot_kmeans_gmm_contours(means, covariances, weights, data, digit, title_prefix):
    """
    Plot contour plots for selected MFCC pairs for K-Means GMM parameters.
    Since covariance is tied, each component shares the same covariance.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    mfcc_pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ['MFCC1 vs MFCC2', 'MFCC1 vs MFCC3', 'MFCC2 vs MFCC3']

    resolution = 200
    tied_cov = covariances[0]  # since tied, all are same

    for i, (mfcc_x, mfcc_y) in enumerate(mfcc_pairs):
        ax = axs[i]
        ax.scatter(data[:, mfcc_x], data[:, mfcc_y], s=5, color='gray', label="Data")

        x = np.linspace(data[:, mfcc_x].min(), data[:, mfcc_x].max(), resolution)
        y = np.linspace(data[:, mfcc_y].min(), data[:, mfcc_y].max(), resolution)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        # Plot contours for each component
        for m in means:
            mean_2d = m[[mfcc_x, mfcc_y]]
            cov_2d = tied_cov[[mfcc_x, mfcc_y]][:, [mfcc_x, mfcc_y]]
            rv = multivariate_normal(mean_2d, cov_2d)
            Z = rv.pdf(pos)
            ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)

        ax.set_title(f'{title_prefix}: Digit {digit}\n{titles[i]}')
        ax.set_xlabel(f'MFCC {mfcc_x+1}')
        ax.set_ylabel(f'MFCC {mfcc_y+1}')

    plt.tight_layout()
    plt.show()

def plot_em_gmm_contours(gmm, data, digit, title_prefix):
    """
    Plot contour plots for selected MFCC pairs for EM GMM with tied covariance.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    mfcc_pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ['MFCC1 vs MFCC2', 'MFCC1 vs MFCC3', 'MFCC2 vs MFCC3']

    resolution = 200
    tied_cov = gmm.covariances_  # single tied cov, shape (n_features, n_features)

    for i, (mfcc_x, mfcc_y) in enumerate(mfcc_pairs):
        ax = axs[i]
        ax.scatter(data[:, mfcc_x], data[:, mfcc_y], s=5, color='gray', label="Data")

        x = np.linspace(data[:, mfcc_x].min(), data[:, mfcc_x].max(), resolution)
        y = np.linspace(data[:, mfcc_y].min(), data[:, mfcc_y].max(), resolution)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        for c in range(gmm.n_components):
            mean_2d = gmm.means_[c, [mfcc_x, mfcc_y]]
            cov_2d = tied_cov[[mfcc_x, mfcc_y]][:, [mfcc_x, mfcc_y]]
            rv = multivariate_normal(mean_2d, cov_2d)
            Z = rv.pdf(pos)
            ax.contour(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)

        ax.set_title(f'{title_prefix}: Digit {digit}\n{titles[i]}')
        ax.set_xlabel(f'MFCC {mfcc_x+1}')
        ax.set_ylabel(f'MFCC {mfcc_y+1}')

    plt.tight_layout()
    plt.show()



def main():
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')

    # Fit K-Means GMM (No EM) Tied
    kmeans_params = {}
    for d in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
        means, covs, weights = fit_kmeans_gmm_tied(digit_blocks, d, floor=None)
        kmeans_params[d] = (means, covs, weights)

    # Classify test set using K-Means parameters
    y_true_km, y_pred_km = classify_all_tokens_kmeans(test_blocks, kmeans_params, train=False)
    cm_km, acc_km = evaluate_performance(y_true_km, y_pred_km)
    plot_confusion_matrix(cm_km, acc_km, title='K-Means Tied Covariance GMM (No EM) Confusion Matrix')

    # Fit EM GMM Tied
    em_gmms = {}
    for d in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(train_blocks, d, UTTERANCES_PER_DIGIT_TRAIN)
        em_gmms[d] = fit_em_gmm_tied(digit_blocks, d)

    # Classify test set using EM GMM
    y_true_em, y_pred_em = classify_all_tokens_em(test_blocks, em_gmms, train=False)
    cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
    plot_confusion_matrix(cm_em, acc_em, title='EM Tied Covariance GMM Confusion Matrix')

    print("K-Means Tied Covariance Accuracy:", acc_km)
    print("EM Tied Covariance Accuracy:", acc_em)

    # Plot Contours for a chosen digit using K-Means GMM params
    # Extract data for chosen digit
    digit_train_blocks = extract_digit_blocks(train_blocks, DIGIT_TO_PLOT, UTTERANCES_PER_DIGIT_TRAIN)
    digit_data = np.vstack(digit_train_blocks)
    k_means, k_covs, k_w = kmeans_params[DIGIT_TO_PLOT]
    plot_kmeans_gmm_contours(k_means, k_covs, k_w, digit_data, DIGIT_TO_PLOT, "K-Means Tied GMM (No EM)")

    # Plot Contours for the same digit using EM GMM
    em_model = em_gmms[DIGIT_TO_PLOT]
    plot_em_gmm_contours(em_model, digit_data, DIGIT_TO_PLOT, "EM Tied GMM")

if __name__ == "__main__":
    main()

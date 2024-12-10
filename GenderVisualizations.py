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


def get_speaker_gender(token_index, train=True):
    """
    Given a token index, determine gender.
    For the training set:
      - Each digit has 660 blocks: first 330 male, next 330 female
    For the test set:
      - Each digit has 220 blocks: first 110 male, next 110 female
    """
    if train:
        digit_token_pos = token_index % 660
        return 'M' if digit_token_pos < 330 else 'F'
    else:
        digit_token_pos = token_index % 220
        return 'M' if digit_token_pos < 110 else 'F'

def classify_all_tokens_with_gender(blocks, kmeans_gmms=None, em_gmms=None, use_em=False, train=False):
    predictions = []
    true_labels = []
    genders = []
    token_index = 0
    per_digit_utterances = UTTERANCES_PER_DIGIT_TRAIN if train else UTTERANCES_PER_DIGIT_TEST
    for digit in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(blocks, digit, utterances_per_digit=per_digit_utterances)
        for token in digit_blocks:
            token_data = np.array(token)
            pred = classify_token(token_data, kmeans_gmms, em_gmms, use_em=use_em)
            predictions.append(pred)
            true_labels.append(digit)
            genders.append(get_speaker_gender(token_index, train=train))
            token_index += 1
    return np.array(true_labels), np.array(predictions), np.array(genders)



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
    # Load data
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')
    
    # Time the training of K-means GMM
    start_time_kmeans = time.time()
    kmeans_gmms = {}
    for d in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(train_blocks, d, utterances_per_digit=UTTERANCES_PER_DIGIT_TRAIN)
        means, covs, weights = fit_kmeans_gmm(digit_blocks, d)
        kmeans_gmms[d] = (means, covs, weights)
    end_time_kmeans = time.time()
    print(f"Time taken to train K-means GMMs: {end_time_kmeans - start_time_kmeans:.2f} seconds")

    # Confusion Matrix for K-means GMM (no gender)
    y_true_km, y_pred_km = classify_all_tokens(test_blocks, kmeans_gmms=kmeans_gmms, use_em=False, train=False)
    cm_km, acc_km = evaluate_performance(y_true_km, y_pred_km)
    print("K-means GMM Accuracy:", acc_km)
    plot_confusion_matrix(cm_km, title='K-means GMM Confusion Matrix (No Gender)')

    # Confusion Matrices for K-means GMM by gender
    y_true_kmg, y_pred_kmg, genders_test = classify_all_tokens_with_gender(test_blocks, kmeans_gmms=kmeans_gmms, use_em=False, train=False)
    male_mask = (genders_test == 'M')
    female_mask = (genders_test == 'F')

    cm_km_male, acc_km_male = evaluate_performance(y_true_kmg[male_mask], y_pred_kmg[male_mask])
    print("K-means GMM Male Accuracy:", acc_km_male)
    plot_confusion_matrix(cm_km_male, title='K-means GMM Confusion Matrix (Male)')

    cm_km_female, acc_km_female = evaluate_performance(y_true_kmg[female_mask], y_pred_kmg[female_mask])
    print("K-means GMM Female Accuracy:", acc_km_female)
    plot_confusion_matrix(cm_km_female, title='K-means GMM Confusion Matrix (Female)')

    # Time the training of EM GMM
    start_time_em = time.time()
    em_gmms = {}
    for d in range(NUM_DIGITS):
        digit_blocks = extract_digit_blocks(train_blocks, d, utterances_per_digit=UTTERANCES_PER_DIGIT_TRAIN)
        em_gmms[d] = fit_gmm_em(digit_blocks, d, covariance_type='full')
    end_time_em = time.time()
    print(f"Time taken to train EM GMMs: {end_time_em - start_time_em:.2f} seconds")

    # Confusion Matrix for EM GMM (no gender)
    y_true_em, y_pred_em = classify_all_tokens(test_blocks, em_gmms=em_gmms, use_em=True, train=False)
    cm_em, acc_em = evaluate_performance(y_true_em, y_pred_em)
    print("EM GMM Accuracy:", acc_em)
    plot_confusion_matrix(cm_em, title='EM GMM Confusion Matrix (No Gender)')

    # Confusion Matrices for EM GMM by gender
    y_true_emg, y_pred_emg, genders_test_em = classify_all_tokens_with_gender(test_blocks, em_gmms=em_gmms, use_em=True, train=False)
    male_mask_em = (genders_test_em == 'M')
    female_mask_em = (genders_test_em == 'F')

    cm_em_male, acc_em_male = evaluate_performance(y_true_emg[male_mask_em], y_pred_emg[male_mask_em])
    print("EM GMM Male Accuracy:", acc_em_male)
    plot_confusion_matrix(cm_em_male, title='EM GMM Confusion Matrix (Male)')

    cm_em_female, acc_em_female = evaluate_performance(y_true_emg[female_mask_em], y_pred_emg[female_mask_em])
    print("EM GMM Female Accuracy:", acc_em_female)
    plot_confusion_matrix(cm_em_female, title='EM GMM Confusion Matrix (Female)')

    # Accuracies to plot
    labels = ['K-Means Overall', 'K-Means Male', 'K-Means Female', 
            'EM Overall', 'EM Male', 'EM Female']
    accuracies = [acc_km, acc_km_male, acc_km_female, 
                acc_em, acc_em_male, acc_em_female]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, alpha=0.7)

    # Annotate each bar with its value
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  #
            bar.get_height() + 0.01,           
            f"{acc:.4f}",                      
            ha='center', va='bottom', fontsize=10
        )

    # Configure the chart
    plt.ylim(0, 1)  
    plt.title('Classification Accuracies for K-Means and EM GMM')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

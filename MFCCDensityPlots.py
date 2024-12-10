import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




NUM_DIGITS = 10
UTTERANCES_PER_DIGIT = 660
NUM_CEPSTRA = 13  

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

def extract_digit_blocks(blocks, digit, utterances_per_digit=UTTERANCES_PER_DIGIT):
    """Extract blocks for a given digit."""
    start_index = digit * utterances_per_digit
    end_index = start_index + utterances_per_digit
    return blocks[start_index:end_index]



def plot_digit_distribution(data, digit):
    """Overlay density plots for each coefficient of a given digit."""
    digit_data = extract_digit_blocks(data, digit)
    all_data = np.vstack(digit_data)
    
    fig, axes = plt.subplots(4,4, figsize=(12,12))
    axes = axes.ravel()
    for i in range(NUM_CEPSTRA):
        sns.kdeplot(all_data[:, i], ax=axes[i], shade=True)
        axes[i].set_title(f'Coeff {i}')
    # plt.suptitle(f'Aggregate Distribution of MFCC Coeffs for Digit {digit}')
    plt.tight_layout()
    plt.show()
def plot_mfccs_for_digit(blocks, digit):
    """Plot the first 3 MFCCs for a specific digit."""
    first_utterance = blocks[0]  # Get the first utterance

    # Initialize lists for the first three MFCCs
    mfcc1 = []
    mfcc2 = []
    mfcc3 = []
    
    # Populate the lists
    for frame in first_utterance:
        mfcc1.append(frame[0])
        mfcc2.append(frame[1])
        mfcc3.append(frame[2])

    # Create a figure for plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mfcc1, label='MFCC 1')
    plt.plot(mfcc2, label='MFCC 2')
    plt.plot(mfcc3, label='MFCC 3')
    
    plt.title(f'MFCCs for Digit {digit}')
    plt.xlabel('Analysis Window (Frame)')
    plt.ylabel('MFCC Value')
    plt.legend()
    plt.show()


def main():
    # Load data
    train_blocks = read_mfcc_file('Train_Arabic_Digit.txt')
    test_blocks = read_mfcc_file('Test_Arabic_Digit.txt')
    # Plot distribution for digit 0
    plot_digit_distribution(train_blocks, 0)
    
    

if __name__ == "__main__":
    main()

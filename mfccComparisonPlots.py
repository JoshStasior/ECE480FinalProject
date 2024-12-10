import matplotlib.pyplot as plt

def read_mfcc_file(file_path):
    """Read the file and organize the data into blocks (utterances)."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    blocks = []
    current_block = []
    for line in lines:
        line = line.strip()
        if line == '':  # Blank line indicates a new block (utterance)
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            string_values = line.split()
            # Convert string values to floats\
            float_values = [float(x) for x in string_values]
            # Add the list of float values to the current block
            current_block.append(float_values)

    if current_block:  # Add the last block
        blocks.append(current_block)

    return blocks

def extract_digit_blocks(blocks, digit):
    """Extract blocks corresponding to the given digit (0-9)."""
    start_idx = digit * 660  # Each digit has 660 blocks
    end_idx = start_idx + 660
    return blocks[start_idx:end_idx]

def plot_mfcc_scatter_for_digit(blocks, digit):
    """Plot scatter plots for the first 3 MFCC coefficients for the first utterance of the digit."""
    first_utterance = blocks[0]  # Select the first utterance (first block)
    
     # Initialize lists for the first three MFCCs
    mfcc1 = []
    mfcc2 = []
    mfcc3 = []
    
    # Populate the lists
    for frame in first_utterance:
        mfcc1.append(frame[0])
        mfcc2.append(frame[1])
        mfcc3.append(frame[2])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # MFCC1 vs MFCC2
    axs[0].scatter(mfcc1, mfcc2, color='blue')
    axs[0].set_title(f'Digit {digit}: MFCC1 vs MFCC2')
    axs[0].set_xlabel('MFCC1')
    axs[0].set_ylabel('MFCC2')
    
    # MFCC1 vs MFCC3
    axs[1].scatter(mfcc1, mfcc3, color='green')
    axs[1].set_title(f'Digit {digit}: MFCC1 vs MFCC3')
    axs[1].set_xlabel('MFCC1')
    axs[1].set_ylabel('MFCC3')
    
    # MFCC2 vs MFCC3
    axs[2].scatter(mfcc2, mfcc3, color='red')
    axs[2].set_title(f'Digit {digit}: MFCC2 vs MFCC3')
    axs[2].set_xlabel('MFCC2')
    axs[2].set_ylabel('MFCC3')
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'Train_Arabic_Digit.txt'  # Path to the data file
    blocks = read_mfcc_file(file_path)

    # Plot scatter plots for digits 0 to 9
    for digit in range(10):
        digit_blocks = extract_digit_blocks(blocks, digit)
        plot_mfcc_scatter_for_digit(digit_blocks, digit)

if __name__ == "__main__":
    main()

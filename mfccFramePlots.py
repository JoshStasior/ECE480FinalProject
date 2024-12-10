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
    """Extract blocks corresponding to a specific digit (0-9)."""
    start_index = digit * 660
    end_index = start_index + 660
    return blocks[start_index:end_index]

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
    file_path = 'Train_Arabic_Digit.txt'  # Path to the data file
    blocks = read_mfcc_file(file_path)

    for digit in range(10):
        digit_blocks = extract_digit_blocks(blocks, digit)
        plot_mfccs_for_digit(digit_blocks, digit)

if __name__ == "__main__":
    main()

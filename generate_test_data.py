import pandas as pd
import numpy as np
import random
import os

def generate_test_mutation_data(sequence_file, output_file, top_k=5):
    """
    Generate test mutation data for a protein sequence.
    
    Args:
        sequence_file: Path to the file containing the protein sequence
        output_file: Path to save the generated mutation data
        top_k: Number of mutations to generate per position
    """
    # Read the protein sequence
    with open(sequence_file, 'r') as f:
        sequence = f.read().strip()
    
    print(f"Generating test mutation data for sequence of length {len(sequence)}...")
    
    # Define amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Generate mutation data
    data = []
    
    for pos in range(1, len(sequence) + 1):
        original_aa = sequence[pos - 1]
        
        # Generate random mutations for this position
        possible_mutations = [aa for aa in amino_acids if aa != original_aa]
        selected_mutations = random.sample(possible_mutations, min(top_k, len(possible_mutations)))
        
        # Generate probabilities (higher for positions near the middle of the sequence)
        position_factor = 1.0 - abs(pos - len(sequence)/2) / (len(sequence)/2) * 0.5
        
        for i, mutated_aa in enumerate(selected_mutations):
            # Generate a probability (higher for first mutations in each position)
            rank_factor = 1.0 - i * 0.15
            probability = min(0.95, max(0.05, np.random.beta(2, 2) * position_factor * rank_factor))
            
            # Create mutation string
            mutation = f"{original_aa}{pos}{mutated_aa}"
            
            # Add to data
            data.append({
                'position': pos,
                'original': original_aa,
                'mutation': mutation,
                'mutated_to': mutated_aa,
                'probability': probability,
                'log_probability': np.log(probability)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by probability
    df = df.sort_values('probability', ascending=False)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} test mutations and saved to {output_file}")

if __name__ == "__main__":
    # Check if the sequence file exists
    sequence_file = "fhase_sequence.txt"
    if not os.path.exists(sequence_file):
        print(f"Error: Sequence file {sequence_file} not found")
        exit(1)
    
    # Generate test data
    generate_test_mutation_data(sequence_file, "fhase_mutations_test.csv", top_k=5)

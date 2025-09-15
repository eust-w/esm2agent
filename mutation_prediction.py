import torch
import esm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import os
import argparse

class ESMMutationPredictor:
    """
    A class for predicting protein mutations using the ESM-2 language model.
    This predictor identifies mutations that are likely to maintain protein function
    while potentially improving desired properties.
    """
    
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Initialize the mutation predictor with a specific ESM-2 model.
        
        Args:
            model_name: Name of the ESM-2 model to use
        """
        print(f"Loading {model_name} model...")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the ESM-2 model
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get the batch converter
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Map from token indices to amino acids
        self.index_to_aa = {i: a for i, a in enumerate(self.alphabet.all_toks) if a != '<cls>' and a != '<pad>' and a != '<eos>' and a != '<unk>' and a != '<mask>'}
        
        print("Model loaded successfully!")
    
    def predict_mutations(self, 
                         sequence: str, 
                         positions: Optional[List[int]] = None,
                         exclude_original: bool = True,
                         top_k: int = 5) -> pd.DataFrame:
        """
        Predict mutations for the given protein sequence.
        
        Args:
            sequence: Amino acid sequence of the protein
            positions: List of positions to consider for mutation (1-indexed)
                       If None, all positions will be considered
            exclude_original: Whether to exclude the original amino acid from predictions
            top_k: Number of top mutations to return for each position
            
        Returns:
            DataFrame with mutation predictions and scores
        """
        # If no positions are specified, consider all positions
        if positions is None:
            positions = list(range(1, len(sequence) + 1))
        
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 20  # Adjust based on available memory
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i+batch_size]
            batch_results = self._predict_batch(sequence, batch_positions, exclude_original, top_k)
            results.extend(batch_results)
            
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        return df
    
    def _predict_batch(self, 
                      sequence: str, 
                      positions: List[int],
                      exclude_original: bool,
                      top_k: int) -> List[Dict]:
        """
        Predict mutations for a batch of positions in the sequence.
        """
        results = []
        
        # Create masked sequences for each position
        masked_sequences = []
        for pos in positions:
            # Convert to 0-indexed
            idx = pos - 1
            
            # Create a masked sequence
            masked_seq = sequence[:idx] + "<mask>" + sequence[idx+1:]
            masked_sequences.append((f"pos_{pos}", masked_seq))
        
        # Convert sequences to tokens
        batch_labels, batch_strs, batch_tokens = self.batch_converter(masked_sequences)
        batch_tokens = batch_tokens.to(self.device)
        
        # Get predictions from the model
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        token_probs = torch.nn.functional.softmax(results["logits"], dim=-1)
        
        # Process each position
        output_results = []
        for i, pos in enumerate(positions):
            idx = pos - 1  # Convert to 0-indexed
            original_aa = sequence[idx]
            
            # Find the location of the mask in the sequence
            mask_position = batch_strs[i].find("<mask>")
            
            # Get probabilities for the masked position
            probs = token_probs[i, mask_position+1].cpu().numpy()
            
            # Get amino acid probabilities (exclude special tokens)
            aa_probs = {}
            for token_idx, aa in self.index_to_aa.items():
                if len(aa) == 1:  # Only consider single amino acids
                    aa_probs[aa] = probs[token_idx]
            
            # Sort amino acids by probability
            sorted_aas = sorted(aa_probs.items(), key=lambda x: x[1], reverse=True)
            
            # Filter out the original amino acid if requested
            if exclude_original:
                sorted_aas = [(aa, prob) for aa, prob in sorted_aas if aa != original_aa]
            
            # Get the top-k mutations
            top_mutations = sorted_aas[:top_k]
            
            # Add results for each mutation
            for aa, prob in top_mutations:
                mutation = f"{original_aa}{pos}{aa}"
                output_results.append({
                    "position": pos,
                    "original": original_aa,
                    "mutation": mutation,
                    "mutated_to": aa,
                    "probability": float(prob),
                    "log_probability": float(np.log(prob))
                })
        
        return output_results
    
    def visualize_mutations(self, 
                           df: pd.DataFrame, 
                           output_file: Optional[str] = None,
                           top_n_positions: int = 20,
                           split_charts: bool = True,
                           max_positions_per_chart: int = 30):
        """
        Visualize the mutation predictions.
        
        Args:
            df: DataFrame with mutation predictions
            output_file: Path to save the visualization
            top_n_positions: Number of top positions to visualize
            split_charts: Whether to split large charts into multiple files
            max_positions_per_chart: Maximum number of positions to show in a single chart
        """
        # Group by position and get the sum of probabilities for top mutations
        position_scores = df.groupby('position')['probability'].sum().reset_index()
        position_scores = position_scores.sort_values('probability', ascending=False).head(top_n_positions)
        
        # Get the top positions
        top_positions = position_scores['position'].tolist()
        
        # Add original amino acid information to position_scores
        position_to_original = df[['position', 'original']].drop_duplicates().set_index('position')['original']
        position_scores['original'] = position_scores['position'].map(position_to_original)
        
        # Filter the dataframe to only include the top positions
        df_filtered = df[df['position'].isin(top_positions)]
        
        # Create a pivot table for the heatmap
        pivot_data = df_filtered.pivot_table(
            index='position', 
            columns='mutated_to', 
            values='probability',
            fill_value=0
        )
        
        # Sort the positions by their total probability
        pivot_data = pivot_data.loc[position_scores['position']]
        
        # Create better row labels with original amino acid
        row_labels = [f"{int(pos)} ({orig})" for pos, orig in 
                     zip(position_scores['position'], position_scores['original'])]
        
        # If we have too many positions, split into multiple charts
        if split_charts and top_n_positions > max_positions_per_chart:
            num_charts = (top_n_positions + max_positions_per_chart - 1) // max_positions_per_chart
            charts_data = []
            
            for i in range(num_charts):
                start_idx = i * max_positions_per_chart
                end_idx = min((i + 1) * max_positions_per_chart, top_n_positions)
                
                # Get positions for this chart
                chart_positions = top_positions[start_idx:end_idx]
                chart_labels = row_labels[start_idx:end_idx]
                
                # Filter data for this chart
                chart_data = pivot_data.loc[chart_positions]
                
                # Create the plot
                plt.figure(figsize=(20, min(15, 0.4 * len(chart_positions) + 5)))
                ax = sns.heatmap(chart_data, cmap='viridis', annot=True, fmt='.3f', 
                              linewidths=0.5, annot_kws={"size": 9})
                
                # Set better labels
                ax.set_yticklabels(chart_labels, rotation=0)
                
                plt.title(f'Top Mutation Probabilities by Position (Part {i+1}/{num_charts})')
                plt.xlabel('Mutated Amino Acid')
                plt.ylabel('Position (Original AA)')
                plt.tight_layout()
                
                # Save this chart
                if output_file:
                    base_name, ext = os.path.splitext(output_file)
                    chart_file = f"{base_name}_part{i+1}{ext}"
                    plt.savefig(chart_file, bbox_inches='tight', dpi=300)
                    print(f"Visualization part {i+1} saved to {chart_file}")
                    charts_data.append(chart_file)
                else:
                    plt.show()
                
                plt.close()
        else:
            # Create a single chart for all positions
            plt.figure(figsize=(20, min(20, 0.4 * len(top_positions) + 5)))
            ax = sns.heatmap(pivot_data, cmap='viridis', annot=True, fmt='.3f', 
                          linewidths=0.5, annot_kws={"size": 9})
            
            # Set better labels
            ax.set_yticklabels(row_labels, rotation=0)
            
            plt.title('Top Mutation Probabilities by Position')
            plt.xlabel('Mutated Amino Acid')
            plt.ylabel('Position (Original AA)')
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                print(f"Visualization saved to {output_file}")
            else:
                plt.show()
                
            plt.close()
        
        return pivot_data

def main():
    parser = argparse.ArgumentParser(description='Predict protein mutations using ESM-2')
    parser.add_argument('--sequence', type=str, required=True, help='Protein sequence')
    parser.add_argument('--positions', type=str, help='Comma-separated list of positions to mutate (1-indexed)')
    parser.add_argument('--output', type=str, default='mutations.csv', help='Output CSV file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--vis_output', type=str, default='mutation_heatmap.png', help='Output visualization file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top mutations per position')
    
    args = parser.parse_args()
    
    # Parse positions if provided
    positions = None
    if args.positions:
        positions = [int(pos) for pos in args.positions.split(',')]
    
    # Initialize the predictor
    predictor = ESMMutationPredictor()
    
    # Predict mutations
    results = predictor.predict_mutations(
        sequence=args.sequence,
        positions=positions,
        top_k=args.top_k
    )
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"Mutation predictions saved to {args.output}")
    
    # Visualize if requested
    if args.visualize:
        predictor.visualize_mutations(results, args.vis_output)

if __name__ == "__main__":
    main()

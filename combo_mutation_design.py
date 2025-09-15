import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import argparse
from typing import List, Dict, Tuple, Optional, Set
from mutation_prediction import ESMMutationPredictor

class ComboMutationDesigner:
    """
    A class for designing and evaluating combination mutations based on single mutation predictions.
    This tool helps identify potential synergistic effects between multiple mutations.
    """
    
    def __init__(self, mutations_csv: str):
        """
        Initialize the combo mutation designer with mutation prediction data.
        
        Args:
            mutations_csv: Path to the CSV file containing mutation predictions
        """
        print(f"Loading mutation data from {mutations_csv}...")
        self.mutations_df = pd.read_csv(mutations_csv)
        
        # Extract unique positions and their original amino acids
        self.positions = self.mutations_df[['position', 'original']].drop_duplicates()
        self.positions_dict = dict(zip(self.positions['position'], self.positions['original']))
        
        # Create a dictionary for quick lookup of mutation probabilities
        self.mutation_probs = {}
        for _, row in self.mutations_df.iterrows():
            mutation_key = f"{row['original']}{row['position']}{row['mutated_to']}"
            self.mutation_probs[mutation_key] = row['probability']
        
        print(f"Loaded data for {len(self.positions)} positions and {len(self.mutation_probs)} mutations")
    
    def get_top_mutations_per_position(self, top_k: int = 1) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get the top k mutations for each position.
        
        Args:
            top_k: Number of top mutations to retrieve per position
            
        Returns:
            Dictionary mapping position to list of (mutation, probability) tuples
        """
        top_mutations = {}
        
        for pos in self.positions['position']:
            # Filter mutations for this position
            pos_mutations = self.mutations_df[self.mutations_df['position'] == pos]
            
            # Sort by probability and get top k
            top_k_mutations = pos_mutations.sort_values('probability', ascending=False).head(top_k)
            
            # Store as list of (mutation, probability) tuples
            top_mutations[pos] = [(f"{row['original']}{row['position']}{row['mutated_to']}", row['probability']) 
                                 for _, row in top_k_mutations.iterrows()]
        
        return top_mutations
    
    def generate_mutation_combinations(self, 
                                      top_positions: int = 20, 
                                      mutations_per_position: int = 1,
                                      combo_size: int = 3) -> List[List[str]]:
        """
        Generate combinations of mutations from top positions.
        
        Args:
            top_positions: Number of top positions to consider
            mutations_per_position: Number of mutations to consider per position
            combo_size: Size of mutation combinations
            
        Returns:
            List of mutation combinations, where each combination is a list of mutation strings
        """
        # Get top positions by summing probabilities of all mutations at each position
        position_scores = self.mutations_df.groupby('position')['probability'].sum().reset_index()
        top_pos = position_scores.sort_values('probability', ascending=False).head(top_positions)['position'].tolist()
        
        # Get top mutations for each position
        top_mutations = self.get_top_mutations_per_position(mutations_per_position)
        
        # Filter to only include top positions
        filtered_mutations = {pos: muts for pos, muts in top_mutations.items() if pos in top_pos}
        
        # Generate all possible combinations of positions
        position_combos = list(itertools.combinations(filtered_mutations.keys(), combo_size))
        
        # Generate mutation combinations
        mutation_combos = []
        for pos_combo in position_combos:
            # Get all mutations for each position in this combo
            mutation_options = [filtered_mutations[pos] for pos in pos_combo]
            
            # Generate all combinations of mutations (one from each position)
            for mutation_combo in itertools.product(*[[mut[0] for mut in pos_muts] for pos_muts in mutation_options]):
                mutation_combos.append(list(mutation_combo))
        
        print(f"Generated {len(mutation_combos)} mutation combinations of size {combo_size}")
        return mutation_combos
    
    def calculate_synergy_score(self, 
                              mutation_combo: List[str], 
                              synergy_model: str = 'multiplicative') -> Dict:
        """
        Calculate a synergy score for a combination of mutations.
        
        Args:
            mutation_combo: List of mutation strings (e.g., ['M1L', 'A5E'])
            synergy_model: Model for calculating synergy ('multiplicative', 'additive', or 'geometric')
            
        Returns:
            Dictionary with synergy scores and related information
        """
        # Extract individual mutation probabilities
        individual_probs = [self.mutation_probs.get(mut, 0) for mut in mutation_combo]
        
        if synergy_model == 'multiplicative':
            # Multiplicative model: P(A,B,C) = P(A) * P(B) * P(C)
            expected_score = np.prod(individual_probs)
        elif synergy_model == 'additive':
            # Additive model: P(A,B,C) = P(A) + P(B) + P(C) - overlap
            # We approximate the overlap by taking the average probability
            expected_score = sum(individual_probs) - (sum(individual_probs) * (len(individual_probs) - 1)) / len(individual_probs)
            expected_score = max(0, min(1, expected_score))  # Ensure it's between 0 and 1
        elif synergy_model == 'geometric':
            # Geometric mean model: P(A,B,C) = (P(A) * P(B) * P(C))^(1/3)
            expected_score = np.power(np.prod(individual_probs), 1/len(individual_probs))
        else:
            raise ValueError(f"Unknown synergy model: {synergy_model}")
        
        # Calculate a "synergy factor" - how much better the combination might be
        # This is a heuristic based on:
        # 1. How similar the mutations are (more diverse = potentially more synergistic)
        # 2. The average probability of the mutations
        
        # Extract positions and amino acid changes
        positions = [int(''.join(filter(str.isdigit, mut))) for mut in mutation_combo]
        orig_aas = [mut[0] for mut in mutation_combo]
        new_aas = [mut[-1] for mut in mutation_combo]
        
        # Calculate position diversity (normalized distance between positions)
        pos_range = max(positions) - min(positions)
        sequence_length = max(self.positions['position'])  # Approximate sequence length
        position_diversity = min(1.0, pos_range / (sequence_length * 0.3))  # Normalize, cap at 1.0
        
        # Calculate amino acid diversity (how different the changes are)
        # Simple metric: count unique amino acids
        orig_aa_diversity = len(set(orig_aas)) / len(orig_aas)
        new_aa_diversity = len(set(new_aas)) / len(new_aas)
        aa_diversity = (orig_aa_diversity + new_aa_diversity) / 2
        
        # Calculate average probability
        avg_prob = sum(individual_probs) / len(individual_probs)
        
        # Calculate synergy factor (heuristic)
        # Higher for diverse positions, diverse amino acids, and high individual probabilities
        synergy_factor = (0.4 * position_diversity + 0.3 * aa_diversity + 0.3 * avg_prob) * 0.5
        
        # Apply synergy factor to expected score
        synergistic_score = expected_score * (1 + synergy_factor)
        synergistic_score = min(1.0, synergistic_score)  # Cap at 1.0
        
        return {
            'mutations': mutation_combo,
            'individual_probabilities': individual_probs,
            'expected_score': expected_score,
            'synergy_factor': synergy_factor,
            'synergistic_score': synergistic_score,
            'position_diversity': position_diversity,
            'aa_diversity': aa_diversity,
            'positions': positions
        }
    
    def evaluate_combinations(self, 
                             combos: List[List[str]], 
                             synergy_model: str = 'multiplicative') -> pd.DataFrame:
        """
        Evaluate a list of mutation combinations and return results as a DataFrame.
        
        Args:
            combos: List of mutation combinations
            synergy_model: Model for calculating synergy
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for combo in combos:
            result = self.calculate_synergy_score(combo, synergy_model)
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by synergistic score
        results_df = results_df.sort_values('synergistic_score', ascending=False)
        
        return results_df
    
    def visualize_top_combinations(self, 
                                  results_df: pd.DataFrame, 
                                  output_file: Optional[str] = None,
                                  top_n: int = 20) -> None:
        """
        Visualize the top mutation combinations.
        
        Args:
            results_df: DataFrame with evaluation results
            output_file: Path to save the visualization
            top_n: Number of top combinations to visualize
        """
        # Get top combinations
        top_combos = results_df.head(top_n)
        
        # Create a more readable mutation representation
        top_combos['mutation_combo'] = top_combos['mutations'].apply(lambda x: ' + '.join(x))
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot synergistic scores
        ax = sns.barplot(x='synergistic_score', y='mutation_combo', data=top_combos, color='skyblue')
        
        # Add expected scores as smaller bars
        ax2 = sns.barplot(x='expected_score', y='mutation_combo', data=top_combos, color='navy', alpha=0.5)
        
        # Add synergy factor as text
        for i, row in enumerate(top_combos.iterrows()):
            plt.text(row[1]['expected_score'] + 0.02, i, f"+{row[1]['synergy_factor']:.2f}", 
                    va='center', color='darkgreen', fontweight='bold')
        
        plt.xlabel('Score')
        plt.ylabel('Mutation Combination')
        plt.title('Top Mutation Combinations with Synergy Potential')
        plt.legend(['Synergistic Score', 'Expected Score'])
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_diverse_combinations(self, 
                                    results_df: pd.DataFrame, 
                                    n_combinations: int = 10,
                                    diversity_weight: float = 0.5) -> pd.DataFrame:
        """
        Generate a diverse set of high-scoring mutation combinations.
        
        Args:
            results_df: DataFrame with evaluation results
            n_combinations: Number of diverse combinations to select
            diversity_weight: Weight for diversity vs. score (0-1)
            
        Returns:
            DataFrame with selected diverse combinations
        """
        if len(results_df) <= n_combinations:
            return results_df
        
        # Start with the highest scoring combination
        selected_indices = [0]
        selected_positions = set(results_df.iloc[0]['positions'])
        
        while len(selected_indices) < n_combinations and len(selected_indices) < len(results_df):
            max_score = -1
            best_idx = -1
            
            # Evaluate remaining combinations
            for i in range(len(results_df)):
                if i in selected_indices:
                    continue
                
                # Calculate position overlap with already selected combinations
                positions = set(results_df.iloc[i]['positions'])
                overlap = len(positions.intersection(selected_positions))
                diversity_score = 1 - (overlap / len(positions))
                
                # Calculate combined score (weighted average of synergistic score and diversity)
                synergistic_score = results_df.iloc[i]['synergistic_score']
                combined_score = (1 - diversity_weight) * synergistic_score + diversity_weight * diversity_score
                
                if combined_score > max_score:
                    max_score = combined_score
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_positions.update(results_df.iloc[best_idx]['positions'])
            else:
                break
        
        # Return selected combinations
        return results_df.iloc[selected_indices].reset_index(drop=True)
    
    def save_results(self, 
                    results_df: pd.DataFrame, 
                    output_csv: str,
                    top_n: int = 100) -> None:
        """
        Save evaluation results to a CSV file.
        
        Args:
            results_df: DataFrame with evaluation results
            output_csv: Path to save the CSV file
            top_n: Number of top combinations to save
        """
        # Prepare data for saving
        save_df = results_df.head(top_n).copy()
        
        # Convert list columns to strings
        save_df['mutations'] = save_df['mutations'].apply(lambda x: ' + '.join(x))
        save_df['individual_probabilities'] = save_df['individual_probabilities'].apply(lambda x: ', '.join([f"{p:.4f}" for p in x]))
        save_df['positions'] = save_df['positions'].apply(lambda x: ', '.join([str(p) for p in x]))
        
        # Save to CSV
        save_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='Design and evaluate combination mutations')
    parser.add_argument('--mutations_csv', type=str, default='fhase_mutations_full.csv',
                        help='Path to the CSV file containing mutation predictions')
    parser.add_argument('--output_prefix', type=str, default='fhase_combo_mutations',
                        help='Prefix for output files')
    parser.add_argument('--top_positions', type=int, default=30,
                        help='Number of top positions to consider')
    parser.add_argument('--mutations_per_position', type=int, default=2,
                        help='Number of mutations to consider per position')
    parser.add_argument('--combo_size', type=int, default=3,
                        help='Size of mutation combinations')
    parser.add_argument('--synergy_model', type=str, default='multiplicative',
                        choices=['multiplicative', 'additive', 'geometric'],
                        help='Model for calculating synergy')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top combinations to save')
    parser.add_argument('--diversity', action='store_true',
                        help='Generate diverse combinations')
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                        help='Weight for diversity vs. score (0-1)')
    
    args = parser.parse_args()
    
    # Initialize designer
    designer = ComboMutationDesigner(args.mutations_csv)
    
    # Generate combinations
    combos = designer.generate_mutation_combinations(
        top_positions=args.top_positions,
        mutations_per_position=args.mutations_per_position,
        combo_size=args.combo_size
    )
    
    # Evaluate combinations
    results = designer.evaluate_combinations(combos, args.synergy_model)
    
    # Generate diverse combinations if requested
    if args.diversity:
        diverse_results = designer.generate_diverse_combinations(
            results,
            n_combinations=args.top_n,
            diversity_weight=args.diversity_weight
        )
        
        # Save diverse results
        designer.save_results(
            diverse_results,
            f"{args.output_prefix}_diverse.csv",
            top_n=len(diverse_results)
        )
        
        # Visualize diverse combinations
        designer.visualize_top_combinations(
            diverse_results,
            f"{args.output_prefix}_diverse.png",
            top_n=min(20, len(diverse_results))
        )
    
    # Save all results
    designer.save_results(results, f"{args.output_prefix}.csv", top_n=args.top_n)
    
    # Visualize top combinations
    designer.visualize_top_combinations(results, f"{args.output_prefix}.png", top_n=20)
    
    print(f"Completed combination mutation design for {len(combos)} combinations")

if __name__ == "__main__":
    main()

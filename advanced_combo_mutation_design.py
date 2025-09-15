import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import argparse
import warnings
from typing import List, Dict, Tuple, Optional, Set, Union
from biopandas.pdb import PandasPdb
import networkx as nx
from scipy.spatial.distance import pdist, squareform

class AdvancedComboMutationDesigner:
    """
    An advanced class for designing and evaluating combination mutations based on:
    1. Single mutation predictions
    2. Protein structure information (spatial distances)
    3. Functional domain information
    4. Support for larger combination sizes
    """
    
    def __init__(self, 
                mutations_csv: str,
                pdb_file: Optional[str] = None,
                domains_file: Optional[str] = None):
        """
        Initialize the advanced combo mutation designer.
        
        Args:
            mutations_csv: Path to the CSV file containing mutation predictions
            pdb_file: Optional path to a PDB file with protein structure
            domains_file: Optional path to a CSV file with domain information
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
        
        # Load structure information if provided
        self.pdb_data = None
        self.distance_matrix = None
        self.ca_coords = None
        if pdb_file and os.path.exists(pdb_file):
            self._load_structure(pdb_file)
        
        # Load domain information if provided
        self.domains = None
        if domains_file and os.path.exists(domains_file):
            self._load_domains(domains_file)
        else:
            # Create default domain (whole protein)
            self.domains = {'whole_protein': list(self.positions_dict.keys())}
    
    def _load_structure(self, pdb_file: str) -> None:
        """
        Load protein structure from PDB file and calculate distance matrix.
        
        Args:
            pdb_file: Path to the PDB file
        """
        print(f"Loading protein structure from {pdb_file}...")
        try:
            # Read PDB file
            self.pdb_data = PandasPdb().read_pdb(pdb_file)
            
            # Extract CA coordinates (alpha carbons)
            ca_atoms = self.pdb_data.df['ATOM'][self.pdb_data.df['ATOM']['atom_name'] == 'CA']
            
            # Create mapping from residue number to index
            self.residue_to_idx = {}
            for i, res_num in enumerate(ca_atoms['residue_number']):
                self.residue_to_idx[res_num] = i
            
            # Extract coordinates
            self.ca_coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].values
            
            # Calculate distance matrix (in Angstroms)
            self.distance_matrix = squareform(pdist(self.ca_coords))
            
            print(f"Loaded structure with {len(ca_atoms)} residues")
            
            # Create a contact map (residues within 10 Angstroms are considered in contact)
            self.contact_map = self.distance_matrix < 10.0
            
            # Create a graph representation for network analysis
            self.structure_graph = nx.Graph()
            for i in range(len(self.ca_coords)):
                self.structure_graph.add_node(ca_atoms['residue_number'].iloc[i])
            
            for i in range(len(self.ca_coords)):
                for j in range(i+1, len(self.ca_coords)):
                    if self.contact_map[i, j]:
                        self.structure_graph.add_edge(
                            ca_atoms['residue_number'].iloc[i],
                            ca_atoms['residue_number'].iloc[j],
                            weight=self.distance_matrix[i, j]
                        )
        
        except Exception as e:
            warnings.warn(f"Error loading structure: {e}")
            self.pdb_data = None
            self.distance_matrix = None
            self.ca_coords = None
    
    def _load_domains(self, domains_file: str) -> None:
        """
        Load domain information from a CSV file.
        Expected format: domain_name,start_position,end_position
        
        Args:
            domains_file: Path to the domains CSV file
        """
        print(f"Loading domain information from {domains_file}...")
        try:
            domains_df = pd.read_csv(domains_file)
            self.domains = {}
            
            for _, row in domains_df.iterrows():
                domain_name = row['domain_name']
                start_pos = int(row['start_position'])
                end_pos = int(row['end_position'])
                
                # Create list of positions in this domain
                domain_positions = list(range(start_pos, end_pos + 1))
                
                # Filter to only include positions that exist in our data
                domain_positions = [pos for pos in domain_positions if pos in self.positions_dict]
                
                if domain_positions:
                    self.domains[domain_name] = domain_positions
            
            print(f"Loaded {len(self.domains)} domains")
        
        except Exception as e:
            warnings.warn(f"Error loading domains: {e}")
            # Create default domain (whole protein)
            self.domains = {'whole_protein': list(self.positions_dict.keys())}
    
    def get_spatial_distance(self, pos1: int, pos2: int) -> Optional[float]:
        """
        Get the spatial distance between two residues in Angstroms.
        
        Args:
            pos1: First residue position
            pos2: Second residue position
            
        Returns:
            Distance in Angstroms, or None if structure information is not available
        """
        if self.distance_matrix is None or self.residue_to_idx is None:
            return None
        
        if pos1 not in self.residue_to_idx or pos2 not in self.residue_to_idx:
            return None
        
        idx1 = self.residue_to_idx[pos1]
        idx2 = self.residue_to_idx[pos2]
        
        return self.distance_matrix[idx1, idx2]
    
    def get_domain_for_position(self, position: int) -> List[str]:
        """
        Get the domain(s) that a position belongs to.
        
        Args:
            position: Residue position
            
        Returns:
            List of domain names
        """
        if self.domains is None:
            return ['whole_protein']
        
        return [domain for domain, positions in self.domains.items() if position in positions]
    
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
                                      combo_size: int = 3,
                                      domain_focus: Optional[str] = None,
                                      max_combinations: int = 100000) -> List[List[str]]:
        """
        Generate combinations of mutations from top positions.
        
        Args:
            top_positions: Number of top positions to consider
            mutations_per_position: Number of mutations to consider per position
            combo_size: Size of mutation combinations
            domain_focus: Optional domain name to focus on
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            List of mutation combinations, where each combination is a list of mutation strings
        """
        # Get top positions by summing probabilities of all mutations at each position
        position_scores = self.mutations_df.groupby('position')['probability'].sum().reset_index()
        
        # Filter positions by domain if specified
        if domain_focus and domain_focus in self.domains:
            domain_positions = self.domains[domain_focus]
            position_scores = position_scores[position_scores['position'].isin(domain_positions)]
        
        # Get top positions
        top_pos = position_scores.sort_values('probability', ascending=False).head(top_positions)['position'].tolist()
        
        # Get top mutations for each position
        top_mutations = self.get_top_mutations_per_position(mutations_per_position)
        
        # Filter to only include top positions
        filtered_mutations = {pos: muts for pos, muts in top_mutations.items() if pos in top_pos}
        
        # Generate all possible combinations of positions
        position_combos = list(itertools.combinations(filtered_mutations.keys(), combo_size))
        
        # If there are too many combinations, sample randomly
        if len(position_combos) > max_combinations:
            print(f"Warning: {len(position_combos)} combinations is too many. Sampling {max_combinations} combinations.")
            position_combos = random.sample(position_combos, max_combinations)
        
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
    
    def calculate_spatial_score(self, positions: List[int]) -> float:
        """
        Calculate a score based on the spatial arrangement of residues.
        Higher scores for mutations that are spatially close or in contact.
        
        Args:
            positions: List of residue positions
            
        Returns:
            Spatial score between 0 and 1
        """
        if self.distance_matrix is None or not all(pos in self.residue_to_idx for pos in positions):
            return 0.5  # Neutral score if structure information is not available
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos1, pos2 = positions[i], positions[j]
                dist = self.get_spatial_distance(pos1, pos2)
                if dist is not None:
                    distances.append(dist)
        
        if not distances:
            return 0.5
        
        # Calculate average distance
        avg_distance = np.mean(distances)
        
        # Convert to score (closer = higher score)
        # Typical protein interaction distances are 3-10 Angstroms
        # We want a score of 1.0 for distances <= 5A, and 0.0 for distances >= 25A
        score = max(0.0, min(1.0, 1.0 - (avg_distance - 5.0) / 20.0))
        
        return score
    
    def calculate_domain_diversity_score(self, positions: List[int]) -> float:
        """
        Calculate a score based on domain diversity.
        Higher scores for mutations that span multiple domains.
        
        Args:
            positions: List of residue positions
            
        Returns:
            Domain diversity score between 0 and 1
        """
        if not self.domains or len(self.domains) <= 1:
            return 0.5  # Neutral score if domain information is not available
        
        # Get domains for each position
        position_domains = [self.get_domain_for_position(pos) for pos in positions]
        
        # Flatten and count unique domains
        all_domains = [domain for domains in position_domains for domain in domains]
        unique_domains = set(all_domains)
        
        # Calculate diversity score
        # Higher score if positions span multiple domains
        score = len(unique_domains) / min(len(positions), len(self.domains))
        
        return score
    
    def calculate_synergy_score(self, 
                              mutation_combo: List[str], 
                              synergy_model: str = 'multiplicative',
                              structure_weight: float = 0.3,
                              domain_weight: float = 0.2) -> Dict:
        """
        Calculate a synergy score for a combination of mutations.
        
        Args:
            mutation_combo: List of mutation strings (e.g., ['M1L', 'A5E'])
            synergy_model: Model for calculating synergy ('multiplicative', 'additive', or 'geometric')
            structure_weight: Weight for structural information in synergy calculation
            domain_weight: Weight for domain information in synergy calculation
            
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
        
        # Extract positions and amino acid changes
        positions = [int(''.join(filter(str.isdigit, mut))) for mut in mutation_combo]
        orig_aas = [mut[0] for mut in mutation_combo]
        new_aas = [mut[-1] for mut in mutation_combo]
        
        # Calculate position diversity (normalized distance between positions)
        pos_range = max(positions) - min(positions)
        sequence_length = max(self.positions['position'])  # Approximate sequence length
        position_diversity = min(1.0, pos_range / (sequence_length * 0.3))  # Normalize, cap at 1.0
        
        # Calculate amino acid diversity (how different the changes are)
        orig_aa_diversity = len(set(orig_aas)) / len(orig_aas)
        new_aa_diversity = len(set(new_aas)) / len(new_aas)
        aa_diversity = (orig_aa_diversity + new_aa_diversity) / 2
        
        # Calculate spatial score if structure information is available
        spatial_score = self.calculate_spatial_score(positions)
        
        # Calculate domain diversity score if domain information is available
        domain_score = self.calculate_domain_diversity_score(positions)
        
        # Calculate average probability
        avg_prob = sum(individual_probs) / len(individual_probs)
        
        # Calculate synergy factor with structure and domain information
        # Base synergy factor (from sequence only)
        base_synergy_factor = (0.4 * position_diversity + 0.3 * aa_diversity + 0.3 * avg_prob) * 0.5
        
        # Adjust with structure and domain information
        structure_factor = structure_weight * spatial_score
        domain_factor = domain_weight * domain_score
        
        # Final synergy factor
        synergy_factor = base_synergy_factor + structure_factor + domain_factor
        synergy_factor = min(1.0, synergy_factor)  # Cap at 1.0
        
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
            'spatial_score': spatial_score,
            'domain_score': domain_score,
            'positions': positions
        }
    
    def evaluate_combinations(self, 
                             combos: List[List[str]], 
                             synergy_model: str = 'multiplicative',
                             structure_weight: float = 0.3,
                             domain_weight: float = 0.2) -> pd.DataFrame:
        """
        Evaluate a list of mutation combinations and return results as a DataFrame.
        
        Args:
            combos: List of mutation combinations
            synergy_model: Model for calculating synergy
            structure_weight: Weight for structural information
            domain_weight: Weight for domain information
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for combo in combos:
            result = self.calculate_synergy_score(
                combo, 
                synergy_model=synergy_model,
                structure_weight=structure_weight,
                domain_weight=domain_weight
            )
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by synergistic score
        results_df = results_df.sort_values('synergistic_score', ascending=False)
        
        return results_df
    
    def visualize_top_combinations(self, 
                                  results_df: pd.DataFrame, 
                                  output_file: Optional[str] = None,
                                  top_n: int = 20,
                                  show_structure_score: bool = True,
                                  show_domain_score: bool = True) -> None:
        """
        Visualize the top mutation combinations with enhanced information.
        
        Args:
            results_df: DataFrame with evaluation results
            output_file: Path to save the visualization
            top_n: Number of top combinations to visualize
            show_structure_score: Whether to show structure scores
            show_domain_score: Whether to show domain scores
        """
        # Get top combinations
        top_combos = results_df.head(top_n).copy()
        
        # Create a more readable mutation representation
        top_combos['mutation_combo'] = top_combos['mutations'].apply(lambda x: ' + '.join(x))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot synergistic scores
        bars = sns.barplot(x='synergistic_score', y='mutation_combo', data=top_combos, color='skyblue', ax=ax)
        
        # Add expected scores as smaller bars
        sns.barplot(x='expected_score', y='mutation_combo', data=top_combos, color='navy', alpha=0.5, ax=ax)
        
        # Add synergy factor as text
        for i, row in enumerate(top_combos.iterrows()):
            ax.text(row[1]['expected_score'] + 0.02, i, f"+{row[1]['synergy_factor']:.2f}", 
                   va='center', color='darkgreen', fontweight='bold')
        
        # Add structure and domain scores if available
        if show_structure_score and 'spatial_score' in top_combos.columns:
            for i, row in enumerate(top_combos.iterrows()):
                ax.text(row[1]['synergistic_score'] + 0.02, i, f"S:{row[1]['spatial_score']:.2f}", 
                       va='center', color='purple')
        
        if show_domain_score and 'domain_score' in top_combos.columns:
            for i, row in enumerate(top_combos.iterrows()):
                ax.text(row[1]['synergistic_score'] + 0.12, i, f"D:{row[1]['domain_score']:.2f}", 
                       va='center', color='brown')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Mutation Combination')
        ax.set_title('Top Mutation Combinations with Synergy Potential')
        ax.legend(['Synergistic Score', 'Expected Score'])
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
    
    def visualize_mutations_on_structure(self,
                                        mutations: List[str],
                                        pdb_file: str,
                                        output_html: str) -> None:
        """
        Visualize mutations on the protein structure.
        
        Args:
            mutations: List of mutation strings
            pdb_file: Path to the PDB file
            output_html: Path to save the HTML visualization
        """
        if not os.path.exists(pdb_file):
            print(f"Error: PDB file {pdb_file} not found")
            return
        
        # Extract positions from mutations
        positions = [int(''.join(filter(str.isdigit, mut))) for mut in mutations]
        
        # Read the PDB file
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()
        
        # Create a 3Dmol view
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, 'pdb')
        
        # Style the protein as a cartoon
        view.setStyle({'cartoon': {'color': 'lightgray'}})
        
        # Highlight mutation positions
        for pos in positions:
            view.addStyle({'resi': pos}, {'cartoon': {'color': 'red'}, 'stick': {}})
        
        # Zoom to fit the model
        view.zoomTo()
        
        # Save the view to an HTML file
        view.write_html(output_html)
        print(f"Structure visualization saved to {output_html}")

def main():
    parser = argparse.ArgumentParser(description='Advanced combination mutation design with structure and domain information')
    parser.add_argument('--mutations_csv', type=str, required=True,
                        help='Path to the CSV file containing mutation predictions')
    parser.add_argument('--pdb_file', type=str, default=None,
                        help='Optional path to a PDB file with protein structure')
    parser.add_argument('--domains_file', type=str, default=None,
                        help='Optional path to a CSV file with domain information')
    parser.add_argument('--output_prefix', type=str, default='advanced_combo_mutations',
                        help='Prefix for output files')
    parser.add_argument('--top_positions', type=int, default=30,
                        help='Number of top positions to consider')
    parser.add_argument('--mutations_per_position', type=int, default=2,
                        help='Number of mutations to consider per position')
    parser.add_argument('--combo_size', type=int, default=3,
                        help='Size of mutation combinations')
    parser.add_argument('--max_combinations', type=int, default=100000,
                        help='Maximum number of combinations to generate')
    parser.add_argument('--synergy_model', type=str, default='multiplicative',
                        choices=['multiplicative', 'additive', 'geometric'],
                        help='Model for calculating synergy')
    parser.add_argument('--structure_weight', type=float, default=0.3,
                        help='Weight for structural information in synergy calculation')
    parser.add_argument('--domain_weight', type=float, default=0.2,
                        help='Weight for domain information in synergy calculation')
    parser.add_argument('--domain_focus', type=str, default=None,
                        help='Optional domain name to focus on')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top combinations to save')
    parser.add_argument('--diversity', action='store_true',
                        help='Generate diverse combinations')
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                        help='Weight for diversity vs. score (0-1)')
    parser.add_argument('--visualize_structure', action='store_true',
                        help='Visualize mutations on structure')
    
    args = parser.parse_args()
    
    # Initialize designer
    designer = AdvancedComboMutationDesigner(
        mutations_csv=args.mutations_csv,
        pdb_file=args.pdb_file,
        domains_file=args.domains_file
    )
    
    # Generate combinations
    combos = designer.generate_mutation_combinations(
        top_positions=args.top_positions,
        mutations_per_position=args.mutations_per_position,
        combo_size=args.combo_size,
        domain_focus=args.domain_focus,
        max_combinations=args.max_combinations
    )
    
    # Evaluate combinations
    results = designer.evaluate_combinations(
        combos, 
        synergy_model=args.synergy_model,
        structure_weight=args.structure_weight,
        domain_weight=args.domain_weight
    )
    
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
            top_n=min(20, len(diverse_results)),
            show_structure_score=args.pdb_file is not None,
            show_domain_score=args.domains_file is not None
        )
        
        # Visualize on structure if requested
        if args.visualize_structure and args.pdb_file:
            top_mutations = diverse_results.iloc[0]['mutations']
            if isinstance(top_mutations, str):
                top_mutations = top_mutations.split(' + ')
            designer.visualize_mutations_on_structure(
                top_mutations,
                args.pdb_file,
                f"{args.output_prefix}_structure.html"
            )
    
    # Save all results
    designer.save_results(results, f"{args.output_prefix}.csv", top_n=args.top_n)
    
    # Visualize top combinations
    designer.visualize_top_combinations(
        results, 
        f"{args.output_prefix}.png", 
        top_n=20,
        show_structure_score=args.pdb_file is not None,
        show_domain_score=args.domains_file is not None
    )
    
    print(f"Completed advanced combination mutation design for {len(combos)} combinations")

if __name__ == "__main__":
    main()

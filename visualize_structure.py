import py3Dmol
from biopandas.pdb import PandasPdb
import os

def visualize_pdb(pdb_file: str, output_html: str):
    """
    Generates an interactive 3D visualization of a PDB file, colored by pLDDT score,
    and saves it as an HTML file.

    Args:
        pdb_file (str): Path to the input PDB file.
        output_html (str): Path to save the output HTML file.
    """
    # Check if the file exists before trying to open it
    if not os.path.exists(pdb_file):
        print(f"Error: PDB file not found at the specified path.")
        print(f"Attempted path: {pdb_file}")
        print(f"Current working directory: {os.getcwd()}")
        return

    # Read the PDB file
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()

    # Use biopandas to get pLDDT scores (stored in b_factor column)
    pdb_df = PandasPdb().read_pdb(pdb_file)
    b_factors = pdb_df.df['ATOM']['b_factor']

    # Create a 3Dmol view
    view = py3Dmol.view(width=800, height=600)

    # Add the model to the view
    view.addModel(pdb_data, 'pdb')

    # Define the color scheme based on pLDDT (B-factor)
    # AlphaFold's color scheme:
    # >90:   High accuracy (blue)
    # 70-90: Confident (cyan)
    # 50-70: Low confidence (yellow)
    # <50:   Very low confidence (orange)
    color_map = {
        'blue':   [[90, 100]],
        'cyan':   [[70, 90]],
        'yellow': [[50, 70]],
        'orange': [[0, 50]]
    }

    # Style the protein as a cartoon and color by pLDDT
    view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'linear', 'colors': ['orange', 'yellow', 'cyan', 'blue'], 'min': 0, 'max': 100}}})

    # Zoom to fit the model
    view.zoomTo()

    # Save the view to an HTML file
    view.write_html(output_html)

    print(f"Visualization saved to {output_html}")
    print("You can now open this file in your web browser.")

if __name__ == "__main__":
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_path = os.path.join(script_dir, 'result.pdb')
    html_path = os.path.join(script_dir, 'protein_visualization.html')
    
    visualize_pdb(pdb_path, html_path)

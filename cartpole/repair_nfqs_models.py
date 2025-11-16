# repairs nfqs models that have received an extended action space but have not been saved with the original action normalization parameters. These models would "move" the action values when re-loaded from disk and would, thus, not perform well.

import os
import sys
import numpy as np

from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.preprocessing.normalization import StackNormalizer
from psipy.rl.plants.real.pact_cartpole.cartpole import SwingupContinuousDiscreteAction

def main():
    # Check if there are command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python repair_nfqs_models.py <file1> [<file2> ...]")
        sys.exit(1)

    normalizer = StackNormalizer("minmax")
    normalizer.fit(np.asarray(SwingupContinuousDiscreteAction.legal_values[0], dtype=float)[..., None])

    print("REPARING WITH NORMALIZER CONFIG: ", normalizer.get_config())

    # Iterate over the files provided as command-line arguments
    for file_path in sys.argv[1:]:
        try:
            print("Repairing: ", file_path)

            # Load the NFQs model
            nfqs_model = NFQs.load(file_path)
            print("Replacing present normalizer config: ", nfqs_model.action_normalizer.get_config())
            print("With: ", normalizer.get_config())

            nfqs_model.action_normalizer = normalizer # keep the normalizer of the original, non-extended space.
            nfqs_model.action_values = nfqs_model.action_type.legal_values[0] # make sure, the action_values are stored to config this time.

            # Remove the original file
            os.remove(file_path)
            
            # Save the model back to the same file
            nfqs_model.save(file_path)
            
            print(f"Successfully repaired and resaved: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main()





import shutil
import os
import meds_reader
import femr.models.tokenizer
import pickle

TARGET_DIR = '/data/processed_datasets/processed_datasets/zj2398/femr'
database = meds_reader.SubjectDatabase('/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform_meds_reader')

# First, we need to train a tokenizer
# Note, we need to use a hierarchical tokenizer for MOTOR

with open('input/ontology.pkl', 'rb') as f:
    ontology = pickle.load(f)

# NOTE: A vocab size of 128 is probably too low for a real model. 128 was chosen to make this tutorial quick to run
# NOTE: Normally you would train the tokenizer on only the train database, but for such a tiny dataset that's not enough
tokenizer = femr.models.tokenizer.HierarchicalTokenizer.train(
    database, vocab_size=1024 * 16, ontology=ontology, min_fraction=1e-9) # Normally min_fraction should be set higher, to 1e-4, but need a small min fraction to get enough codes

# Save the tokenizer to the same directory as the model
tokenizer.save_pretrained(os.path.join(TARGET_DIR, "motor_model"))
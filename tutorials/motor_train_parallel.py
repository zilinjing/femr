import shutil
import os
import meds_reader
import femr.models.tokenizer
from femr.models.tokenizer import hierarchical_tokenizer_parallel
import pickle

TARGET_DIR = '/user/zj2398/cache/motor'
meds_db_path = '/user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader'
database = meds_reader.SubjectDatabase('/user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader')

# First, we need to train a tokenizer
# Note, we need to use a hierarchical tokenizer for MOTOR

with open('input/ontology.pkl', 'rb') as f:
    ontology = pickle.load(f)

# with open('/user/zj2398/cache/motor/input/ontology.pkl', 'rb') as f:
#     ontology = pickle.load(f)

# count = 0
# for subject_id in database:
#     count += 1
#     print(database[subject_id].events)
#     if count > 10:
#         break

# subjects = list(database)
# print(subjects)

# NOTE: A vocab size of 128 is probably too low for a real model. 128 was chosen to make this tutorial quick to run
# NOTE: Normally you would train the tokenizer on only the train database, but for such a tiny dataset that's not enough
tokenizer = hierarchical_tokenizer_parallel.HierarchicalTokenizer.train(
    meds_db_path, vocab_size=1024 * 16, ontology=ontology, min_fraction=1e-9, n_workers=100) # Normally min_fraction should be set higher, to 1e-4, but need a small min fraction to get enough codes

# Save the tokenizer to the same directory as the model
tokenizer.save_pretrained(os.path.join(TARGET_DIR, "motor_model"))

import femr.models.tasks
import pickle
import meds_reader
import femr.splits
import os

# Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
load_path = '/user/zj2398/cache/motor/motor_model'

with open('input/ontology.pkl', 'rb') as f:
    ontology = pickle.load(f)
tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(pretrained_model_name_or_path=load_path,ontology=ontology)

database = meds_reader.SubjectDatabase('/user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader')

# use hash split to split the database into train and test (ratio = frac_test)
main_split = femr.splits.generate_hash_split(list(database), 97, frac_test=0.15)

# Note that we want to save this to the target directory since this is important information

train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 87, frac_test=0.15)

main_database = database.filter(main_split.train_subject_ids)
train_database = main_database.filter(train_split.train_subject_ids)
val_database = main_database.filter(train_split.test_subject_ids)

motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
    train_database, tokenizer, num_tasks=2048, num_bins=4, final_layer_size=32, min_fraction=1e-9)  # Normally min_fraction should be set higher, to 1e-4, but need a small min fraction to get enough codes

# It's recommended to save this with pickle to avoid recomputing since it's an expensive operation
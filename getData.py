from datasets import load_dataset

ds = load_dataset("RahulPil/Dermi_Acne_Dataset")
ds.save_to_disk("./skintelligent-acne/data")
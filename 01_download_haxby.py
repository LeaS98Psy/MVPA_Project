from nilearn import datasets

print("Downloading Haxby dataset (approx. 5 GB)...")
haxby = datasets.fetch_haxby()
print("Dataset saved in:", haxby.data_dir)
print("Functional image:", haxby.func[0])

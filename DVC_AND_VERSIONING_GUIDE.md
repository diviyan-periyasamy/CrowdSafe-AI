# Version Control & DVC — CrowdSafe AI

## 1. Git / GitHub for code
Standard git tracks `.py` files, notebooks, Dockerfile, workflow yml, etc.

```bash
git init
git add .
git commit -m "Initial CrowdSafe AI MLOps pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/crowdsafe-ai.git   # EDIT HERE
git push -u origin main
```

**Why plain Git is not enough here:** `best.pt` (model weights) and any
training image datasets are large binary files. Committing them directly
to Git bloats the repository, makes `git clone` painfully slow, and Git
was never designed to diff binary files meaningfully.

## 2. Why DVC (Data Version Control)
DVC solves exactly this problem — it version-controls large files (datasets,
model weights) the same way Git version-controls code, but stores the
actual bytes in separate remote storage (Google Drive, S3, local folder,
DagsHub, etc.) while Git only tracks a small `.dvc` pointer file.

**How it would be used in CrowdSafe AI:**

```bash
pip install dvc

dvc init

# Track the dataset and the trained model with DVC instead of Git
dvc add data/crowd_dataset
dvc add models/best.pt

# This creates crowd_dataset.dvc and best.pt.dvc — small text pointer files
git add data/crowd_dataset.dvc models/best.pt.dvc .gitignore
git commit -m "Track dataset and model weights with DVC"

# Point DVC at remote storage (Google Drive is easiest for a student project)
dvc remote add -d gdrive_storage gdrive://YOUR_FOLDER_ID   # EDIT HERE
dvc push
```

**Benefit for this project specifically:**
- Every time `train_mlflow.py` produces a new `best.pt` (e.g. after adding
  more Roboflow images), running `dvc add models/best.pt` + `git commit`
  creates a new versioned pointer — so we can always `dvc checkout` an
  older model version if a new fine-tune performs worse.
- Team members (or the examiner) can `git clone` the repo (fast — no
  binaries) and then `dvc pull` to fetch only the dataset/model version
  they need.
- Combined with MLflow (which tracks *metrics* per run) and DVC (which
  tracks *data/model files* per run), the pipeline becomes fully
  reproducible: any past run's code, data, and resulting model can be
  restored exactly.

## 3. Practical note for this coursework
Setting up a real DVC remote (Google Drive auth, S3 keys, etc.) is not
required to demonstrate understanding — `dvc init` + `dvc add` + committing
the `.dvc` pointer files locally is enough to show the workflow to the
examiner. Explain the remote-push step conceptually if you don't want to
set up cloud credentials for a coursework demo.

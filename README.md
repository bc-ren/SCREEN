# SCREEN

**Turning Visual Matches into Relative Class Evidence for Generalized Zero-Shot Recognition**

SCREEN scores each image against class descriptors, then subtracts a candidate-specific reference level estimated from seen-class training images. The ViT-B/16 encoder and the class descriptors stay frozen.

## Benchmark results

Mean per-class top-1 accuracy (%). Best-H supplies the generalized zero-shot results (U, S and H). Best-CZSL supplies conventional zero-shot accuracy. The seen-class offsets were fitted on held-out splits of the seen training data.

| Dataset and configurations | CZSL | U | S | H |
|---|---:|---:|---:|---:|
| [AWA2](configs/AWA2.json) | 89.6 | 85.2 | 93.1 | 89.0 |
| [CUB](configs/CUB.json) | 93.5 | 89.6 | 86.0 | 87.8 |
| [SUN](configs/SUN.json) | 80.2 | 66.7 | 57.6 | 61.9 |

Each file contains the training and scoring hyperparameters for both profiles. To print a complete configuration, run `python -m screen config CUB Best-H` or `python -m screen config CUB Best-CZSL`.

## Data

Use Python 3.11 or later. Install the dependencies and download the frozen [ViT-B/16 weights](https://huggingface.co/google/vit-base-patch16-224-in21k/tree/b4569560a39a0f1af58e3ddaf17facf20ab919b0).

```sh
python -m pip install -r requirements.txt
hf download google/vit-base-patch16-224-in21k config.json model.safetensors \
    --revision b4569560a39a0f1af58e3ddaf17facf20ab919b0 --local-dir backbone
```

Place each dataset's images under `images/DATASET/`, preserving their class directories. Put `att_splits.mat` and `res101.mat` in `splits/DATASET/`. SCREEN reads image paths and labels from the latter. CUB also needs the 1,024-dimensional sentence descriptors in `sent_splits.mat`. [Benchmark splits](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) · [CUB descriptors](https://github.com/Hanzy1996/CE-GZSL/blob/7bf5358ac4727ea1dc2dc9dec2f453b014500bd8/data/CUB/sent_splits.mat)

The training cohorts contain 23,467 AWA2 images, 7,055 CUB images and 10,290 SUN images. The split file lists the omitted training rows in MATLAB's one-based indexing. Evaluation uses every official test row. [Split definitions](configs/splits.json)

## Train and evaluate

Extract features once per dataset. Both profiles reuse the same cache.

```sh
for role in train test; do
    python -m screen.prepare CUB --splits splits/CUB --images images/CUB \
        --backbone backbone --out data/CUB --role "$role"
done

for profile in Best-H Best-CZSL; do
    python -m screen train CUB "$profile" --data data/CUB --weights "runs/CUB/$profile"
    python -m screen evaluate CUB "$profile" --data data/CUB --weights "runs/CUB/$profile"
done
```

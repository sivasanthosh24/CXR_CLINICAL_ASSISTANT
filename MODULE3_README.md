# Module 3: similar chest X-ray retrieval

Module 3 retrieves visually similar dataset images using frozen OpenCLIP ViT-B/32 image embeddings and cosine similarity. It runs on CPU, does not train a model, and does not use Module 1 predictions or Module 2 captions. The index is built once and saved under `.runtime/module3/image_index.npz`; app launches and chat requests load this saved file instead of rebuilding it.

## Build or rebuild the index

Open the project folder in VS Code, open **Terminal → New Terminal**, and run this single command from the project root:

```powershell
python -m src.module3.build_index
```

The first build downloads the pretrained OpenCLIP weights if they are not already cached. The script processes the source images in batches of 16, then reports indexed rows and skipped/missing data. Re-run the same command after changing the CSV or image collection.

## Launch the app

From the project root terminal:

```powershell
streamlit run app.py
```

Upload a chest X-ray in the sidebar, then either click **⑥ Find Similar Images (Module 3)** or ask **“give me similar images”** in chat. The button works independently of the chatbot and displays up to five ranked images with their matching cleaned original `text` values in the main page. If no index is present, both paths show the index-build command. Other questions continue through the existing chatbot flow.

## Configurable data and index paths

Defaults are resolved relative to the project folder's parent:

- `../10000_JPG_IMAGES`
- `../02_dataset/cxr_10k_master.csv`
- `.runtime/module3/image_index.npz`

Set `CXR_IMAGE_DIR`, `CXR_CSV_PATH`, or `CXR_M3_INDEX_PATH` before building/launching to override a path. Absolute paths and project-root-relative paths are supported. Images are matched only as `<image_id>.jpg`; the original `text` column is used and `text_augment` is ignored.

## VS Code setup

1. Open `E:\ChestXpert_ceg_phase1\CXR_Module1_View_Deployment` in VS Code.
2. Select the project's `.venv` Python interpreter, or create/activate a virtual environment and install dependencies with `pip install -r requirements.txt`.
3. Run `python -m src.module3.build_index` once in the integrated terminal.
4. Run `streamlit run app.py` in that terminal and use the local URL printed by Streamlit.


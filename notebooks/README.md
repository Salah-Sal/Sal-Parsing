# VS Code Colab Workflow & File Handling Guide

This directory contains experimental notebooks (like `dolphin_experiment.ipynb`) designed to run on Google Colab via VS Code. This guide documents the research findings and setup steps required to efficiently handle local files and remote execution.

## 1. Research Findings: `colab-vscode` Extension

We analyzed the [googlecolab/colab-vscode](https://github.com/googlecolab/colab-vscode) repository to understand how it handles local-to-remote file interactions.

**Key Insights:**
*   **Built-in but Experimental:** The extension has native capabilities for uploading, downloading, and mounting files, but they are marked as "experimental" and disabled by default.
*   **No "Sync" Required:** The "Backup and Sync Settings" prompt is a generic VS Code feature and is not required for the Colab extension to function.
*   **Connection Dependency:** File management features (like "Upload to Colab") only appear in the UI when an active connection to a Colab server is established.

## 2. Setup Instructions

To enable full file management capabilities between your local machine and the Colab runtime:

### Step 1: Enable Experimental Settings
1.  Open VS Code Settings (`Cmd + ,`).
2.  Search for "Colab" and enable the following checks:
    *   **`colab.uploading`**: Enables "Right-click -> Upload to Colab" in the file explorer.
    *   **`colab.activityBar`**: Adds a Colab icon to the sidebar to browse the remote filesystem (`/content`).
    *   **`colab.serverMounting`**: Allows mounting the remote filesystem directly into your VS Code workspace.

### Step 2: Establish Connection
1.  Open a `.ipynb` notebook.
2.  Click the kernel picker (top-right) -> **Select Kernel** -> **Colab** -> **New Colab Server** (or connect to an existing one).
3.  Wait for the status to change to "Connected".

## 3. Workflow: Managing Files

### Uploading Local Files
*   **Action:** Right-click any file or folder in the local VS Code Explorer.
*   **Select:** "Upload to Colab".
*   **Result:** The file is uploaded to the root (`/content/`) or the active working directory of the Colab runtime.

### Accessing Remote Files
*   Click the **Colab icon** in the Activity Bar (left sidebar).
*   Browse the remote file system.
*   Right-click remote files to **Download**, **Rename**, or **Delete**.

## 4. Notebook Modifications

We updated `dolphin_experiment.ipynb` to support this non-interactive workflow, replacing the standard `google.colab.files.upload()` widget (which blocks execution in VS Code) with auto-detection logic.

### Code Change:
The notebook now automatically finds PDFs uploaded via the VS Code extension, checking both the current directory and specific paths like `colab_ocr/`.

```python
# Updated logic to locate uploaded PDFs
import glob
import os

possible_paths = [
    "../colab_ocr/2512.24601v2.pdf",  # Check specific user path
    "/content/colab_ocr/2512.24601v2.pdf",
    "./*.pdf",
    "../*.pdf"
]

original_pdf_name = None
for path_pattern in possible_paths:
    matches = [f for f in glob.glob(path_pattern) if "first_9_pages.pdf" not in f]
    if matches:
        original_pdf_name = matches[0]
        break

if not original_pdf_name:
    raise FileNotFoundError("No PDF found. Please upload using 'Right Click -> Upload to Colab'.")
```

## 5. Troubleshooting & Optimizations

### Model Loading Hang (T4 GPU)
**Issue:** The default `demo_layout.py` script from the Dolphin repo hung at `Loading checkpoint shards: 50%` on Colab's standard T4 GPU runtime.
**Cause:** Loading the full model in `float32` (default) exhausted the system RAM (12GB) before it could be moved to the GPU, causing the process to freeze or swap.
**Solution:** We replaced the external script call with an optimized inline `DOLPHIN_OPTIMIZED` class that uses:
*   `torch_dtype=torch.float16`: Loads the model in half-precision (50% memory savings).
*   `device_map="auto"`: Automatically handles offloading to disk/CPU if GPU RAM fills up.
*   `low_cpu_mem_usage=True`: Optimizes loading to avoid RAM spikes.

### Missing Output Directories
**Issue:** `FileNotFoundError: [Errno 2] No such file or directory: './results/output_json/page_001.json'`
**Cause:** The `save_outputs` utility expects specific subdirectories (`output_json`, `layout_visualization`) to exist but doesn't create them.
**Solution:** We added a call to `setup_output_dirs(save_dir)` before the processing loop to ensure the full directory structure is created.

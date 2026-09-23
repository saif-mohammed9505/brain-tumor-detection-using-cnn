"""Every setting you might need to change lives here.

If predictions look wrong, the cause is almost always a mismatch between
these settings and how the model was trained (see README, "Troubleshooting").
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "brain_tumor.h5"
STYLE_PATH = ROOT / "assets" / "style.css"

APP_TITLE = "Brain MRI tumor check"
ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]

# Purely decorative — used by src/ui.py for the header banner and empty state.
# Both free-license (Unsplash), on-topic photos: a CT/MRI head-scan composite
# from the National Cancer Institute, and two clinicians reviewing a scan.
HERO_IMAGE = "https://images.unsplash.com/photo-1631563019676-dade0dbdb8fc?auto=format&fit=crop&w=1400&q=80"
EMPTY_IMAGE = "https://images.unsplash.com/photo-1758691463569-66de91d76452?auto=format&fit=crop&w=800&q=80"

# ---- Preprocessing (must match what you did during training) ---------------

# Used only when the model does not report a fixed input size. (height, width)
FALLBACK_IMAGE_SIZE = (224, 224)

# "0-1"  : divide pixels by 255 (ImageDataGenerator(rescale=1/255), most tutorials)
# "none" : keep 0-255 (the model has its own Rescaling layer, or trained on raw pixels)
RESCALE = "0-1"

# "nearest", "bilinear", "bicubic" or "lanczos".
# Keras load_img / flow_from_directory default to nearest; cv2.resize defaults to bilinear.
RESIZE_METHOD = "bilinear"

# ---- What the model outputs mean -------------------------------------------

# One name per output unit, IN THE ORDER THE MODEL WAS TRAINED.
# Keras numbers class folders alphabetically, so folders glioma / meningioma /
# notumor / pituitary give the list below. To confirm, print
# `train_generator.class_indices` (or `class_names` from image_dataset_from_directory)
# in your training notebook. The number of names must equal the model's output count.
CLASS_NAMES = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

# Position of the "no tumor" class in CLASS_NAMES (0-based).
NO_TUMOR_INDEX = 2

# The app reports a tumor when P(tumor) = 1 - P(no tumor) reaches this value.
# Adjustable in the sidebar.
DEFAULT_THRESHOLD = 0.5

# Results within this distance of the threshold are flagged as uncertain.
UNCERTAIN_MARGIN = 0.10

# ============================================================
# Plant/Crop Disease Detection using CNN
# Google Colab Complete Code
# ------------------------------------------------------------

# ─────────────────────────────────────────────
# CELL 1: Install & Import
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 55)
print("  Plant Disease Detection using CNN")
print("  Artificial Intelligence Course Project")
print("=" * 55)
print(f"\nTensorFlow version : {tf.__version__}")
print(f"GPU Available      : {len(tf.config.list_physical_devices('GPU')) > 0}")

# ─────────────────────────────────────────────
# CELL 2: Download Dataset from Kaggle
# ─────────────────────────────────────────────

from google.colab import files
print("kaggle.json file upload:")
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/*.json
!kaggle datasets download -d abdallahalidev/plantvillage-dataset
!unzip -q plantvillage-dataset.zip -d /content/plantvillage/
print("Dataset is downloaded!")

# ─────────────────────────────────────────────
# CELL 3: Configuration
# ─────────────────────────────────────────────
DATASET_PATH = '/content/plantvillage/plantvillage dataset/color'

IMG_SIZE   = 128   
BATCH_SIZE = 32
EPOCHS     = 20

print(f"Image Size : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Epochs     : {EPOCHS}")

# ─────────────────────────────────────────────
# CELL 4: Data Generator (Augmentation)
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale           = 1.0/255.0,   
    rotation_range    = 20,          
    horizontal_flip   = True,        
    zoom_range        = 0.2,         
    width_shift_range = 0.1,         
    height_shift_range= 0.1,         
    validation_split  = 0.2          
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'training',
    shuffle     = True,
    seed        = 42
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'validation',
    shuffle     = False,
    seed        = 42
)

NUM_CLASSES = len(train_gen.class_indices)
CLASS_NAMES = list(train_gen.class_indices.keys())

print(f"\nTotal Classes      : {NUM_CLASSES}")
print(f"Training Samples   : {train_gen.samples}")
print(f"Validation Samples : {val_gen.samples}")

# ─────────────────────────────────────────────
# CELL 5: Sample Images Showing
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Sample Images from PlantVillage Dataset', fontsize=14, fontweight='bold')

imgs, labels = next(train_gen)
label_names = {v: k for k, v in train_gen.class_indices.items()}

for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i])
    ax.set_title(label_names[np.argmax(labels[i])].replace('_', '\n'), fontsize=7)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=120, bbox_inches='tight')
plt.show()
print("Sample images Showing Done!")

# ─────────────────────────────────────────────
# CELL 6: Make CNN Model
# ─────────────────────────────────────────────
def build_model(num_classes, img_size):
    model = keras.Sequential([

        # ── Input ──
        layers.Input(shape=(img_size, img_size, 3)),

        # ── Block 1: 32 Filters ──
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),

        # ── Block 2: 64 Filters ──
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),

        # ── Block 3: 128 Filters ──
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),

        # ── Block 4: 256 Filters ──
        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),

        # ── Classification Head ──
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')

    ], name="PlantDisease_CNN")

    return model

model = build_model(NUM_CLASSES, IMG_SIZE)

# Compile
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# CELL 7: Callbacks
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor             = 'val_loss',
        patience            = 5,
        restore_best_weights= True,
        verbose             = 1
    ),
    ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-6,
        verbose  = 1
    ),
    ModelCheckpoint(
        filepath   = 'best_model.h5',
        monitor    = 'val_accuracy',
        save_best_only = True,
        verbose    = 1
    )
]

# ─────────────────────────────────────────────
# CELL 8: Model Train
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  Model Training Starting...")
print("="*55 + "\n")

history = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs          = EPOCHS,
    callbacks       = callbacks,
    verbose         = 1
)

print("\n Training Fully Done!")

# ─────────────────────────────────────────────
# CELL 9: Accuracy & Loss Graph
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Results — Plant Disease CNN', fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(history.history['accuracy'],     label='Train Accuracy',
             color='#2E75B6', linewidth=2.5)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
             color='#E36C09', linewidth=2.5, linestyle='--')
axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Loss
axes[1].plot(history.history['loss'],     label='Train Loss',
             color='#2E75B6', linewidth=2.5)
axes[1].plot(history.history['val_loss'], label='Validation Loss',
             color='#E36C09', linewidth=2.5, linestyle='--')
axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Final results print
final_acc  = history.history['val_accuracy'][-1]  * 100
best_acc   = max(history.history['val_accuracy'])  * 100
final_loss = history.history['val_loss'][-1]

print(f"\n{'='*55}")
print(f"  FINAL RESULTS")
print(f"{'='*55}")
print(f"  Best Validation Accuracy : {best_acc:.2f}%")
print(f"  Final Validation Accuracy: {final_acc:.2f}%")
print(f"  Final Validation Loss    : {final_loss:.4f}")
print(f"{'='*55}")

# ─────────────────────────────────────────────
# CELL 10: Confusion Matrix
# ─────────────────────────────────────────────
print("\nConfusion Matrix Making...")
print("(This take some times...)\n")

# Predictions
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

# Classification Report
print("\n📊 Classification Report:")
print("─" * 60)
report = classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                digits=3)
print(report)

top_n = min(15, NUM_CLASSES)
cm = confusion_matrix(y_true, y_pred)
cm_top = cm[:top_n, :top_n]
labels_top = CLASS_NAMES[:top_n]
labels_top = [l.replace('___', '\n').replace('_', ' ') for l in labels_top]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_top, yticklabels=labels_top,
            linewidths=0.5, cbar=True)
plt.title(f'Confusion Matrix — Plant Disease CNN\n(First {top_n} Classes)',
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual Label', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n Confusion Matrix Fully Done!")

# ─────────────────────────────────────────────
# CELL 11: Per-class Accuracy Bar Chart
# ─────────────────────────────────────────────
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
top_idx = np.argsort(per_class_acc)[::-1][:15]
top_labels = [CLASS_NAMES[i].replace('___', '\n').replace('_', ' ') for i in top_idx]
top_accs   = per_class_acc[top_idx]

plt.figure(figsize=(13, 5))
bars = plt.bar(range(len(top_idx)), top_accs,
               color=['#1A3A5C' if a >= 90 else '#2E75B6' if a >= 80 else '#E36C09'
                      for a in top_accs],
               edgecolor='white', linewidth=0.5)
plt.xticks(range(len(top_idx)), top_labels, rotation=45, ha='right', fontsize=8)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Per-Class Accuracy (Top 15 Classes)', fontsize=13, fontweight='bold')
plt.ylim([0, 110])
plt.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, top_accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
legend_patches = [
    mpatches.Patch(color='#1A3A5C', label='≥ 90% (Excellent)'),
    mpatches.Patch(color='#2E75B6', label='80–90% (Good)'),
    mpatches.Patch(color='#E36C09', label='< 80% (Needs Improvement)')
]
plt.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# CELL 12: Model Save
# ─────────────────────────────────────────────
model.save('plant_disease_cnn_final.h5')
print(" Model save done: plant_disease_cnn_final.h5")

# ─────────────────────────────────────────────
# CELL 13: Predict with new image
# ─────────────────────────────────────────────
def predict_single_image(image_path):
    """
    Give a leaf image , I will tell the disease!
    Usage: predict_single_image('/content/my_leaf.jpg')
    """
    from tensorflow.keras.preprocessing import image as keras_image

    # Load & preprocess
    img = keras_image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = keras_image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predict
    preds = model.predict(img_arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    # Show result
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Image
    axes[0].imshow(img)
    axes[0].set_title(f"Input Leaf Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Top 3 predictions bar chart
    top3_labels = [CLASS_NAMES[i].replace('___', '\n').replace('_', ' ') for i in top3_idx]
    top3_probs  = [preds[i] * 100 for i in top3_idx]
    colors_bar  = ['#1A3A5C', '#2E75B6', '#8DB3E2']
    axes[1].barh(top3_labels[::-1], top3_probs[::-1], color=colors_bar[::-1])
    axes[1].set_xlabel('Confidence (%)', fontsize=10)
    axes[1].set_title('Top 3 Predictions', fontsize=11, fontweight='bold')
    axes[1].set_xlim([0, 110])
    for i, (prob, label) in enumerate(zip(top3_probs[::-1], top3_labels[::-1])):
        axes[1].text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)

    plt.suptitle(f"Predicted: {CLASS_NAMES[top3_idx[0]].replace('___', ' — ').replace('_', ' ')}\n"
                 f"Confidence: {preds[top3_idx[0]]*100:.1f}%",
                 fontsize=12, fontweight='bold', color='#1A3A5C')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=130, bbox_inches='tight')
    plt.show()

    print(f"\n🌿 Predicted Disease : {CLASS_NAMES[top3_idx[0]]}")
    print(f"   Confidence        : {preds[top3_idx[0]]*100:.2f}%")
    return CLASS_NAMES[top3_idx[0]], preds[top3_idx[0]]


# ─────────────────────────────────────────────
# CELL 14: Final Summary Print
# ─────────────────────────────────────────────
overall_acc = np.sum(cm.diagonal()) / np.sum(cm) * 100

print("\n" + "="*55)
print("  PROJECT SUMMARY")
print("="*55)
print(f"  Topic      : Plant Disease Detection using CNN")
print(f"  Dataset    : PlantVillage ({train_gen.samples + val_gen.samples} images)")
print(f"  Classes    : {NUM_CLASSES}")
print(f"  Model      : Custom 4-Block CNN")
print(f"  Parameters : ~26M")
print(f"  Train Acc  : {max(history.history['accuracy'])*100:.2f}%")
print(f"  Val Acc    : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  Overall Acc: {overall_acc:.2f}%")
print("="*55)
print("\n The Training is fully done! ")
print("\nSaved files:")
print("  📊 training_curves.png     — Accuracy & Loss graph")
print("  📊 confusion_matrix.png    — Confusion Matrix")
print("  📊 per_class_accuracy.png  — Per-class bar chart")
print("  💾 plant_disease_cnn_final.h5 — Trained model")
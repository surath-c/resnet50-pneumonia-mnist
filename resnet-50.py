# Pneumonia Detection using Transfer Learning (ResNet50)
# Author: [Your Name] | Dataset: PneumoniaMNIST

# --- Imports ---
from google.colab import files
uploaded = files.upload()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold

# --- Reproducibility ---
np.random.seed(42)
tf.random.set_seed(42)

# --- Load Dataset ---
with np.load("pneumoniamnist.npz") as data:
    x_train = data['train_images']
    y_train = data['train_labels']
    x_val = data['val_images']
    y_val = data['val_labels']
    x_test = data['test_images']
    y_test = data['test_labels']

# --- Focal Loss with Class Weights ---
def focal_loss(gamma=2., alpha=None):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        if alpha is not None:
            alpha_factor = y_true * alpha[1] + (1 - y_true) * alpha[0]
        else:
            alpha_factor = 1.0
        weight = alpha_factor * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return loss

# --- Preprocessing ---
x_train = np.repeat(x_train[..., np.newaxis], 3, -1).astype('float32') / 255.0
x_val = np.repeat(x_val[..., np.newaxis], 3, -1).astype('float32') / 255.0
x_test = np.repeat(x_test[..., np.newaxis], 3, -1).astype('float32') / 255.0

x_train = tf.image.resize(x_train, [224, 224]).numpy()
x_val = tf.image.resize(x_val, [224, 224]).numpy()
x_test = tf.image.resize(x_test, [224, 224]).numpy()

num_classes = 2
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print("Class distribution in training set:", np.bincount(y_train.flatten()))
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.flatten())
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)
alpha = [class_weights_dict[0], class_weights_dict[1]]

# --- Build ResNet50 Transfer Learning Model ---
def build_resnet50_model():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# --- 3-Fold Stratified Cross-Validation ---
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_no = 1
val_metrics_per_fold = []
models = []

print("\n--- 3-Fold Stratified Cross-Validation ---")
for train_index, _ in skf.split(x_train, y_train):
    print(f"\n--- Fold {fold_no}/{n_splits} ---")
    x_tr = x_train[train_index]
    y_tr = y_train_cat[train_index]

    model = build_resnet50_model()
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=focal_loss(gamma=2., alpha=alpha),
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_tr, y_tr, epochs=15, batch_size=32,
              validation_data=(x_val, y_val_cat),
              class_weight=class_weights_dict,
              callbacks=[es], verbose=2)

    for layer in model.layers[-40:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=focal_loss(gamma=2., alpha=alpha),
                  metrics=['accuracy'])

    es_ft = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(x_tr, y_tr, epochs=5, batch_size=32,
              validation_data=(x_val, y_val_cat),
              class_weight=class_weights_dict,
              callbacks=[es_ft], verbose=2)

    y_pred_val_prob = model.predict(x_val)
    y_pred_val_labels = np.argmax(y_pred_val_prob, axis=1)

    accuracy = accuracy_score(y_val, y_pred_val_labels)
    f1 = f1_score(y_val, y_pred_val_labels)
    auc_roc = roc_auc_score(y_val, y_pred_val_prob[:, 1])

    print(f"Fold {fold_no} Validation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}\n  F1-Score: {f1:.4f}\n  AUC-ROC: {auc_roc:.4f}")
    print(confusion_matrix(y_val, y_pred_val_labels))

    val_metrics_per_fold.append({'accuracy': accuracy, 'f1': f1, 'auc_roc': auc_roc, 'model': model})
    models.append(model)
    fold_no += 1

# --- Best Model Selection ---
def combined_metric(metrics):
    return (metrics['auc_roc'] + metrics['f1'] + metrics['accuracy']) / 3

best_fold_index = np.argmax([combined_metric(m) for m in val_metrics_per_fold])
best_model_cv = val_metrics_per_fold[best_fold_index]['model']

# --- Final Evaluation on Test Set ---
print("\n--- Best Model Evaluation on Test Set ---")
test_preds_prob = best_model_cv.predict(x_test)
test_pred_labels = np.argmax(test_preds_prob, axis=1)

print(f"Accuracy: {accuracy_score(y_test, test_pred_labels):.4f}")
print(f"F1-Score: {f1_score(y_test, test_pred_labels):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, test_preds_prob[:, 1]):.4f}")
print(confusion_matrix(y_test, test_pred_labels))
print(classification_report(y_test, test_pred_labels))

fpr, tpr, _ = roc_curve(y_test, test_preds_prob[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, test_preds_prob[:, 1]):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend()
plt.grid()
plt.show()

# --- Ensemble of CV Models ---
print("\n--- Ensemble Evaluation on Test Set ---")
test_preds_all = np.array([model.predict(x_test) for model in models])
test_preds_ensemble = np.mean(test_preds_all, axis=0)
test_pred_labels_ensemble = np.argmax(test_preds_ensemble, axis=1)

print(f"Accuracy (Ensemble): {accuracy_score(y_test, test_pred_labels_ensemble):.4f}")
print(f"F1-Score (Ensemble): {f1_score(y_test, test_pred_labels_ensemble):.4f}")
print(f"AUC-ROC (Ensemble): {roc_auc_score(y_test, test_preds_ensemble[:, 1]):.4f}")
print(confusion_matrix(y_test, test_pred_labels_ensemble))
print(classification_report(y_test, test_pred_labels_ensemble))

fpr, tpr, _ = roc_curve(y_test, test_preds_ensemble[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_ensemble:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Ensemble")
plt.legend()
plt.grid()
plt.show()

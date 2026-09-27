import csv
import random
import os
import json
import numpy as np
import tensorflow as tf
from collections import Counter

# Fixed random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_and_validate_dataset(csv_path="data/landmarks.csv"):
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found.")
        return [], []

    labels = []
    features = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return [], []

        for row_idx, row in enumerate(reader, start=2):
            if len(row) != 64:
                continue
            
            label = row[0]
            try:
                feat = [float(val) for val in row[1:]]
            except ValueError:
                continue
            
            labels.append(label)
            features.append(feat)

    return labels, features

def balance_classes(labels, features):
    # Dynamically discover all unique labels and sort alphabetically
    unique_classes = sorted(list(set(labels)))
    class_dict = {c: [] for c in unique_classes}
    for l, f in zip(labels, features):
        class_dict[l].append(f)
    
    print(f"\nClasses: {len(unique_classes)}")
    print(f"Labels: {' '.join(unique_classes)}")
    print("Samples before balancing:")
    min_samples = min(len(class_dict[c]) for c in unique_classes)
    
    for c in unique_classes:
        print(f"  {c}: {len(class_dict[c])}")
        
    print(f"\nSamples per class after balancing: {min_samples}")

    balanced_labels = []
    balanced_features = []
    
    for c in unique_classes:
        samples = class_dict[c]
        if len(samples) > min_samples:
            samples = random.sample(samples, min_samples)
        
        balanced_features.extend(samples)
        balanced_labels.extend([c] * len(samples))
        
    print(f"Total balanced samples: {len(balanced_labels)}\n")
    return balanced_labels, balanced_features, unique_classes

def stratified_split(labels, features, train_ratio=0.7, val_ratio=0.15):
    class_dict = {}
    for l, f in zip(labels, features):
        class_dict.setdefault(l, []).append(f)
        
    train_labels, train_feats = [], []
    val_labels, val_feats = [], []
    test_labels, test_feats = [], []
    
    for c, samples in class_dict.items():
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_feats.extend(samples[:n_train])
        train_labels.extend([c] * n_train)
        
        val_feats.extend(samples[n_train:n_train+n_val])
        val_labels.extend([c] * n_val)
        
        test_feats.extend(samples[n_train+n_val:])
        test_labels.extend([c] * (n - n_train - n_val))
        
    return (train_labels, train_feats), (val_labels, val_feats), (test_labels, test_feats)

def encode_data(labels, features, label_to_idx):
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int32)
    X = np.array(features, dtype=np.float32)
    return X, y

def custom_evaluation(y_true, y_pred, classes):
    print("Note: scikit-learn is not installed. Using custom evaluation implementation.")
    print("\nClassification Report:\n")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    for i, c_name in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{c_name:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}")

    print("\nConfusion Matrix:\n")
    header = f"{'':<6}" + "".join([f"{c:<6}" for c in classes])
    print(header)
    for i, c_name in enumerate(classes):
        row_str = f"{c_name:<6}" + "".join([f"{val:<6}" for val in cm[i]])
        print(row_str)

def main():
    labels, features = load_and_validate_dataset()
    if not labels:
        return
        
    print(f"Dataset loaded: {len(labels)} samples")
    
    b_labels, b_features, unique_classes = balance_classes(labels, features)
    num_classes = len(unique_classes)
    
    train_data, val_data, test_data = stratified_split(b_labels, b_features, 0.7, 0.15)
    
    print(f"Training: {len(train_data[0])}")
    print(f"Validation: {len(val_data[0])}")
    print(f"Test: {len(test_data[0])}\n")
    
    # Encoding mapping
    label_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    X_train, y_train = encode_data(train_data[0], train_data[1], label_to_idx)
    X_val, y_val = encode_data(val_data[0], val_data[1], label_to_idx)
    X_test, y_test = encode_data(test_data[0], test_data[1], label_to_idx)
    
    print("Training model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Predict on test set
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred, target_names=unique_classes))
        print("\nConfusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))
    except ImportError:
        custom_evaluation(y_test, y_pred, unique_classes)
        
    os.makedirs("models", exist_ok=True)
    model_path = "models/sign_language_model.keras"
    model.save(model_path)
    
    json_path = "models/labels.json"
    with open(json_path, 'w') as f:
        # Save as standard JSON dict with string keys
        str_idx_to_label = {str(k): v for k, v in idx_to_label.items()}
        json.dump(str_idx_to_label, f, indent=4)
        
    print(f"\nModel saved to:\n{model_path}")

if __name__ == "__main__":
    main()

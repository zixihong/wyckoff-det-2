import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

def load_and_preprocess_data(file_path):
    """Load and preprocess test data"""
    print("Loading test data...")
    data = pd.read_csv(file_path, header=None)
    
    # Separate features and target
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def evaluate_model(model, X, y):
    """Evaluate model performance"""
    # Get predictions
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    loss, accuracy = model.evaluate(X, y, verbose=0)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model Loss: {loss:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    return y_pred, y_pred_prob

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Invalid', 'Valid'],
                yticklabels=['Invalid', 'Valid'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('detailed_confusion_matrix.png')
    
def plot_roc_curve(y_true, y_pred_prob):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('detailed_roc_curve.png')

def plot_example_patterns(X, y, y_pred, n_samples=5):
    """Plot example patterns with their true and predicted labels"""
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(y == y_pred)[0]
    incorrect_indices = np.where(y != y_pred)[0]
    
    # Plot some correct predictions
    plt.figure(figsize=(15, 10))
    for i in range(min(n_samples, len(correct_indices))):
        idx = correct_indices[i]
        plt.subplot(2, n_samples, i+1)
        plt.plot(X[idx])
        plt.title(f"True: {'Valid' if y[idx] else 'Invalid'}\nPred: {'Valid' if y_pred[idx] else 'Invalid'}")
        plt.grid(True, alpha=0.3)
    
    # Plot some incorrect predictions
    for i in range(min(n_samples, len(incorrect_indices))):
        idx = incorrect_indices[i]
        plt.subplot(2, n_samples, n_samples+i+1)
        plt.plot(X[idx], color='red')
        plt.title(f"True: {'Valid' if y[idx] else 'Invalid'}\nPred: {'Valid' if y_pred[idx] else 'Invalid'}")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_patterns.png')

def main():
    # Load model
    print("Loading model...")
    model = load_model('wyckoff_pattern_model.h5')
    
    # Load and preprocess test data
    X, y = load_and_preprocess_data('pattern_data_padded_shuffled.csv')
    
    # Evaluate model
    print("Evaluating model...")
    y_pred, y_pred_prob = evaluate_model(model, X, y)
    
    # Plot confusion matrix
    print("Generating visualizations...")
    plot_confusion_matrix(y, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y, y_pred_prob)
    
    # Plot example patterns
    plot_example_patterns(X, y, y_pred)
    
    print("Evaluation complete! Check the generated visualization files.")

if __name__ == "__main__":
    main() 
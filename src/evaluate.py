import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_processing import X_test, y_test
from model import model
from train import history

# 1. Generate Predictions
print("--- Performance Evaluation on Test Set ---")
# Predict and flatten the array to 1D
y_pred = model.predict(X_test).flatten()

# 2. Calculate Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f} BPM")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# 3. Calculate Custom Accuracy (Tolerance-based)
tolerance_1 = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100
tolerance_2 = np.mean(np.abs(y_test - y_pred) <= 2.0) * 100
tolerance_3 = np.mean(np.abs(y_test - y_pred) <= 3.0) * 100
tolerance_4 = np.mean(np.abs(y_test - y_pred) <= 4.0) * 100
tolerance_5 = np.mean(np.abs(y_test - y_pred) <= 5.0) * 100

print(f"Accuracy (within ±1 BPM): {tolerance_1:.2f}%")
print(f"Accuracy (within ±2 BPM): {tolerance_2:.2f}%")
print(f"Accuracy (within ±3 BPM): {tolerance_3:.2f}%")
print(f"Accuracy (within ±4 BPM): {tolerance_4:.2f}%")
print(f"Accuracy (within ±5 BPM): {tolerance_5:.2f}%")


print("-" * 40)

# 4. Plotting Analytics
plt.figure(figsize=(18, 5))

# Plot A: Training & Validation Loss vs Epoch
plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='red')
plt.title('Model Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


# Plot B: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5, color='green', edgecolor='k')
# Draw the ideal perfect prediction line (y = x)
min_val = min(np.min(y_test), np.min(y_pred))
max_val = max(np.max(y_test), np.max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted BPM')
plt.xlabel('Actual BPM')
plt.ylabel('Predicted BPM')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


# Plot C: Error (Residuals) Distribution
#plt.subplot(1, 3, 3)
errors = y_pred - y_test
plt.hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Zero Error')
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Predicted BPM - Actual BPM)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

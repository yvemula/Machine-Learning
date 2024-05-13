# Evaluate the final model on the test set
final_accuracy = accuracy_score(y_test, y_pred_best)
final_classification_report = classification_report(y_test, y_pred_best)
final_confusion_matrix = confusion_matrix(y_test, y_pred_best)

print("Final Model Accuracy:", final_accuracy)
print("Final Model Classification Report:\n", final_classification_report)
print("Final Model Confusion Matrix:\n", final_confusion_matrix)

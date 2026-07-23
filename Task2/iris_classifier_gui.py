import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)


class IrisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Classifier - KNN")
        self.root.geometry("420x520")
        self.root.configure(bg="#1e1e2f")

        title = tk.Label(root, text="Iris Species Classifier", font=("Segoe UI", 14, "bold"),
                          bg="#1e1e2f", fg="white")
        title.pack(pady=(15, 5))

        subtitle = tk.Label(root, text="KNN Model | K = 5", font=("Segoe UI", 9),
                             bg="#1e1e2f", fg="#a0a0b0")
        subtitle.pack(pady=(0, 15))

        self.entries = []
        form_frame = tk.Frame(root, bg="#1e1e2f")
        form_frame.pack(pady=5)

        for name in feature_names:
            row = tk.Frame(form_frame, bg="#1e1e2f")
            row.pack(pady=6, fill="x")

            label = tk.Label(row, text=name.replace("(cm)", "").strip().title(),
                              width=18, anchor="w", font=("Segoe UI", 10),
                              bg="#1e1e2f", fg="white")
            label.pack(side="left")

            entry = tk.Entry(row, font=("Segoe UI", 10), width=10)
            entry.pack(side="left", ipady=4)
            self.entries.append(entry)

        predict_btn = tk.Button(root, text="Predict Species", command=self.predict,
                                 bg="#4e9af1", fg="white", relief="flat",
                                 font=("Segoe UI", 10, "bold"))
        predict_btn.pack(pady=15, ipadx=10, ipady=5)

        self.result_label = tk.Label(root, text="", font=("Segoe UI", 12, "bold"),
                                      bg="#1e1e2f", fg="#4caf50")
        self.result_label.pack(pady=5)

        metrics_btn = tk.Button(root, text="Show Model Metrics", command=self.show_metrics,
                                 bg="#2b2b3d", fg="white", relief="flat",
                                 font=("Segoe UI", 9))
        metrics_btn.pack(pady=10, ipadx=5, ipady=3)

        self.metrics_text = tk.Text(root, height=10, width=45, bg="#2b2b3d", fg="white",
                                     font=("Consolas", 9), state="disabled")
        self.metrics_text.pack(pady=10)

    def predict(self):
        try:
            values = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields.")
            return

        sample = np.array(values).reshape(1, -1)
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]
        species = class_names[prediction]

        self.result_label.config(text=f"Predicted Species: {species}")

    def show_metrics(self):
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")

        report = f"Accuracy: {accuracy:.4f}\n"
        report += f"F1 Score: {f1:.4f}\n\n"
        report += "Confusion Matrix:\n"
        report += "           " + "  ".join(class_names) + "\n"
        for i, row in enumerate(cm):
            report += f"{class_names[i]:<10} " + "  ".join(f"{v:^{len(class_names[0])}}" for v in row) + "\n"

        self.metrics_text.insert("1.0", report)
        self.metrics_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = IrisGUI(root)
    root.mainloop()

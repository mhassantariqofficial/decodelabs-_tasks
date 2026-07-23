import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_roles = [
    {"role": "Data Scientist", "skills": "python machine learning statistics pandas numpy sql"},
    {"role": "Cloud Architect", "skills": "cloud computing aws azure networking security automation"},
    {"role": "Frontend Developer", "skills": "javascript react html css web design ui ux"},
    {"role": "Backend Developer", "skills": "python java sql api design databases automation"},
    {"role": "DevOps Engineer", "skills": "automation cloud computing docker kubernetes ci cd linux"},
    {"role": "AI Engineer", "skills": "python machine learning deep learning neural networks tensorflow"},
    {"role": "Mobile Developer", "skills": "java kotlin swift android ios mobile design"},
    {"role": "Cybersecurity Analyst", "skills": "security networking encryption linux threat analysis"},
    {"role": "Database Administrator", "skills": "sql databases optimization security backup automation"},
    {"role": "Full Stack Developer", "skills": "javascript react python sql api web design"},
]

corpus = [job["skills"] for job in job_roles]
vectorizer = TfidfVectorizer()
item_vectors = vectorizer.fit_transform(corpus)


def recommend(user_skills, top_n=3):
    user_text = " ".join(user_skills).lower()
    user_vector = vectorizer.transform([user_text])
    scores = cosine_similarity(user_vector, item_vectors)[0]

    ranked = sorted(zip(job_roles, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


class RecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tech Stack Recommender")
        self.root.geometry("420x480")
        self.root.configure(bg="#1e1e2f")

        title = tk.Label(root, text="Tech Stack Recommender", font=("Segoe UI", 14, "bold"),
                          bg="#1e1e2f", fg="white")
        title.pack(pady=(15, 5))

        subtitle = tk.Label(root, text="Content-Based Filtering | TF-IDF + Cosine Similarity",
                             font=("Segoe UI", 9), bg="#1e1e2f", fg="#a0a0b0")
        subtitle.pack(pady=(0, 15))

        self.entries = []
        labels = ["Skill 1", "Skill 2", "Skill 3"]

        for label_text in labels:
            row = tk.Frame(root, bg="#1e1e2f")
            row.pack(pady=6, fill="x", padx=30)

            label = tk.Label(row, text=label_text, width=10, anchor="w",
                              font=("Segoe UI", 10), bg="#1e1e2f", fg="white")
            label.pack(side="left")

            entry = tk.Entry(row, font=("Segoe UI", 10))
            entry.pack(side="left", fill="x", expand=True, ipady=4)
            self.entries.append(entry)

        recommend_btn = tk.Button(root, text="Get Recommendations", command=self.get_recommendations,
                                   bg="#4e9af1", fg="white", relief="flat",
                                   font=("Segoe UI", 10, "bold"))
        recommend_btn.pack(pady=15, ipadx=10, ipady=5)

        self.result_text = tk.Text(root, height=12, width=45, bg="#2b2b3d", fg="white",
                                    font=("Consolas", 9), state="disabled")
        self.result_text.pack(pady=10)

    def get_recommendations(self):
        skills = [e.get().strip() for e in self.entries]
        skills = [s for s in skills if s]

        if len(skills) < 3:
            messagebox.showerror("Invalid Input", "Please enter all 3 skills.")
            return

        results = recommend(skills, top_n=3)

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")

        output = "Top 3 Recommended Career Paths:\n\n"
        for i, (job, score) in enumerate(results, start=1):
            output += f"{i}. {job['role']}  (match: {score * 100:.1f}%)\n"

        self.result_text.insert("1.0", output)
        self.result_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderGUI(root)
    root.mainloop()

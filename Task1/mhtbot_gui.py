import tkinter as tk
from datetime import datetime

def get_response(user_input):
    msg = user_input.lower().strip()

    if msg in ["hi", "hello", "hey"]:
        return "Hello! I'm MHTBot. How can I help you today?"

    elif msg in ["how are you", "how r u", "how are you doing"]:
        return "I'm just a bot, but I'm running perfectly! How about you?"

    elif msg in ["what is your name", "who are you", "your name", "name"]:
        return "I'm MHTBot! built by Hassan."

    elif msg in ["who made you", "who created you", "who built you"]:
        return "I was created by Hassan."

    elif "what is ai" in msg or "artificial intelligence" in msg:
        return "AI (Artificial Intelligence) is the simulation of human intelligence in machines."

    elif "what is machine learning" in msg or "ml" == msg:
        return "Machine Learning is a subset of AI where systems learn from data instead of explicit rules."

    elif "what is a chatbot" in msg:
        return "A chatbot is a program that simulates conversation with humans, like me!"

    elif "time" in msg:
        return f"Current time: {datetime.now().strftime('%I:%M %p')}"

    elif "date" in msg or "today" in msg:
        return f"Today is: {datetime.now().strftime('%A, %d %B %Y')}"

    elif "tell me a joke" in msg or "joke" in msg:
        return "There are 10 types of people in the world, Those who understand binary, and those who don't."

    elif "your favorite language" in msg or "best language" in msg:
        return "Python, because it having clean syntax, powerful libraries."

    elif msg in ["help", "what can you do", "commands"]:
        return (
            "I can help with:\n"
            "  • Greetings & small talk\n"
            "  • AI / ML definitions\n"
            "  • Current time & date\n"
            "  • Jokes"
        )

    elif msg in ["exit", "quit", "bye", "goodbye"]:
        return "Goodbye! Keep building!"

    else:
        return "I'm not sure about that yet. Try asking something else or type 'help'."


class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MHTBot")
        self.root.geometry("450x550")
        self.root.configure(bg="#1e1e2f")

        self.chat_area = tk.Text(root, bg="#2b2b3d", fg="white", wrap="word",
                                  font=("Segoe UI", 10), state="disabled")
        self.chat_area.pack(padx=10, pady=10, fill="both", expand=True)

        bottom_frame = tk.Frame(root, bg="#1e1e2f")
        bottom_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.entry = tk.Entry(bottom_frame, font=("Segoe UI", 10))
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", self.send_message)

        send_btn = tk.Button(bottom_frame, text="Send", command=self.send_message,
                              bg="#4e9af1", fg="white", relief="flat")
        send_btn.pack(side="right", padx=(5, 0))

        self.display("Bot", "Hello! I'm MHTBot. Type 'help' for commands.")

    def display(self, sender, text):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"{sender}: {text}\n\n")
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.display("You", user_input)
        self.entry.delete(0, "end")

        response = get_response(user_input)
        self.display("Bot", response)

        if user_input.lower().strip() in ["exit", "quit", "bye", "goodbye"]:
            self.root.after(1500, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatGUI(root)
    root.mainloop()

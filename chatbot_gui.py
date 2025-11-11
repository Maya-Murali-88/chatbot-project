# chatbot_gui.py
import tkinter as tk
from tkinter import scrolledtext
from chatbot import get_response_from_bot

APP_TITLE = "Chatbot"
WINDOW_W, WINDOW_H = 520, 560

def send_message_event(event=None):
    send_message()

def send_message():
    user_message = entry_box.get("1.0", tk.END).strip()
    if not user_message:
        return

    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_message + '\n\n')
    chat_log.config(foreground="#222222", font=("Verdana", 11))

    try:
        response = get_response_from_bot(user_message)
    except Exception as e:
        response = f"Error: {e}"

    chat_log.insert(tk.END, "Bot: " + response + '\n\n')
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

    entry_box.delete("1.0", tk.END)
    entry_box.focus_set()

def build_ui(root):
    root.title(APP_TITLE)
    root.geometry(f"{WINDOW_W}x{WINDOW_H}")
    root.resizable(False, False)

    # --- Chat area (top) ---
    global chat_log
    chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
    chat_log.place(x=10, y=10, width=WINDOW_W-30, height=WINDOW_H-180)

    # --- Input area (bottom) ---
    global entry_box
    entry_box = tk.Text(root, bd=1, wrap=tk.WORD, font=("Verdana", 11))
    entry_box.place(x=10, y=WINDOW_H-160, height=60, width=WINDOW_W-130)
    entry_box.bind("<Return>", lambda e: ("break", send_message_event())[1])  # Enter to send
    entry_box.focus_set()

    # --- Send button on the right ---
    send_button = tk.Button(
        root,
        text="Send",
        font=("Verdana", 12, "bold"),
        bd=0,
        bg="#b35110",
        activebackground="#b35110",
        fg="#f2eaea",
        command=send_message
    )
    send_button.place(x=WINDOW_W-110, y=WINDOW_H-160, height=60, width=90)


if __name__ == "__main__":
    root = tk.Tk()
    build_ui(root)
    root.mainloop()

from tkinter import *
from tkinter import ttk

# colors
YELLOW = "#fece2f"
PURPLE = "#8B5CF6"
DARKER = "#1F2937"
DKBLUE = "#111827"
DKGREY = "#374151"
GRAY   = "#545454"
BLUE   = "#1F2937"
DARK   = "#303340"
LIGHT  = "#d3d3d3"


class HooverButton(Button):
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self["borderwidth"] = 0
        self["font"] = 7
        self["width"] = 12
        self["fg"] = "white"
        self["bg"] = GRAY
        self["cursor"] = "hand2"
        self["activeforeground"] = "white"
        self["activebackground"] = DARKER
        self["disabledforeground"] = DARKER

        self.bind('<Enter>', lambda e: self.config(background=PURPLE))
        self.bind('<Leave>', lambda e: self.config(background=GRAY))


class myLabel(Label):
    def __init__(self, *args, **kwargs):
        Label.__init__(self, *args, **kwargs)
        self["font"] = 15
        self["fg"] = "white"
        self["bg"] = DKBLUE
        self["padx"] = 10
        self["pady"] = 5


class myEntry(Entry):
    def __init__(self, *args, **kwargs):
        Entry.__init__(self, *args, **kwargs)
        self["font"] = 15
        self["width"] = 30
        self["fg"] = YELLOW
        self["bg"] = DARKER
        self["insertbackground"] = "orange"

    def set(self, text):
        self.delete(0, END)
        self.insert(0, text)


class myCombobox(ttk.Combobox):
    def __init__(self, *args, **kwargs):
        ttk.Combobox.__init__(self, *args, **kwargs)
        self["font"] = 15
        self["width"] = 28
        self["state"] = "readonly"


# must be in this order (to calculate edu.num)
education_value = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", 
    "9th", "10th", "11th", "12th",
    "HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college", 
    "Bachelors", "Masters", "Prof-school", "Doctorate"
]
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


class myCombobox(ttk.Combobox):
    def __init__(self, *args, **kwargs):
        ttk.Combobox.__init__(self, *args, **kwargs)
        self["font"] = 15
        self["width"] = 28
        self["state"] = "readonly"
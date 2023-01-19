import pickle
import pandas as pd

from tkinter import *
from tkinter import ttk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Constants
YELLOW = "#fece2f"
PURPLE = "#8B5CF6"
DARKER = "#1F2937"
DARK   = "#303340"
DKGREY = "#374151"

edu_order = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
"HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college", "Bachelors", "Masters", "Prof-school", "Doctorate"]

country = sorted(['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 
'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 
'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 
'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 
'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'])


# Save and Load model in "Python Pickle File"
def save_model(model):
    pickle.dump(model, open("model.pkl", "wb"))


def load_model():
    return pickle.load(open("model.pkl", "rb"))


# Encode dataset column to number base
def cleandata(dataset):
    dataset.drop("fnlwgt", axis = 1, inplace = True)
    for column in dataset.columns:
        MODE = dataset[column].mode()[0]
        dataset[column].fillna(MODE, inplace = True)
    for column in dataset.columns:
        if isinstance(dataset[column].dtype, object):
            LE = LabelEncoder()
            dataset[column] = LE.fit_transform(dataset[column])
    return dataset


# Split income and all others for features
def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1)
    labels = dataset[feature].copy()
    return features, labels


# Dataset: https://www.kaggle.com/uciml/adult-census-income
def train_model():
    dataset = pd.read_csv("adult.csv")
    dataset = cleandata(dataset)

    # train 80 % and test 20 %
    train_set, test_set = train_test_split(dataset, test_size=0.2)

    # train and test
    train_features, train_labels = split_feature_class(train_set, "income")
    test_features, test_labels   = split_feature_class(test_set,  "income")

    # model prediction accuracy
    MODEL = GaussianNB()
    MODEL.fit(train_features, train_labels)

    clf_pred = MODEL.predict(test_features)
    print("Accuracy =", accuracy_score(test_labels, clf_pred))
    save_model(MODEL)
    print("Done Saving Model")


# Predict whether a person has income over $50k per year 
def predict_model(feature, name):
    MODEL = load_model()
    prediction = MODEL.predict(feature)
    if prediction == 0:
        text  = f"{name.get()} has income less than $50k"
        color = "lightblue"
    else:
        text  = f"{name.get()} has income more than $50k"
        color = "lime"
    Label(text=text, padx=10, font=15, fg=color, bg=DARKER
    ).grid(row=6, column=2, rowspan=2, sticky="news")


# Update DataFrame and Label in UI
def update_data(age, workclass, education, status, occupation, 
relation, race, sex, gain, loss, hour, native, name):
    try:
        feature = {
            "age": int(age.get()),
            "workclass": workclass.get(),
            "fnlwgt": 13400,
            "education": education.get(),
            "education.num": edu_order.index(education.get()) + 1,
            "marital.status": status.get(),
            "occupation": occupation.get(),
            "relationship": relation.get(),
            "race": race.get(),
            "sex": sex.get(),
            "capital.gain": int(gain.get()),
            "capital.loss": int(loss.get()),
            "hours.per.week": int(hour.get()),
            "native.country": native.get()
        }
        DF = pd.DataFrame([feature])
        dataset = pd.read_csv("adult.csv")
        dataset, _ = split_feature_class(dataset, "income")
        dataset = pd.concat([dataset, DF], axis = 0)
        dataset = cleandata(dataset)
        predict_model(dataset.iloc[[-1]], name)
    except Exception:
        text = "Please inform all features!"
        Label(text=text, padx=10, font=15, fg="red", bg=DARKER
        ).grid(row=6, column=2, rowspan=2, sticky="news")


# main GUI for prediction
def user_interface():
    ROOT = Tk()
    ROOT.title("Adult Census Income")

    # Set combobox style
    combostyle = ttk.Style()
    combostyle.theme_use('clam')
    combostyle.configure("TCombobox", fieldbackground = DARK)

    # Input Combobox Features
    educhoice = StringVar(value="Bachelors")
    Label(text="Education", padx=10, font=15, fg="white", bg=DARKER).grid(row=0,column=3, sticky="news")
    education = ttk.Combobox(width=28, font=15, textvariable=educhoice, state="readonly")
    education["values"] = edu_order
    education.grid(row=0, column=4, padx=2)

    workchoice = StringVar(value="Please select a Work Class..")
    Label(text="Work Class", padx=10, font=15, fg="white", bg=DARKER).grid(row=1, column=3, sticky="news")
    workclass = ttk.Combobox(width=28, font=15, textvariable=workchoice, state="readonly")
    workclass["values"] = ('Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay')
    workclass.grid(row=1, column=4, padx=2)

    statuschoice = StringVar(value="Please select a Marital Status..")
    Label(text="Marital Status", padx=10, font=15, fg="white", bg=DARKER).grid(row=2, column=3, sticky="news")
    status = ttk.Combobox(width=28, font=15, textvariable=statuschoice, state="readonly")
    status["values"] = ("Married-civ-spouse", "Never-married", "Divorced", "Seperated", "Widowed")
    status.grid(row=2, column=4)

    occuchoice = StringVar(value="Please select a Occcupation..")
    Label(text="Occupation", padx=10, font=15, fg="white", bg=DARKER).grid(row=3, column=3, sticky="news")
    occupation = ttk.Combobox(width=28, font=15, textvariable=occuchoice, state="readonly")
    occupation["values"] = ('Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 
    'Machine-op-inspct', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Other-service')
    occupation.grid(row=3, column=4)

    ralachoice = StringVar(value="Please select a Relation..")
    Label(text="Relation", padx=10, font=15, fg="white", bg=DARKER).grid(row=4, column=3, sticky="news")
    relation = ttk.Combobox(width=28, font=15, textvariable=ralachoice, state="readonly")
    relation["values"] = ('Not-in-family', 'Husband', 'Wife', 'Own-Child', 'Unmarried', 'Other-relative')
    relation.grid(row=4, column=4)

    racechoice = StringVar(value="Please select a Race..")
    Label(text="Race", padx=10, font=15, fg="white", bg=DARKER).grid(row=5, column=3, sticky="news")
    race = ttk.Combobox(width=28, font=15, textvariable=racechoice, state="readonly")
    race["values"] = ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other')
    race.grid(row=5, column=4)

    sexchoice = StringVar(value="Please select a Sex..")
    Label(text="Sex", padx=10, font=15, fg="white", bg=DARKER).grid(row=6, column=3, sticky="news")
    sex = ttk.Combobox(width=28, font=15, textvariable=sexchoice, state="readonly")
    sex["values"] = ('Male', 'Female')
    sex.grid(row=6, column=4)

    choice = StringVar(value="Please select a Native Country..")
    Label(text="Native Country", padx=10, font=15, fg="white", bg=DARKER).grid(row=7, column=3, sticky="news")
    native = ttk.Combobox(width=28, font=15, textvariable=choice, state="readonly")
    native["values"] = country
    native.grid(row=7, column=4)

    # Input Entry Features
    text = StringVar()
    Label(text="Name", padx=10, font=15, fg="white", bg=DARKER).grid(row=0, column=0, sticky="news")
    name = Entry(font=15, width=30, textvariable=text, fg=YELLOW, bg=DARK)
    name.configure(insertbackground="orange")
    name.grid(row=0, column=2, padx=2)

    text = StringVar()
    Label(text="Age", padx=10, font=15, fg="white", bg=DARKER).grid(row=1, column=0, sticky="news")
    age = Entry(font=15, width=30, textvariable=text, fg=YELLOW, bg=DARK)
    age.configure(insertbackground="orange")
    age.grid(row=1, column=2, padx=2)

    text = StringVar(value="0")
    Label(text="Capital Gain", padx=10, font=15, fg="white", bg=DARKER).grid(row=2, column=0, sticky="news")
    gain = Entry(font=15, width=30, textvariable=text, fg=YELLOW, bg=DARK)
    gain.configure(insertbackground="orange")
    gain.grid(row=2, column=2)

    text = StringVar(value="0")
    Label(text="Capital Loss", padx=10, font=15, fg="white", bg=DARKER).grid(row=3, column=0, sticky="news")
    loss = Entry(font=15, width=30, textvariable=text, fg=YELLOW, bg=DARK)
    loss.configure(insertbackground="orange")
    loss.grid(row=3, column=2)

    text = StringVar(value="40")
    Label(text="Hour Per Week", padx=10, font=15, fg="white", bg=DARKER).grid(row=4, column=0, sticky="news")
    hour = Entry(font=15, width=30, textvariable=text, fg=YELLOW, bg=DARK)
    hour.configure(insertbackground="orange")
    hour.grid(row=4, column=2)

    # Output Prediction
    Label(text=None, bg=DARKER).grid(row=5, column=0, columnspan=3, sticky="news")
    Label(text="Person Income", padx=10, font=20, fg="white", bg=DARKER).grid(row=6, column=2, rowspan=2, sticky="news")

    button = Button(ROOT, text="Predict", font=("comicsans", 15), bg=PURPLE, fg="black", cursor="hand2", activebackground=DARKER, activeforeground=PURPLE,
             command=lambda: update_data(age, workclass, education, status, occupation, relation, race, sex, gain, loss, hour, native, name))
    button.grid(row=6, column=0, rowspan=2, sticky="news")
    button.bind('<Enter>', lambda e: button.config(background=DKGREY, fg=PURPLE))
    button.bind('<Leave>', lambda e: button.config(background=PURPLE, fg="black"))

    ROOT.mainloop()


if __name__ == '__main__':
    print(" Adult Census Income ".center(30, '='))
    print("1 Train Model\n2 Prediction")

    command = input("Enter Command: ").strip()
    if command == '1':
        train_model()
    else:
        user_interface()
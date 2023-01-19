import pickle
import pandas as pd

from constant import *
from sklearn.preprocessing import LabelEncoder


def load_model():
    return pickle.load(open("./assets/model.pkl", "rb"))


# Encode dataset column to number base
def cleandata(dataset):
    dataset.drop("fnlwgt", axis=1, inplace=True)
    for column in dataset.columns:
        mode = dataset[column].mode()[0]
        dataset[column].fillna(mode, inplace=True)
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


# Predict whether a person has income over $50k per year
def predict_model(feature, name):
    model = load_model()
    prediction = model.predict(feature)
    user = f'{name.get()} has' if name.get() else 'You have'

    if prediction == 0:
        text = f"{user} income less than $50k"
        color = "lightblue"
    else:
        text = f"{user} income more than $50k"
        color = "lime"

    Label(
        input_frame, text=text, padx=10, font=15, fg=color, bg=DKBLUE
    ).grid(row=6, column=2, rowspan=2, sticky="news")


# Update DataFrame and Label in UI
def update_data(age, workclass, education, status, occupation,
                relation, race, sex, gain, loss, hour, native, name):
    try:
        education_num = education_value.index(education.get()) + 1
        feature = {
            "age": int(age.get()),
            "workclass": workclass.get(),
            "fnlwgt": 13400,
            "education": education.get(),
            "education.num": education_num,
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
        dataset = pd.read_csv("./assets/adult.csv")
        dataset, _ = split_feature_class(dataset, "income")
        dataset = pd.concat([dataset, DF], axis=0)
        dataset = cleandata(dataset)
        predict_model(dataset.iloc[[-1]], name)

    except Exception:
        text = "Please inform all features!"
        Label(
            input_frame, text=text, padx=10, font=15, fg="red", bg=DKBLUE
        ).grid(row=6, column=2, rowspan=2, sticky="news")


if __name__ == '__main__':
    ROOT = Tk()
    ROOT.geometry("875x275")
    ROOT.title("Adult Census Income")
    ROOT.resizable(False, False)
    ROOT.configure(background=DKBLUE)
    ROOT.bind('<Escape>', lambda e: ROOT.destroy())

    # Set combobox style
    combostyle = ttk.Style()
    combostyle.theme_use('clam')
    combostyle.configure("TCombobox", fieldbackground=DARKER)

    # Input frame
    global input_frame
    input_frame = Frame(ROOT, pady=10, bg=DKBLUE)
    input_frame.pack()

    # labels for GUI
    global dataset, education_value
    dataset = pd.read_csv("./assets/adult.csv")
    education_value = sorted(dataset['education'].unique().tolist())

    # Input Combobox Features
    eduChoice = StringVar(value="Please select an Education..")
    myLabel(input_frame, text="Education").grid(row=0, column=3, sticky="news")
    education = myCombobox(input_frame, textvariable=eduChoice)
    education["values"] = education_value
    education.grid(row=0, column=4, padx=2)

    workChoice = StringVar(value="Please select a Work Class..")
    myLabel(input_frame, text="Work Class").grid(row=1, column=3, sticky="news")
    workclass = myCombobox(input_frame, textvariable=workChoice)
    workclass_value = sorted(dataset['workclass'].unique().tolist())
    workclass_value.remove('?')
    workclass["values"] = workclass_value
    workclass.grid(row=1, column=4, padx=2)

    statusChoice = StringVar(value="Please select a Marital Status..")
    myLabel(input_frame, text="Marital Status").grid(row=2, column=3, sticky="news")
    status = myCombobox(input_frame, textvariable=statusChoice)
    status["values"] = sorted(dataset['marital.status'].unique().tolist())
    status.grid(row=2, column=4)

    occuChoice = StringVar(value="Please select an Occcupation..")
    myLabel(input_frame, text="Occupation").grid(row=3, column=3, sticky="news")
    occupation = myCombobox(input_frame, textvariable=occuChoice)
    occupation_value = sorted(dataset['occupation'].unique().tolist())
    occupation_value.remove('?')
    occupation["values"] = occupation_value
    occupation.grid(row=3, column=4)

    relaChoice = StringVar(value="Please select a Relation..")
    myLabel(input_frame, text="Relation").grid(row=4, column=3, sticky="news")
    relation = myCombobox(input_frame, textvariable=relaChoice)
    relation["values"] = sorted(dataset['relationship'].unique().tolist())
    relation.grid(row=4, column=4)

    raceChoice = StringVar(value="Please select a Race..")
    myLabel(input_frame, text="Race").grid(row=5, column=3, sticky="news")
    race = myCombobox(input_frame, textvariable=raceChoice)
    race["values"] = sorted(dataset['race'].unique().tolist())
    race.grid(row=5, column=4)

    sexChoice = StringVar(value="Please select a Sex..")
    myLabel(input_frame, text="Sex").grid(row=6, column=3, sticky="news")
    sex = myCombobox(input_frame, textvariable=sexChoice)
    sex["values"] = ('Male', 'Female')
    sex.grid(row=6, column=4)

    countryChoice = StringVar(value="Please select a Native Country..")
    myLabel(input_frame, text="Native Country").grid(row=7, column=3, sticky="news")
    native = myCombobox(input_frame, textvariable=countryChoice)
    native_country = sorted(dataset['native.country'].unique().tolist())
    native_country.remove('?')
    native["values"] = native_country
    native.grid(row=7, column=4)

    # Input Entry Features
    text = StringVar()
    myLabel(input_frame, text="Name").grid(row=0, column=0, sticky="news")
    name = myEntry(input_frame, textvariable=text)
    name.grid(row=0, column=2, padx=2)

    text = StringVar()
    myLabel(input_frame, text="Age").grid(row=1, column=0, sticky="news")
    age = myEntry(input_frame, textvariable=text)
    age.grid(row=1, column=2, padx=2)

    text = StringVar(value="0")
    myLabel(input_frame, text="Capital Gain").grid(row=2, column=0, sticky="news")
    gain = myEntry(input_frame, textvariable=text)
    gain.grid(row=2, column=2)

    text = StringVar(value="0")
    myLabel(input_frame, text="Capital Loss").grid(row=3, column=0, sticky="news")
    loss = myEntry(input_frame, textvariable=text)
    loss.grid(row=3, column=2)

    text = StringVar(value="40")
    myLabel(input_frame, text="Hour Per Week").grid(row=4, column=0, sticky="news")
    hour = myEntry(input_frame, textvariable=text)
    hour.grid(row=4, column=2)

    # Output Prediction
    myLabel(input_frame, text=None).grid(row=5, column=0, columnspan=3, sticky="news")
    myLabel(input_frame, text="Person Income").grid(row=6, column=2, rowspan=2, sticky="news")

    # TODO random button
    predict_command = lambda: update_data(
        age, workclass, education, status, occupation, relation, race, sex, gain, loss, hour, native, name
    )
    predict_button = HooverButton(input_frame, text="Predict", command=predict_command)
    predict_button.grid(row=7, column=0, rowspan=1, sticky="news")

    ROOT.mainloop()
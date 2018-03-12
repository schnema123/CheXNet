import csv
import random

# Split Data 70/10/20 train/eval/test

def csv_writer(filename, fieldnames):
    training_file = open(filename, "w", newline="")
    csv_writer = csv.DictWriter(training_file, fieldnames)
    csv_writer.writeheader()
    return csv_writer

    
def parse_rows(rows, fieldnames):

    # We have to make sure that there is no patient overlap between sets
    patients = []
    for row in rows:

        patient_id = int(row["Patient ID"])
        patients_len = len(patients)
        if patients_len < patient_id:
            patients.append([row])
        else:
            patients[patient_id - 1].append(row)

    random.shuffle(patients)

    training_writer = csv_writer("../nihcc/Data_Train.csv", fieldnames)
    eval_writer = csv_writer("../nihcc/Data_Eval.csv", fieldnames)
    test_writer = csv_writer("../nihcc/Data_Test.csv", fieldnames)

    for patient in patients:
        rand_number = random.random()
        if rand_number < 0.7:
            training_writer.writerows(patient)
        elif rand_number < 0.8:
            eval_writer.writerows(patient)
        else:
            test_writer.writerows(patient)


with open("../nihcc/Data_Entry_2017.csv") as csv_file:
    rows = []

    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        rows.append(row)

    parse_rows(rows, csv_reader.fieldnames)

import csv
import random

# Fichiers d'entrée et de sortie
input_file = "data/airports.csv"
output_file = "data/waiting_times.csv"

# Lecture du fichier CSV et écriture du nouveau fichier avec les temps d'attente
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Sauter l'en-tête si nécessaire

    with open(output_file, mode='w', newline='', encoding='utf-8') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(["Airport_Code", "Waiting_Time"])  # En-tête du nouveau fichier

        for row in reader:
            if len(row) >= 4:  # Vérifier que la ligne a au moins 4 colonnes
                airport_code = row[3]  # 4ème colonne (index 3)
                waiting_time = random.randint(15, 100)  # Temps d'attente aléatoire
                writer.writerow([airport_code, waiting_time])

print(f"Le fichier '{output_file}' a été généré avec succès !")

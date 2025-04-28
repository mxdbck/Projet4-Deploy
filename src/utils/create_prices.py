import csv
import random

# Fichiers d'entrée et de sortie
input_file = "data/pre_existing_routes.csv"
output_file = "data/prices.csv"

# Lecture du fichier CSV et écriture du nouveau fichier avec les prix
with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Sauter l'en-tête si nécessaire

    with open(output_file, mode='w', newline='', encoding='utf-8') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(["ID_start","ID_end","price"])  # En-tête du nouveau fichier

        for row in reader:
            if len(row) >= 2:  # Vérifier que la ligne a au moins 2 colonnes
                airport1_code = row[0]  # 1ème colonne (index 0)
                airport2_code = row[1]  # 2ème colonne (index 1)
                price = random.randint(50, 400)  # Prix aléatoire
                writer.writerow([airport1_code, airport2_code, price])

print(f"Le fichier '{output_file}' a été généré avec succès !")

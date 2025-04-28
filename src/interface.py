import streamlit as st
import pydeck as pdk
import pandas as pd
from utils.import_csv import import_csv
import algos.proprietes_du_trajet as algos
import numpy as np


# pour run :
# streamlit run src/interface.py (dans le terminal)

def format_minutes(time) :
    string_time = [0,0,0]
    for i in range(3):
        hours = int(time[i] //60)
        minutes = time[i] % 60
        minutes = round( minutes / 5) * 5
        if minutes == 60 :
            minutes = 0
            hours += 1
        if minutes<10 :
            string_time[i] = f"{hours}h0{minutes}"
        else :
            string_time[i] = f"{hours}h{minutes}"
    return string_time





# configurarion de la page
st.set_page_config(page_title="AirPL", page_icon=":airplane:", layout="wide", initial_sidebar_state="auto", menu_items=None)



# importation des données
problem_instance  = import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys.csv", "data/prices.csv", "data/waiting_times.csv", 3500.0)
# Dictionnaire qui permet de convertir les id des aéroports en string
airport_options = {airp.id: f"{airp.id}" for airp in problem_instance.all_airports}
airport_options = dict(sorted(airport_options.items(), key=lambda item: item[1]))


# preparation des structures
binary_selection_time = [False for _ in range(len(problem_instance.all_connexions))]
binary_selection_distance = [False for _ in range(len(problem_instance.all_connexions))]
binary_selection_price = [False for _ in range(len(problem_instance.all_connexions))]
selection_time = [False for _ in range(len(problem_instance.all_connexions))]
selection_price = [False for _ in range(len(problem_instance.all_connexions))]
selection_distance = [False for _ in range(len(problem_instance.all_connexions))]

# SIDEBAR
# interface utilisateur pour selectionner son trajet
with st.sidebar:
    st.markdown('##  :airplane: :cloud: AirPL ')
    st.markdown('##### Votre outil de planification de trajets aériens')
    st.markdown("---")

    source_airport_id = st.selectbox(
        "Départ :",
        options=list(airport_options.keys()),
        format_func=lambda x: airport_options[x],
        placeholder= "Choisir un aéroport")

    st.write(" ")

    destination_airport_id = st.selectbox(
        "Arrivée :",
        options=list(airport_options.keys()),
        format_func=lambda x: airport_options[x],
        placeholder= "Choisir un aéroport")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        aller_retour = st.checkbox("Aller-Retour")


    with col2:
        searched = st.button ("Rechercher",icon=":material/search:")

    st.markdown("---")



# Affichage par defaut
if not searched:
    st.markdown('## Bienvenue')
    st.markdown('##### Pour commencer, veuillez choisir un aéroport de départ et un aéroport d\'arrivée.')
    st.write("")


# Affichage si recherche lancee
# Gestion des problemes avec l'input
error = True
if searched:
    if source_airport_id == destination_airport_id:
        st.markdown('## Bienvenue')
        st.error("Veuillez choisir deux aéroports différents.")
    elif source_airport_id == None and destination_airport_id == None:
        st.markdown('## Bienvenue :airplane: :cloud:')
        st.error("Veuillez choisir deux aéroports.")
    elif source_airport_id == None:
        st.markdown('## Bienvenue :airplane: :cloud:')
        st.error("Veuillez choisir un aéroport de départ.")
    elif destination_airport_id == None:
        st.markdown('## Bienvenue :airplane: :cloud:')
        st.error("Veuillez choisir un aéroport d'arrivée.")

    else :
        st.markdown('## Propositions de trajets')
        st.markdown(f"#### {source_airport_id} - {destination_airport_id}")
        st.write("")
        error = False




if aller_retour and not error :
    ar_binary_selection_time = [False for _ in range(len(problem_instance.all_connexions))]
    ar_binary_selection_distance = [False for _ in range(len(problem_instance.all_connexions))]
    ar_binary_selection_price = [False for _ in range(len(problem_instance.all_connexions))]
    ar_selection_time = [False for _ in range(len(problem_instance.all_connexions))]
    ar_selection_price = [False for _ in range(len(problem_instance.all_connexions))]
    ar_selection_distance = [False for _ in range(len(problem_instance.all_connexions))]

# recherche des trajets optimaux
if searched and not error:

    solution = pd.DataFrame()

    Categories = ["🏁 Meilleure Distance", "💸 Meilleur Prix", "⏱️ Meilleur Temps"]
    solution.index = Categories
    path, escales, distances, prices, temps, nombre_escales, path_binaires = algos.find_proprietes_du_trajet(problem_instance,source_airport_id , destination_airport_id)
    if escales == None:
        st.error("Aucune liaison n'existe pour ce trajet.")
        searched =  False
        ar_searched = False
    # On adapte l'affichage si aller retour
    elif aller_retour :
        ar_searched = True
        st.write("##### Trajet Aller :")

# si il existe un trajet et la recherche est lancée
if searched and not error:
    solution["Durée"] = format_minutes(temps)
    solution["Distance Totale"] = [f"{round(distance / 5) * 5} km "for distance in distances]
    solution["Prix Total"] = [ f"{round(price)} PLN" for price in prices]
    solution["Nombre d'escales"] = nombre_escales
    solution["Escales"] = escales

    st.write(solution)
    st.write("")

    binary_selection_distance = path_binaires[0]
    binary_selection_price =  path_binaires[1]
    binary_selection_time  = path_binaires[2]


sample_connexions_time = [problem_instance.all_connexions[i] for i in range(len(problem_instance.all_connexions)) if binary_selection_time[i]]
sample_connexions_price = [problem_instance.all_connexions[i] for i in range(len(problem_instance.all_connexions)) if binary_selection_price[i]]
sample_connexions_distance = [problem_instance.all_connexions[i] for i in range(len(problem_instance.all_connexions)) if binary_selection_distance[i]]



# Gestion du trajet retour si l'aller existe et la recherche est lancée
if searched and aller_retour and not error :
    st.write("##### Trajet Retour :")

    ar_solution = pd.DataFrame()

    ar_Categories = ["🏁 Meilleure Distance", "💸 Meilleur Prix", "⏱️ Meilleur Temps"]
    ar_solution.index = ar_Categories
    ar_path, ar_escales, ar_distances, ar_prices, ar_temps, ar_nombre_escales, ar_path_binaires = algos.find_proprietes_du_trajet( problem_instance ,destination_airport_id, source_airport_id )
    ar_path_binaires_array = np.array(ar_path_binaires)
    path_binaires_array = np.array(path_binaires)
    print (np.array_equal(path_binaires_array, ar_path_binaires_array))
    if ar_escales == None:
        st.error("Aucune liaison n'existe pour le trajet retour.")
        ar_searched =  False
    else :
        ar_solution["Durée"] = format_minutes(ar_temps)
        ar_solution["Distance Totale"] = [f"{round(ar_distance / 5) * 5} km "for ar_distance in ar_distances]
        ar_solution["Prix Total"] = [ f"{round(ar_price)} PLN" for ar_price in ar_prices]
        ar_solution["Nombre d'escales"] = ar_nombre_escales
        ar_solution["Escales"] = ar_escales


        st.write(ar_solution)
        st.write("")

        ar_binary_selection_distance = ar_path_binaires[0]
        ar_binary_selection_price =  ar_path_binaires[1]
        ar_binary_selection_time  = ar_path_binaires[2]

    ar_sample_connexions_time = [problem_instance.all_connexions[i] for i in range(len(problem_instance.all_connexions)) if ar_binary_selection_time[i]]
    ar_sample_connexions_price = [problem_instance.all_connexions[i] for i in range(len(problem_instance.all_connexions)) if ar_binary_selection_price[i]]
    ar_sample_connexions_distance = [problem_instance.all_connexions[i] for i in range(len(problem_instance.all_connexions)) if ar_binary_selection_distance[i]]



st.write("---")
st.write("##### Affichage carte :")

# Affichage des aéroports sur la carte :
# Convert airports to DataFrame
airport_data = pd.DataFrame(
    {
        "id": [airp.id for airp in problem_instance.all_airports],
        "latitude": [airp.location[0] for airp in problem_instance.all_airports],
        "longitude": [airp.location[1] for airp in problem_instance.all_airports],
    }
)
# Convert connections to ArcLayer format
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    airport_data,
    get_position=["longitude", "latitude"],
    get_radius=30000,
    get_color = [2, 15, 32, 240],
    pickable=True,
)
# Create pydeck map
view_state = pdk.ViewState(latitude=37.5, longitude=-100, zoom=3)

if not searched or error :
        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[
                scatter_layer
            ],
            initial_view_state=view_state,
            tooltip={"text": "Airport: {id}"}
        )
        st.write("")

        st.pydeck_chart(deck)


# --- Légende HTML avec tes couleurs ---
legend_html = """
    <div style="position:relative; padding:10px; background-color:white; border-radius:10px;">
        <b>Légende des trajets :</b><br>
        <div style="display:flex; align-items:center;">
            <div style="width:15px; height:15px; background-color:#2E8B57; margin-right:5px;"></div> Meilleure Distance
        </div>
        <div style="display:flex; align-items:center;">
            <div style="width:15px; height:15px; background-color:#E3A72F; margin-right:5px;"></div> Meilleur Prix
        </div>
        <div style="display:flex; align-items:center;">
            <div style="width:15px; height:15px; background-color:#FF7F50; margin-right:5px;"></div> Meilleur Temps
        </div>


    </div>
    """


distances_RGB = [82, 183, 136, 255]  # Vert clair doux (au lieu de vert émeraude)
time_RGB = [255, 145, 77, 255]  # Orange chaud (au lieu d’orange vif)
prices_RGB = [244, 196, 48, 255]  # Jaune safran (au lieu de jaune doré)


if not aller_retour :
    # Affichage des trajets sur la carte :
    # Affichage tout en une colonne
    if searched and not error:

               # Jaune moutarde

        # Time
        path_time = []
        for start_id, end_id in sample_connexions_time:
            start_airport = problem_instance.airp_from_id(start_id)
            end_airport = problem_instance.airp_from_id(end_id)
            path_time.append({
                "start_lat": start_airport.location[0],
                "start_lon": start_airport.location[1],
                "end_lat": end_airport.location[0],
                "end_lon": end_airport.location[1],
            })

        path_layer_time = pdk.Layer(
            "ArcLayer",
            path_time,
            get_name="Meilleur Temps",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_width=2,
            get_height=0.25,
            get_source_color= time_RGB,
            get_target_color= time_RGB,
            pickable=True,
            auto_highlight=True,
        )

        # Price
        path_price = []
        for start_id, end_id in sample_connexions_price:
            start_airport = problem_instance.airp_from_id(start_id)
            end_airport = problem_instance.airp_from_id(end_id)
            path_price.append({
                "start_lat": start_airport.location[0],
                "start_lon": start_airport.location[1],
                "end_lat": end_airport.location[0],
                "end_lon": end_airport.location[1],
            })

        path_layer_price = pdk.Layer(
            "ArcLayer",
            path_price,
            get_name="Meilleur Prix",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_width=2,
            get_height=0.25,
            get_source_color= prices_RGB,
            get_target_color= prices_RGB,
            pickable=True,
            auto_highlight=True,
        )

        # Distance
        path_distance = []
        for start_id, end_id in sample_connexions_distance:
            start_airport = problem_instance.airp_from_id(start_id)
            end_airport = problem_instance.airp_from_id(end_id)
            path_distance.append({
                "start_lat": start_airport.location[0],
                "start_lon": start_airport.location[1],
                "end_lat": end_airport.location[0],
                "end_lon": end_airport.location[1],
            })

        path_layer_distance = pdk.Layer(
            "ArcLayer",
            path_distance,
            get_name="Meilleure Distance",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_width=2,
            get_height=0.25,
            get_source_color= distances_RGB,
            get_target_color= distances_RGB,
            pickable=True,
            auto_highlight=True,
        )


        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[
                scatter_layer,
                path_layer_price,
                path_layer_time,
                path_layer_distance
            ],
            initial_view_state=view_state,
            tooltip={"text": "Airport: {id}"}
        )

        st.write("")

        st.pydeck_chart(deck)







if searched and aller_retour and not error:

    cola, colr = st.columns(2)

    with cola:
        st.write("###### Aller :")
        # Time
        path_time = []
        for start_id, end_id in sample_connexions_time:
            start_airport = problem_instance.airp_from_id(start_id)
            end_airport = problem_instance.airp_from_id(end_id)
            path_time.append({
                "start_lat": start_airport.location[0],
                "start_lon": start_airport.location[1],
                "end_lat": end_airport.location[0],
                "end_lon": end_airport.location[1],
            })


        path_layer_time = pdk.Layer(
            "ArcLayer",
            path_time,
            get_name="Meilleur Temps",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_width=2,
            get_height=0.25,
            get_source_color= time_RGB,
            get_target_color= time_RGB,
            pickable=True,
            auto_highlight=True,
        )

        # Price
        path_price = []
        for start_id, end_id in sample_connexions_price:
            start_airport = problem_instance.airp_from_id(start_id)
            end_airport = problem_instance.airp_from_id(end_id)
            path_price.append({
                "start_lat": start_airport.location[0],
                "start_lon": start_airport.location[1],
                "end_lat": end_airport.location[0],
                "end_lon": end_airport.location[1],
            })


        path_layer_price = pdk.Layer(
            "ArcLayer",
            path_price,
            get_name="Meilleur Prix",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_width=2,
            get_height=0.25,
            get_source_color= prices_RGB,
            get_target_color= prices_RGB,
            pickable=True,
            auto_highlight=True,
        )

        # Distance
        path_distance = []
        for start_id, end_id in sample_connexions_distance:
            start_airport = problem_instance.airp_from_id(start_id)
            end_airport = problem_instance.airp_from_id(end_id)
            path_distance.append({
                "start_lat": start_airport.location[0],
                "start_lon": start_airport.location[1],
                "end_lat": end_airport.location[0],
                "end_lon": end_airport.location[1],
            })

        path_layer_distance = pdk.Layer(
            "ArcLayer",
            path_distance,
            get_name="Meilleure Distance",
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_width=2,
            get_height=0.25,
            get_source_color= distances_RGB,
            get_target_color= distances_RGB,
            pickable=True,
            auto_highlight=True,
        )


        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[
                scatter_layer,
                path_layer_price,
                path_layer_time,
                path_layer_distance
            ],
            initial_view_state=view_state,
            tooltip={"text": "Airport: {id}"}
        )
        st.pydeck_chart(deck)



    if not ar_searched :
        with colr :
            st.write("###### Retour :")
            deck = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                layers=[
                    scatter_layer
                ],
                initial_view_state=view_state,
                tooltip={"text": "Airport: {id}"}
            )
            st.pydeck_chart(deck)

    if ar_searched :
        with colr :
            st.write("###### Retour :")
            # Time
            path_time = []
            for start_id, end_id in ar_sample_connexions_time:
                start_airport = problem_instance.airp_from_id(start_id)
                end_airport = problem_instance.airp_from_id(end_id)
                path_time.append({
                    "start_lat": start_airport.location[0],
                    "start_lon": start_airport.location[1],
                    "end_lat": end_airport.location[0],
                    "end_lon": end_airport.location[1],
                })


            ar_path_layer_time = pdk.Layer(
                "ArcLayer",
                path_time,
                get_name="Meilleur Temps",
                get_source_position=["start_lon", "start_lat"],
                get_target_position=["end_lon", "end_lat"],
                get_width=2,
                get_height=0.25,
                get_source_color= time_RGB,
                get_target_color= time_RGB,
                pickable=True,
                auto_highlight=True,
            )

            # Price
            path_price = []
            for start_id, end_id in ar_sample_connexions_price:
                start_airport = problem_instance.airp_from_id(start_id)
                end_airport = problem_instance.airp_from_id(end_id)
                path_price.append({
                    "start_lat": start_airport.location[0],
                    "start_lon": start_airport.location[1],
                    "end_lat": end_airport.location[0],
                    "end_lon": end_airport.location[1],
                })


            ar_path_layer_price = pdk.Layer(
                "ArcLayer",
                path_price,
                get_name="Meilleur Prix",
                get_source_position=["start_lon", "start_lat"],
                get_target_position=["end_lon", "end_lat"],
                get_width=2,
                get_height=0.25,
                get_source_color= prices_RGB,
                get_target_color= prices_RGB,
                pickable=True,
                auto_highlight=True,
            )

            # Distance
            path_distance = []
            for start_id, end_id in ar_sample_connexions_distance:
                start_airport = problem_instance.airp_from_id(start_id)
                end_airport = problem_instance.airp_from_id(end_id)
                path_distance.append({
                    "start_lat": start_airport.location[0],
                    "start_lon": start_airport.location[1],
                    "end_lat": end_airport.location[0],
                    "end_lon": end_airport.location[1],
                })

            ar_path_layer_distance = pdk.Layer(
                "ArcLayer",
                path_distance,
                get_name="Meilleure Distance",
                get_source_position=["start_lon", "start_lat"],
                get_target_position=["end_lon", "end_lat"],
                get_width=2,
                get_height=0.25,
                get_source_color= distances_RGB,
                get_target_color= distances_RGB,
                pickable=True,
                auto_highlight=True,
            )


            deck = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                layers=[
                    scatter_layer,
                    ar_path_layer_price,
                    ar_path_layer_time,
                    ar_path_layer_distance
                ],
                initial_view_state=view_state,
                tooltip={"text": "Airport: {id}"}
            )
            st.pydeck_chart(deck)


if searched and not error :
    st.markdown(legend_html, unsafe_allow_html=True)

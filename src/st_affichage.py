import streamlit as st
import pydeck as pdk
import pandas as pd
from utils.import_csv import import_csv
from algos.union_dijkstra import union_dijkstra, union_spanner, t_spanner_full, union_spanner_search
import numpy as np

# pour run (dans le terminal) :
# streamlit run src/st_affichage.py

# configurarion de la page
st.set_page_config(page_title="Airport Visualisation", page_icon=":eye:", layout="wide", initial_sidebar_state="auto", menu_items=None)
# Streamlit App
st.title("Airport Connections Visualization")


# Options disponibles
options = ["t-spanner", "union-dijkstra", "union-spanner", "union-spanner-search", "all"]

# Valeur par défaut
algo = st.selectbox("Choisir l'algo :", options, index=options.index("union-spanner"))


if not st.button("Valider"):
    st.warning("Clique sur 'Valider' pour continuer 🚀")
    st.stop()


st.success(f"Algorithme sélectionné : **{algo}**")

# commande `pip install streamlit` pour lancer l'application

# importation des données
prob  = import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys.csv", "data/prices.csv", "data/waiting_times.csv", 3500.0)
airport_options = {airp.id: f"{airp.id}" for airp in prob.all_airports}
airport_options = dict(sorted(airport_options.items(), key=lambda item: item[1]))



# quel algo tu veux ???
if algo == "t-spanner":
    routes = t_spanner_full(prob, t=1.1 + 0.25*3500.0 / 10_000.0)
elif algo == "union-dijkstra":
    routes = union_dijkstra(prob, 0)
elif algo == "union-spanner" :
    routes = union_spanner(prob, 1.1 + 0.25*3500.0 / 10_000.0 )
elif algo == "union-spanner-search" :
    routes = union_spanner_search(prob, 1.1 + 0.25*3500.0 / 10_000.0 )
elif algo == "all":
    routes = list(np.ones(len(prob.all_connexions)))
elif algo == "genetic":
    routes_df = pd.read_csv("resultat.csv", header=None, sep=',')
    print(routes_df)
    routes_df = routes_df.stack().reset_index(drop=True)
    routes = routes_df.values.tolist()
    print(len(routes))


# sample_connexions = random.sample(problem_instance.all_connexions, 10)

# Filter connections based on binary selection

sample_connexions = [prob.all_connexions[i] for i in range(len(prob.all_connexions)) if routes[i]]



# Convert airports to DataFrame
airport_data = pd.DataFrame(
    {
        "id": [airp.id for airp in prob.all_airports],
        "latitude": [airp.location[0] for airp in prob.all_airports],
        "longitude": [airp.location[1] for airp in prob.all_airports],
    }
)


# if st.button("Randomize Connections"):
#     sample_connexions = random.sample(problem_instance.all_connexions, min(10, len(problem_instance.all_connexions)))

# Convert connections to ArcLayer format
arcs = []
for start_id, end_id in sample_connexions:
    start_airport = prob.airp_from_id(start_id)
    end_airport = prob.airp_from_id(end_id)
    arcs.append({
        "start_lat": start_airport.location[0],
        "start_lon": start_airport.location[1],
        "end_lat": end_airport.location[0],
        "end_lon": end_airport.location[1],
    })

# arcs.append({
#     "start_lat": problem_instance.airp_from_id(source_airport_id).location[0],
#     "start_lon": problem_instance.airp_from_id(source_airport_id).location[1],
#     "end_lat": problem_instance.airp_from_id(destination_airport_id).location[0],
#     "end_lon": problem_instance.airp_from_id(destination_airport_id).location[1],
# })


GREEN_RGB = [128, 0, 128, 40]
RED_RGB = [65, 105, 225, 40]

arc_layer = pdk.Layer(
    "ArcLayer",
    arcs,
    get_source_position=["start_lon", "start_lat"],
    get_target_position=["end_lon", "end_lat"],
    get_width=2,
    # get_tilt=90,
    get_height=0.25,
    get_source_color=GREEN_RGB,
    get_target_color=RED_RGB,
    pickable=True,
    auto_highlight=True,
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    airport_data,
    get_position=["longitude", "latitude"],
    get_radius=50000,
    get_color=[0, 0, 255, 255],
    pickable=True,
)

# Create pydeck map
view_state = pdk.ViewState(latitude=37.5, longitude=-100, zoom=3)

deck = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    layers=[
        scatter_layer,
        arc_layer
    ],
    initial_view_state=view_state,
    tooltip={"text": "Airport: {id}"}
)

st.pydeck_chart(deck)

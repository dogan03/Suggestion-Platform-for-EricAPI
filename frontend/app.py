import random

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = "http://0.0.0.0:8000"
if "start_year" not in st.session_state:
    st.session_state.start_year = 1950
if "end_year" not in st.session_state:
    st.session_state.end_year = 2023
if "liked_ids" not in st.session_state:
    st.session_state.liked_ids = []
if "liked_papers" not in st.session_state:
    st.session_state.liked_papers = []
if "subjects_list" not in st.session_state:
    with open("subjects.txt", "r", encoding="utf-8") as f:
        st.session_state.subjects_list = f.readlines()
        st.session_state.subjects_list = [subject.strip() for subject in st.session_state.subjects_list]
st.session_state.start_year = str(st.sidebar.selectbox("Start Year", options=range(1950, 2024), index=0))
st.session_state.end_year = str(st.sidebar.selectbox("End Year", options=range(1950, 2024), index=73))
def removeLikedPapers():
    st.session_state.liked_ids = []
    st.session_state.liked_papers = []
def runSuggestionsAlgorithm():
    trainModel()
    st.success("Suggestion Algorithm has been run successfully")
def toggle_favorite(paper_id, paper):
    if paper_id in st.session_state.liked_ids:
        st.session_state.liked_ids.remove(paper_id) if paper_id in st.session_state.liked_ids else None
        st.session_state.liked_papers.remove(paper) if paper in st.session_state.liked_papers else None
    else:
        st.session_state.liked_ids.append(paper_id) if paper_id not in st.session_state.liked_ids else None
        st.session_state.liked_papers.append(paper) if paper not in st.session_state.liked_papers else None
def fetchData(skip, limit, start_year, end_year):
    url = f"{API_BASE_URL}/data"
    response = requests.get(
        url,
        params={
            "skip": skip,
            "limit": limit,
            "start_year": st.session_state.start_year,
            "end_year": st.session_state.end_year
        }
    )
    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        st.error("Failed to fetch data")
        st.stop() 
def fetch_subject_counts(subject_list):
    url = f"{API_BASE_URL}/count"
    response = requests.post(
        url,
        json={
            "subject_list": subject_list,
            "start_year": st.session_state.start_year,
            "end_year": st.session_state.end_year
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch subject counts")
        st.stop()  # Stop execution if count fetching fails
def fetchSuggestions(skip, limit):
    url = f"{API_BASE_URL}/getSuggestions"
    response = requests.get(
        url,
        params={
            "skip": skip,
            "limit": limit,
            "start_year": st.session_state.start_year,
            "end_year": st.session_state.end_year
        }
    )
    if response.status_code == 200:
        data = response.json()["data"]
        return data
    else:
        st.error("Failed to fetch data")
        st.stop()  
def fetchClusterSimilarities(subject_list, n_paper):
    st.write(subject_list)
    url = f"{API_BASE_URL}/cluster"
    response = requests.post(
        url,
        json={
            "topics": subject_list,
            "n_paper": n_paper,
            "start_year": st.session_state.start_year,
            "end_year": st.session_state.end_year
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch subject counts")
        st.stop()  


def plot_paper_topic_network(paper_similarities, n_iter):
    G = nx.Graph()

    for paper, topics in paper_similarities.items():
        title, author = paper.split("###")
        G.add_node(paper, node_type='paper', similarities=topics, title=title, author=author)
        for topic, weight in topics.items():
            G.add_node(topic, node_type='topic')
            G.add_edge(paper, topic, weight=weight)
    
    all_topics = list(set(topic for topics in paper_similarities.values() for topic in topics))
    for topic1 in all_topics:
        for topic2 in all_topics:
            if topic1 != topic2:
                G.add_edge(topic1, topic2, weight=-1000)

    def create_plotly_graph(G, paper_nodes, topic_nodes):
        pos = nx.spring_layout(G, iterations=n_iter, k=0.5/np.sqrt(len(G.nodes())), center=(0, 0))

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        hover_text = []
        node_color = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append('lightblue' if G.nodes[node]['node_type'] == 'paper' else 'red')
            if G.nodes[node]['node_type'] == 'paper':
                title = G.nodes[node]['title']
                author = G.nodes[node]['author']
                # Add author text above paper node
                author_text = f"<b>{author}</b>"
                hover_text.append(f"{title}<br>Author: {author}<br>({G.nodes[node]['similarities']})")
                node_text.append(author_text)
            else:
                node_text.append(node)
                hover_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                color=node_color,
                size=20,
                line_width=2
            )
        )
        
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        node_trace.marker.color = node_adjacencies
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Network Graph Visualization',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[],
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False))
                    )
        return fig


    paper_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'paper']
    topic_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'topic']

    fig = create_plotly_graph(G, paper_nodes, topic_nodes)
    
    st.plotly_chart(fig, use_container_width=True)







def trainModel():
    url = f"{API_BASE_URL}/trainModel"
    response = requests.post(
        url,
        json = {
            "liked": st.session_state.liked_ids
        }
    )

def topPapers():
    st.markdown(
        "<h1 style='text-align: center;'>Top Papers</h1>", unsafe_allow_html=True
    )
    
    
    # Paginate the data
    page_num = st.sidebar.number_input("Page", min_value=1, value=1)
    items_per_page = 20
    skip = (page_num - 1) * items_per_page
    
    data = fetchData(skip, items_per_page, st.session_state.start_year, st.session_state.end_year)
    
    for paper in data:
        if paper["id"] in st.session_state.liked_ids:
            heart_color = "red"
        else:
            heart_color = "white"
        heart_button = st.button(
            f"❤️" if heart_color == "red" else "🤍",
            key=f"heart_{paper['id']}",
            on_click=toggle_favorite,
            args=(paper["id"], paper)
        )
        st.markdown(
            f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <h2 style='text-align: center; font-weight: bold;'>{paper['title']}</h2>
                <p style='text-align: center; font-size: 1.2em;'>{int(paper['publication_year'])}</p>
                <p style='text-align: center; font-size: 1.1em;'>{paper['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        subject_counts = fetch_subject_counts(paper['subject'].split(","))

        st.subheader("Subject Bar Chart")
        st.bar_chart(subject_counts)
        st.write("---")

def suggested_papers():
    st.markdown(
        "<h1 style='text-align: center;'>Suggested Papers</h1>", unsafe_allow_html=True
    )
    st.button("Run Suggestion Algorithm", on_click=runSuggestionsAlgorithm)
    
    page_num = st.sidebar.number_input("Page", min_value=1, value=1)
    items_per_page = 20
    skip = (page_num - 1) * items_per_page
    
    data = fetchSuggestions(skip, items_per_page)
    
    for paper in data:
        if paper["id"] in st.session_state.liked_ids:
            heart_color = "red"
        else:
            heart_color = "white"
        heart_button = st.button(
            f"❤️" if heart_color == "red" else "🤍",
            key=f"heart_{paper['id']}",
            on_click=toggle_favorite,
            args=(paper["id"], paper)
        )
        st.markdown(
            f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <h2 style='text-align: center; font-weight: bold;'>{paper['title']}</h2>
                <p style='text-align: center; font-size: 1.2em;'>{int(paper['publication_year'])}</p>
                <p style='text-align: center; font-size: 1.0em;'>Score: {paper['score']}</p>
                <p style='text-align: center; font-size: 1.1em;'>{paper['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        subject_counts = fetch_subject_counts(paper['subject'].split(","))

        st.subheader("Subject Bar Chart")
        st.bar_chart(subject_counts)
        st.write("---")
def liked_papers():
    st.markdown(
        "<h1 style='text-align: center;'>Liked Papers</h1>", unsafe_allow_html=True
    )
    st.button("Remove All Liked Papers", on_click=removeLikedPapers)
    
    
    page_num = st.sidebar.number_input("Page", min_value=1, value=1)
    
    for paper in st.session_state.liked_papers:
        if paper["id"] in st.session_state.liked_ids:
            heart_color = "red"
        else:
            heart_color = "white"
        heart_button = st.button(
            f"❤️" if heart_color == "red" else "🤍",
            key=f"heart_{paper['id']}",
            on_click=toggle_favorite,
            args=(paper["id"], paper)
        )
        st.markdown(
            f"""
            <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <h2 style='text-align: center; font-weight: bold;'>{paper['title']}</h2>
                <p style='text-align: center; font-size: 1.2em;'>{int(paper['publication_year'])}</p>
                <p style='text-align: center; font-size: 1.1em;'>{paper['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        subject_counts = fetch_subject_counts(paper['subject'].split(","))

        st.subheader("Subject Bar Chart")
        st.bar_chart(subject_counts)
        st.write("---")

def clusterPapers():
    st.markdown("<h1 style='text-align: center;'>Cluster Papers</h1>", unsafe_allow_html=True)

    selected_topics = st.multiselect("Select topics", st.session_state.subjects_list)
    
    n_iter = st.number_input("Number of iterations", min_value=10, max_value=1000000, value=1000, step=10)
    n_paper = st.number_input("Number of papers", min_value=1, max_value=1000000, value=5, step=1)

    if st.button("Cluster"):
        cluster_similarities = fetchClusterSimilarities(selected_topics, n_paper)
        st.success("Clustering completed!")  
        st.write(cluster_similarities)
        plot_paper_topic_network(cluster_similarities, n_iter)
        st.success("Clustering completed!")  

# Navigation
pages = {
    "Top Papers": topPapers,
    "Suggested Papers": suggested_papers,
    "Liked Papers": liked_papers,
    "Cluster Papers": clusterPapers,
}

selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page()

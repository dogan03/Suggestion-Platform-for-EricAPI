import numpy as np
import pandas as pd
import requests
import streamlit as st


def getSubjectCounts(subjects):
    subjects = subjects.split(", ")
    url = "http://0.0.0.0:8000/get_frequencies"
    response = requests.post(
        url,
        json={
            "start_date": st.session_state.start_year,
            "end_date": st.session_state.end_year,
        },
    )
    if response.status_code == 200:
        subject_counts = response.json()
    else:
        st.error("Failed to fetch data")
        st.stop()  # Stop execution if data fetching fails
    res = {subject: subject_counts.get(subject, 0) for subject in subjects}
    return res


def load_all_data():
    url = "http://0.0.0.0:8000/data"
    response = requests.get(url)
    if response.status_code == 200:
        st.session_state["all_data"] = response.json()
    else:
        st.error("Failed to fetch data")
        st.stop()  # Stop execution if data fetching fails


if "all_data" not in st.session_state:
    load_all_data()


st.session_state.data = st.session_state["all_data"]
st.session_state.title_list = st.session_state.data["title"]
st.session_state.description_list = st.session_state.data["description"]
st.session_state.subject_list = st.session_state.data["subject"]
st.session_state.year_list = st.session_state.data["year"]
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
    st.session_state.non_favorites = set()
if "train_svm" not in st.session_state:
    st.session_state.train_svm = 0
if "redesign_for_query" not in st.session_state:
    st.session_state.redesign_for_query = 0


def toggle_favorite(index):
    st.session_state.train_svm = 1
    if index in st.session_state.favorites:
        st.session_state.favorites.discard(index)
    else:
        st.session_state.favorites.add(index)
        for idx in range(len(st.session_state.title_list)):
            if idx not in st.session_state.favorites:
                st.session_state.non_favorites.add(idx)


def year_change():
    st.session_state.do_sort = 1


def top_papers():
    st.markdown(
        "<h1 style='text-align: center;'>Top Papers</h1>", unsafe_allow_html=True
    )
    st.session_state.start_year = st.sidebar.selectbox(
        "Start Year",
        options=range(1980, 2024),
        index=0,
        on_change=year_change,
        key="start_year_selectbox",
    )
    st.session_state.end_year = st.sidebar.selectbox(
        "End Year",
        options=range(1980, 2024),
        index=40,
        on_change=year_change,
        key="end_year_selectbox",
    )
    if "do_sort" not in st.session_state:
        st.session_state.do_sort = 1
    if st.session_state.do_sort == 1:
        st.session_state.sorted_indices = sorted(
            (
                idx
                for idx in range(len(st.session_state.data["title"]))
                if st.session_state.start_year
                <= st.session_state.year_list[idx]
                <= st.session_state.end_year
            ),
            key=lambda i: sum(
                getSubjectCounts(st.session_state.data["subject"][i]).values()
            ),
            reverse=True,
        )
        st.session_state.do_sort = 0

    # Paginate the data
    page_num_key_top = "page_num_input_top"  # Unique key for the number input widget
    page_num = st.sidebar.number_input(
        "Page", min_value=1, value=1, key=page_num_key_top
    )

    items_per_page = 20
    if st.session_state.redesign_for_query == 1:
        st.session_state.title_list = st.session_state.queried_data["title"]
        st.session_state.subject_list = st.session_state.queried_data["subject"]
        st.session_state.description_list = st.session_state.queried_data["description"]
        st.session_state.year_list = st.session_state.queried_data[
            "publicationdateyear"
        ]
        st.session_state.distance_list = st.session_state.queried_data["distance"]

    start_idx = (page_num - 1) * items_per_page
    if st.session_state.redesign_for_query == 1:
        end_idx = min(len(st.session_state.distance_list), start_idx + items_per_page)
    else:
        end_idx = min(len(st.session_state.sorted_indices), start_idx + items_per_page)

    for idx in range(start_idx, end_idx):
        if st.session_state.redesign_for_query == 0:
            i = st.session_state.sorted_indices[idx]
        else:
            i = idx
        if (
            st.session_state.start_year
            <= st.session_state.year_list[i]
            <= st.session_state.end_year
        ):
            if i in st.session_state.favorites:
                heart_color = "red"
            else:
                heart_color = "gray"

            heart_button = st.button(
                f"❤️" if heart_color == "red" else "🤍",
                key=f"heart_{i}_top",
                on_click=toggle_favorite,
                args=(
                    (
                        st.session_state.data["title"].index(
                            st.session_state.queried_data["title"][i]
                        )
                        if st.session_state.redesign_for_query
                        else i
                    ),
                ),
            )
            if st.session_state.redesign_for_query == 1:

                st.markdown(
                    f"""
                    <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <h2 style='text-align: center; font-weight: bold;'>{st.session_state.title_list[i]}</h2>
                        <p style='text-align: center; font-size: 1.2em;'>{int(st.session_state.year_list[i])}</p>
                        <p style='text-align: center; font-size: 1.2em;'>Similarity: {st.session_state.distance_list[i]}</p>
                        <p style='text-align: center; font-size: 1.1em;'>{st.session_state.description_list[i]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <h2 style='text-align: center; font-weight: bold;'>{st.session_state.title_list[i]}</h2>
                        <p style='text-align: center; font-size: 1.2em;'>{int(st.session_state.year_list[i])}</p>
                        <p style='text-align: center; font-size: 1.1em;'>{st.session_state.description_list[i]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Display bar chart for subject
            st.subheader("Subject Bar Chart")
            counts = getSubjectCounts(st.session_state.subject_list[i])
            plot_subject(counts)
            st.write("---")


def plot_subject(counts):
    st.bar_chart(counts)


# Liked Papers Page
def liked_papers():

    st.title("Liked Papers")
    page_num_liked = "page_num_liked"  # Unique key for the number input widget
    page_num = st.sidebar.number_input("Page", min_value=1, value=1, key=page_num_liked)
    items_per_page = 20
    start_idx = (page_num - 1) * items_per_page
    end_idx = min(len(list(st.session_state.favorites)), start_idx + items_per_page)
    for idx in range(start_idx, end_idx):
        i = list(st.session_state.favorites)[idx]
        if (
            st.session_state.start_year
            <= st.session_state.year_list[i]
            <= st.session_state.end_year
        ):
            if i in st.session_state.favorites:
                heart_color = "red"
            else:
                heart_color = "gray"

            heart_button = st.button(
                f"❤️" if heart_color == "red" else "🤍",
                key=f"heart_{i}_liked",
                on_click=toggle_favorite,
                args=(i,),
            )

            st.markdown(
                f"""
                <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <h2 style='text-align: center; font-weight: bold;'>{st.session_state.title_list[i]}</h2>
                    <p style='text-align: center; font-size: 1.2em;'>{int(st.session_state.year_list[i])}</p>
                    <p style='text-align: center; font-size: 1.1em;'>{st.session_state.description_list[i]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display bar chart for subject
            st.subheader("Subject Bar Chart")
            counts = getSubjectCounts(st.session_state.subject_list[i])
            plot_subject(counts)
            st.write("---")


# Suggested Papers Page
def suggested_papers():
    st.title("Suggested Papers")
    if st.session_state.train_svm == 1:

        if st.session_state.favorites:
            trainSVM(
                list(st.session_state.favorites), list(st.session_state.non_favorites)
            )
        else:
            st.error("Failed to fetch embedding data for some papers.")
        sugg_dict = {}
        for idx in range(len(st.session_state.year_list)):
            sugg_dict[idx] = getLogit(idx)
        st.session_state.suggested_indices = sorted(
            sugg_dict.items(), key=lambda item: item[1], reverse=True
        )
        # print(st.session_state.suggested_indices)
    page_num_liked = "page_num_suggested"  # Unique key for the number input widget
    page_num = st.sidebar.number_input("Page", min_value=1, value=1, key=page_num_liked)
    items_per_page = 20
    start_idx = (page_num - 1) * items_per_page
    end_idx = min(
        len(list(st.session_state.suggested_indices)), start_idx + items_per_page
    )
    for idx in range(start_idx, end_idx):
        i = st.session_state.suggested_indices[idx][0]
        svm_score = st.session_state.suggested_indices[idx][1]
        if (
            st.session_state.start_year
            <= st.session_state.year_list[i]
            <= st.session_state.end_year
        ):
            if i in st.session_state.favorites:
                heart_color = "red"
            else:
                heart_color = "gray"

            heart_button = st.button(
                f"❤️" if heart_color == "red" else "🤍",
                key=f"heart_{i}_liked",
                on_click=toggle_favorite,
                args=(i,),
            )

            st.markdown(
                f"""
                <div style='border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <h2 style='text-align: center; font-weight: bold;'>{st.session_state.title_list[i]}</h2>
                    <p style='text-align: center; font-size: 1.2em;'>{int(st.session_state.year_list[i])}</p>
                    <p style='text-align: center; font-size: 1.2em;'>SVM_similarity: {svm_score}</p>
                    <p style='text-align: center; font-size: 1.1em;'>{st.session_state.description_list[i]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display bar chart for subject
            st.subheader("Subject Bar Chart")
            counts = getSubjectCounts(st.session_state.subject_list[i])
            plot_subject(counts)
            st.write("---")


def trainSVM(liked, not_liked):
    url = "http://0.0.0.0:8000/train"
    response = requests.post(
        url,
        json={
            "liked": liked,
            "not_liked": not_liked,
        },
    )
    st.info("Trained the SVM")
    if response.status_code == 200:
        st.session_state.train_svm = 0
    else:
        st.error("Failed to fetch data")
        st.stop()  # Stop execution if data fetching fails


def getLogit(idx):
    url = f"http://0.0.0.0:8000/get_logit?idx={idx}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["logit"]
    else:
        st.error("Failed to fetch data")
        st.stop()  # Stop execution if data fetching fails


def search():
    url = "http://0.0.0.0:8000/query"
    response = requests.post(
        url,
        json={
            "query_txt": st.session_state.search_term,
            "k": len(st.session_state.title_list),
        },
    )
    if response.status_code == 200:
        st.session_state.queried_data = response.json()
        st.session_state.redesign_for_query = 1


# Navigation
pages = {
    "Top Papers": top_papers,
    "Suggested Papers": suggested_papers,
    "Liked Papers": liked_papers,
}
st.session_state.search_term = st.text_input(
    "Search Liked Papers", value="", key="search_liked_papers", on_change=search
)
if st.button("Reset"):
    st.session_state.redesign_for_query = 0

selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page()

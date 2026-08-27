"""Streamlit dashboard for the ALS recommender.

Reads the artifacts written by `main.py`; it never retrains or re-reads the raw
archives.
"""
import streamlit as st

from src.data.preprocess import create_interaction_matrix
from src.models.als import ALSRecommender
from src.models.content import ContentRecommender, cold_start_recommend
from src.models.popularity import PopularityRecommender
from src.pipeline import build_interactions, build_metadata, title_lookup
from src.utils.helpers import load_config, load_pickle, resolve_path

st.set_page_config(page_title="E-commerce Recommender", layout="wide")
st.title("E-commerce Product Recommender")


@st.cache_resource
def get_config():
    return load_config()


cfg = get_config()


@st.cache_resource
def load_artifacts():
    """Loads the trained model and the mappings it was trained against."""
    out = cfg["output"]
    paths = [out["model_path"], out["user_map_path"], out["item_map_path"]]
    if not all(resolve_path(p).exists() for p in paths):
        return None

    return {
        "model": ALSRecommender.load(out["model_path"]),
        "user_map": load_pickle(out["user_map_path"]),
        "item_map": load_pickle(out["item_map_path"]),
    }


@st.cache_data
def load_data(_user_map, _item_map):
    """Rebuilds the interaction matrix inside the trained index space.

    Passing the persisted maps is essential: the model's factors are only
    meaningful against the mapping it was trained with. Rebuilding with fresh
    maps would silently recommend the wrong products.
    """
    reviews = build_interactions(cfg)
    matrix, _, _ = create_interaction_matrix(reviews, user_map=_user_map, item_map=_item_map)
    metadata = build_metadata(cfg, _item_map.keys())
    return matrix, metadata


artifacts = load_artifacts()
if artifacts is None:
    st.error(
        "No trained model found. Run `python main.py` first to train the model "
        "and write the artifacts into `data/`."
    )
    st.stop()

model = artifacts["model"]
user_map, item_map = artifacts["user_map"], artifacts["item_map"]
inv_item_map = {v: k for k, v in item_map.items()}

with st.spinner("Loading interactions and product metadata..."):
    interaction_matrix, metadata = load_data(user_map, item_map)
    titles = title_lookup(metadata)


@st.cache_resource
def build_side_models(_matrix, _metadata, _item_map):
    popularity = PopularityRecommender().fit(_matrix)
    content = ContentRecommender().fit(_metadata, _item_map)
    return popularity, content


popularity, content = build_side_models(interaction_matrix, metadata, item_map)


def product_name(item_index):
    asin = inv_item_map.get(int(item_index), f"index_{item_index}")
    return titles.get(asin, "Unknown Product"), asin


def render_products(indices, empty_message="Nothing to show."):
    if len(indices) == 0:
        st.info(empty_message)
        return
    for rank, item_index in enumerate(indices, start=1):
        title, asin = product_name(item_index)
        with st.container(border=True):
            st.markdown(f"**{rank}. {title}**")
            st.caption(f"ASIN: {asin}")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    st.metric("Users", f"{len(user_map):,}")
    st.metric("Items", f"{len(item_map):,}")
    st.metric("Interactions", f"{interaction_matrix.nnz:,}")
    st.caption(
        f"Metadata titles for {len(titles):,} of {len(item_map):,} items "
        "(the raw metadata archive is truncated)."
    )

# ---------------------------------------------------------------- controls
left, right = st.columns([1, 2])

with left:
    st.subheader("Find recommendations")

    user_ids = list(user_map)
    known_user = st.selectbox(
        "Known user", options=user_ids, index=0,
        help="Users the model was trained on.",
    )
    custom_user = st.text_input(
        "...or type any user ID",
        placeholder="Unknown IDs fall back to cold start",
    )
    selected_user = custom_user.strip() or known_user
    num_recs = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Generate", type="primary"):
        if selected_user in user_map:
            user_index = user_map[selected_user]
            user_items = interaction_matrix[user_index]
            recs = model.recommend(user_index, user_items, n=num_recs)
            st.session_state.update(
                recommendations=list(recs), cold_start=False,
                history=list(user_items.indices), user=selected_user,
            )
        else:
            recs = cold_start_recommend(content, popularity, [], n=num_recs)
            st.session_state.update(
                recommendations=list(recs), cold_start=True,
                history=[], user=selected_user,
            )

    if st.session_state.get("history"):
        st.markdown("#### Purchase history")
        render_products(st.session_state["history"])

# ---------------------------------------------------------------- results
with right:
    if "recommendations" not in st.session_state:
        st.info("Pick a user on the left and press **Generate**.")
    else:
        user = st.session_state["user"]
        if st.session_state["cold_start"]:
            st.subheader("Recommendations (cold start)")
            st.warning(
                f"`{user}` is not in the training data, so there are no learned "
                "preferences to personalise from. Falling back to globally "
                "popular products."
            )
        else:
            st.subheader(f"Recommendations for `{user}`")
        render_products(st.session_state["recommendations"])

st.divider()

# ---------------------------------------------------------------- similar items
st.subheader("Find similar products")
st.caption("Content-based lookalikes from product titles and categories.")

if not content.available:
    st.info("No product metadata available, so content similarity is disabled.")
else:
    # Rank by popularity so the picker opens on products people actually bought.
    known_items = [i for i in popularity.ranking if titles.get(inv_item_map.get(int(i)))]
    options = [int(i) for i in known_items[:2000]]

    choice = st.selectbox(
        "Product",
        options=options,
        format_func=lambda i: titles.get(inv_item_map[i], inv_item_map[i])[:110],
    )
    if choice is not None:
        similar = content.similar_items(int(choice), n=6)
        render_products(similar, "No similar products found.")

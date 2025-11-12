import json
import re
import numpy as np
import pandas as pd


def load_json_to_dataframe(file_path: str) -> pd.DataFrame:
    """Load a JSON file structured like the project's data files and return a DataFrame of the 'data' key.

    Keeps behaviour compatible with the notebook helper it replaces.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # some files might be plain list of dicts
    if isinstance(data, dict) and "data" in data:
        return pd.DataFrame(data["data"])
    return pd.DataFrame(data)


def _normalize_polish_characters(text: str) -> str:
    mapping = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def clean_text(t: str) -> str:
    """Normalize and clean text to a simple token string.

    This mirrors the behaviour used in the notebook: lowercasing, removing
    polish diacritics, stripping urls and non-alphanumeric characters.
    """
    if t is None:
        return ""
    t = str(t).lower()
    t = _normalize_polish_characters(t)
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t.strip()


def prepare_data(
    data_dir: str = "../data",
    id_file: str = "../data/final_combined_results_2937495-7229337.json",
    homepage_file: str = "../data/homepage_posts_data.json",
    misinfo_file: str = "../data/mssinfo_wykop_posts.json",
):
    """Run the notebook's data preparation pipeline and return (df, df_homepage, df_misinfo).

    The function attempts to replicate the transformations present in
    `notebooks/eda.ipynb` up to the point right before the initial model.
    It is defensive: missing optional columns will be handled gracefully.
    """
    # Load files
    try:
        df_homepage = load_json_to_dataframe(homepage_file)
    except Exception:
        df_homepage = pd.DataFrame()

    try:
        df_id = load_json_to_dataframe(id_file)
    except Exception:
        df_id = pd.DataFrame()

    try:
        df_misinfo = load_json_to_dataframe(misinfo_file)
    except Exception:
        df_misinfo = pd.DataFrame()

    # Process homepage like the notebook
    if not df_homepage.empty:
        drop_cols = [
            "created_at",
            "source",
            "slug",
            "published_at",
            "hot",
            "adult",
            "media",
            "observed_discussion",
            "parent",
            "tag_pinned_content",
            "pinnable",
            "editable",
            "deletable",
            "resource",
            "actions",
            "archive",
            "recommended",
            "family_friendly",
            "favourite",
            "ama",
            "voted",
        ]
        for c in drop_cols:
            if c in df_homepage.columns:
                df_homepage = df_homepage.drop(columns=[c])
        if "author" in df_homepage.columns:
            df_homepage["author"] = df_homepage["author"].apply(
                lambda x: x.get("username") if isinstance(x, dict) else x
            )
        if "votes" in df_homepage.columns:
            df_homepage["downvotes"] = df_homepage["votes"].apply(
                lambda x: x.get("down") if isinstance(x, dict) else None
            )
            df_homepage["votes"] = df_homepage["votes"].apply(
                lambda x: x.get("up") if isinstance(x, dict) else x
            )

    # Process id results (df_id) following the notebook logic
    if not df_id.empty:
        if "error" in df_id.columns:
            df_id = df_id[df_id["error"].isna()]
        for c in ["status", "proxy", "error"]:
            if c in df_id.columns:
                df_id = df_id.drop(columns=[c])
        df_id = df_id.reset_index(drop=True)

        # comments_content_points
        if "comments" in df_id.columns:
            df_id["comments_content_points"] = df_id["comments"].apply(
                lambda x: (
                    [
                        (c.get("content"), c.get("points"))
                        for c in x
                        if c.get("content") is not None
                    ]
                    if isinstance(x, list)
                    else []
                )
            )
        else:
            df_id["comments_content_points"] = [[] for _ in range(len(df_id))]

        pattern_manip = re.compile(r"\bmanipula\w*\b", flags=re.IGNORECASE)
        df_id["mentions_manipulation"] = df_id["comments_content_points"].apply(
            lambda list_c: any(
                pattern_manip.search(c[0]) for c in list_c if isinstance(c[0], str)
            )
        )

        def top_comment_by_pattern(comments, regex):
            if not comments:
                return None
            matches = (
                c for c in comments if isinstance(c[0], str) and regex.search(c[0])
            )
            try:
                return max(matches, key=lambda c: c[1])
            except Exception:
                return None

        df_id["top_manipulation_comment"] = df_id["comments_content_points"].apply(
            lambda comments: top_comment_by_pattern(comments, pattern_manip)
        )

        regex_info = re.compile(
            r"informacja nieprawdziwa|nieprawdziwa informacja", re.IGNORECASE
        )
        df_id["comment_informacja_nieprawdziwa"] = df_id[
            "comments_content_points"
        ].apply(lambda comments: top_comment_by_pattern(comments, regex_info))

        # Compute points ratio for 'informacja nieprawdziwa'
        if "points" in df_id.columns:
            mask = df_id["comment_informacja_nieprawdziwa"].notna()
            points_vals = df_id.loc[mask, "comment_informacja_nieprawdziwa"].apply(
                lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else None
            )
            base_points = df_id.loc[mask, "points"].abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = points_vals / base_points
            df_id.loc[mask, "score_informacja_nieprawdziwa"] = ratios

        # score_zakopane_fuzzy and score_zakopane
        alerts_df_id = (
            df_id[df_id.get("alerts").notna()]
            if "alerts" in df_id.columns
            else pd.DataFrame()
        )
        if not alerts_df_id.empty:
            # mark zakopane/duplikat similar to notebook
            alerts_df_id = alerts_df_id.copy()
            alerts_df_id["zakopane"] = alerts_df_id["alerts"].apply(
                lambda x: x.get("title") == "To Znalezisko zostało zakopane"
            )
            alerts_df_id["duplikat"] = alerts_df_id["alerts"].apply(
                lambda x: (x.get("title") == "To Znalezisko zostało wyrzucone")
                and ("Duplikat" in x.get("body", ""))
            )
            df_id.loc[alerts_df_id.index, "score_zakopane_fuzzy"] = np.clip(
                np.random.normal(loc=0.7, scale=0.05, size=len(alerts_df_id.index)),
                0,
                1,
            )
            df_id.loc[alerts_df_id.index, "score_zakopane"] = 0.7

        # Compose scores
        df_id["scores_fuzzy"] = (
            df_id[
                [
                    c
                    for c in ["score_informacja_nieprawdziwa", "score_zakopane_fuzzy"]
                    if c in df_id.columns
                ]
            ]
            .sum(axis=1, skipna=True)
            .fillna(0)
        )
        df_id["scores"] = (
            df_id[
                [
                    c
                    for c in ["score_informacja_nieprawdziwa", "score_zakopane"]
                    if c in df_id.columns
                ]
            ]
            .sum(axis=1, skipna=True)
            .fillna(0)
        )

        # Set small random noise for zeros in scores_fuzzy like notebook
        zero_idx = df_id[df_id["scores_fuzzy"] == 0].index
        if len(zero_idx) > 0:
            df_id.loc[zero_idx, "scores_fuzzy"] = np.clip(
                np.random.normal(loc=0, scale=0.05, size=len(zero_idx)), 0, 1
            )

        # rename points -> votes to match later code
        if "points" in df_id.columns:
            df_id = df_id.rename(columns={"points": "votes"})

    # Process misinfo dataset like notebook
    if not df_misinfo.empty:
        df_misinfo["scores"] = 1
        df_misinfo["scores_fuzzy"] = np.clip(
            np.random.normal(loc=1, scale=0.05, size=len(df_misinfo.index)), 0, 1
        )

    # Prepare ready variants and concat
    df_id_ready = pd.DataFrame()
    if not df_id.empty:
        drop_cols = [
            "id",
            "url",
            "added_date",
            "alerts",
            "comments",
            "comments_content_points",
            "mentions_manipulation",
            "top_manipulation_comment",
            "comment_informacja_nieprawdziwa",
            "score_informacja_nieprawdziwa",
            "score_zakopane",
        ]
        existing = [c for c in drop_cols if c in df_id.columns]
        df_id_ready = df_id.drop(columns=existing)

    df_misinfo_ready = pd.DataFrame()
    if not df_misinfo.empty:
        drop_cols2 = ["id", "url", "added_date", "author", "alerts", "downvotes"]
        existing2 = [c for c in drop_cols2 if c in df_misinfo.columns]
        df_misinfo_ready = df_misinfo.drop(columns=existing2)

    if not df_misinfo_ready.empty or not df_id_ready.empty:
        df = pd.concat([df_misinfo_ready, df_id_ready], ignore_index=True)
    else:
        df = pd.DataFrame()

    # Clip scores and create combined text
    if not df.empty:
        if "scores" in df.columns:
            df["scores"] = df["scores"].apply(
                lambda x: min(1, max(0, x)) if pd.notna(x) else x
            )
        if "scores_fuzzy" in df.columns:
            df["scores_fuzzy"] = df["scores_fuzzy"].apply(
                lambda x: min(1, max(0, x)) if pd.notna(x) else x
            )

        df["text"] = (
            df.get("title", "").fillna("")
            + " "
            + df.get("description", "").fillna("")
            + " "
            + df.get("tags", "").apply(
                lambda x: (
                    " ".join(x) if isinstance(x, list) else (x if pd.notna(x) else "")
                )
            )
        )

        df["clean_text_no_tags"] = (
            df.get("title", "").fillna("") + " " + df.get("description", "").fillna("")
        )
        df["clean_text"] = df["text"].apply(clean_text)

    return df, df_homepage, df_misinfo

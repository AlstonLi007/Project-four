from __future__ import annotations

"""Small Streamlit dashboard for WeChat LLM analytics.

The app expects the analytics views created by
``analytics_memori_wechat.ensure_llm_dashboard_views`` and shows a few
exploration panels:

* Chosen / edited ratios over time
* Style-profile before/after comparisons (computed on the fly)
* LLM failover counts (when heuristics were used)
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from analytics_memori_wechat import (
    ensure_llm_dashboard_views,
    get_engine,
    style_profile_before_after,
)


DEFAULT_DB_PATH = "wechat_assistant.db"


@st.cache_resource
def get_engine_cached(db_path: str):
    engine = get_engine(f"sqlite:///{db_path}")
    ensure_llm_dashboard_views(engine)
    return engine


@st.cache_data(ttl=60)
def load_view(db_path: str, view_name: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    engine = get_engine_cached(db_path)
    with engine.connect() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {view_name}", conn)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


@dataclass
class ContactSelection:
    value: str
    label: str


def _normalize_contacts(df: pd.DataFrame) -> list[ContactSelection]:
    contacts: list[ContactSelection] = []
    if df.empty:
        return contacts
    for _, row in df[["wechat_id", "contact_display_name"]].drop_duplicates().iterrows():
        contacts.append(ContactSelection(value=row["wechat_id"], label=row.get("contact_display_name") or row["wechat_id"]))
    return sorted(contacts, key=lambda c: c.label)


def show_chosen_vs_edited(db_path: str) -> None:
    st.subheader("Chosen / edited ratios over time")
    df_daily = load_view(db_path, "v_wechat_llm_metrics_daily", parse_dates=["llm_date"])
    if df_daily.empty:
        st.info("No daily metrics yet.")
        return

    contacts = _normalize_contacts(df_daily)
    default_values = [c.value for c in contacts[:5]]
    selected_values = st.multiselect(
        "Contacts", [c.value for c in contacts], format_func=lambda v: next((c.label for c in contacts if c.value == v), v), default=default_values,
    )

    if not selected_values:
        st.info("Select at least one contact.")
        return

    min_day, max_day = df_daily["llm_date"].min(), df_daily["llm_date"].max()
    date_range = st.slider(
        "Date range",
        min_value=min_day.to_pydatetime(),
        max_value=max_day.to_pydatetime(),
        value=(min_day.to_pydatetime(), max_day.to_pydatetime()),
    )

    mask = (
        df_daily["wechat_id"].isin(selected_values)
        & (df_daily["llm_date"] >= pd.to_datetime(date_range[0]))
        & (df_daily["llm_date"] <= pd.to_datetime(date_range[1]))
    )
    df_f = df_daily[mask].copy()
    if df_f.empty:
        st.info("No data for this selection yet.")
        return

    chosen_pivot = (
        df_f[["llm_date", "wechat_id", "chosen_ratio"]]
        .pivot(index="llm_date", columns="wechat_id", values="chosen_ratio")
        .sort_index()
    )
    st.markdown("**Chosen ratio over time**")
    st.line_chart(chosen_pivot)

    edited_pivot = (
        df_f[["llm_date", "wechat_id", "edited_ratio_overall"]]
        .pivot(index="llm_date", columns="wechat_id", values="edited_ratio_overall")
        .sort_index()
    )
    st.markdown("**Edited ratio over time**")
    st.line_chart(edited_pivot)

    with st.expander("Raw daily metrics"):
        st.dataframe(df_f.sort_values(["llm_date", "wechat_id"]))


def show_style_profile_before_after(db_path: str) -> None:
    st.subheader("Before / after style profile updates")
    engine = get_engine_cached(db_path)
    df_style = style_profile_before_after(engine)
    if df_style.empty:
        st.info("No style profile updates captured yet.")
        return

    profiles = sorted(df_style["style_profile_id"].dropna().unique())
    selected = st.multiselect("Style profiles", profiles, default=profiles[:5])
    if not selected:
        st.info("Select at least one style profile.")
        return

    df_f = df_style[df_style["style_profile_id"].isin(selected)].copy()
    agg = (
        df_f.groupby(["style_profile_id", "period"])
        .agg(
            chosen_ratio=("chosen_ratio", "mean"),
            edited_ratio=("edited_ratio", "mean"),
        )
        .reset_index()
    )

    st.markdown("**Chosen ratio: before vs after**")
    st.bar_chart(agg.pivot(index="style_profile_id", columns="period", values="chosen_ratio"))

    st.markdown("**Edited ratio: before vs after**")
    st.bar_chart(agg.pivot(index="style_profile_id", columns="period", values="edited_ratio"))

    with st.expander("Raw style stats"):
        st.dataframe(agg)


def show_failover_stats(db_path: str) -> None:
    st.subheader("LLM failover / heuristic usage")
    df_fail = load_view(db_path, "v_wechat_llm_failover_daily", parse_dates=["llm_date"])
    if df_fail.empty:
        st.info("No failover events logged yet.")
        return

    contacts = _normalize_contacts(df_fail)
    default_values = [c.value for c in contacts[:5]]
    selected_values = st.multiselect(
        "Contacts", [c.value for c in contacts], format_func=lambda v: next((c.label for c in contacts if c.value == v), v), default=default_values,
    )
    if not selected_values:
        st.info("Select at least one contact.")
        return

    df_f = df_fail[df_fail["wechat_id"].isin(selected_values)].copy()
    if df_f.empty:
        st.info("No failover data for this selection.")
        return

    pivot = (
        df_f[["llm_date", "wechat_id", "fail_count"]]
        .pivot(index="llm_date", columns="wechat_id", values="fail_count")
        .sort_index()
    )
    st.line_chart(pivot)

    with st.expander("Raw failover counts"):
        st.dataframe(df_f.sort_values(["llm_date", "wechat_id"]))


def main() -> None:
    st.title("WeChat LLM Dashboard")
    db_path = st.text_input("SQLite DB path", value=DEFAULT_DB_PATH)
    if not db_path:
        st.stop()

    st.sidebar.header("Panels")
    view = st.sidebar.radio(
        "Select panel",
        ["Chosen / edited ratios", "Style profile before/after", "Failover stats"],
        index=0,
    )

    if view == "Chosen / edited ratios":
        show_chosen_vs_edited(db_path)
    elif view == "Style profile before/after":
        show_style_profile_before_after(db_path)
    else:
        show_failover_stats(db_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Streamlit dashboard for WeChat LLM analytics.

Expected views (created by analytics_memori_wechat.ensure_llm_dashboard_views):

- v_wechat_llm_candidate_metrics
- v_wechat_llm_metrics_daily
- v_wechat_llm_metrics_style_overall
- v_wechat_llm_metrics_contact_overall
- v_wechat_llm_failover_daily
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


# -------- engine & loader --------


@st.cache_resource
def get_engine_cached(db_path: str):
    engine = get_engine(f"sqlite:///{db_path}")
    ensure_llm_dashboard_views(engine)
    return engine


@st.cache_data(ttl=60)
def load_view(
    db_path: str,
    view_name: str,
    *,
    parse_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
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
    small = df[["wechat_id", "contact_display_name"]].drop_duplicates()
    for _, row in small.iterrows():
        value = row["wechat_id"]
        label = row.get("contact_display_name") or value
        contacts.append(ContactSelection(value=value, label=label))
    return sorted(contacts, key=lambda c: c.label)


def _label_for(value: str, contacts: list[ContactSelection]) -> str:
    for c in contacts:
        if c.value == value:
            return c.label
    return value


# -------- panels --------


def show_chosen_vs_edited(db_path: str) -> None:
    st.subheader("📈 Chosen / Edited ratios over time")

    df_daily = load_view(
        db_path,
        "v_wechat_llm_metrics_daily",
        parse_dates=["llm_date"],
    )
    if df_daily.empty:
        st.info("No daily metrics yet – send some messages and run the dashboard setup first.")
        return

    contacts = _normalize_contacts(df_daily)
    if not contacts:
        st.info("No contacts found in metrics view.")
        return

    default_values = [c.value for c in contacts[:5]]
    selected_values = st.multiselect(
        "Contacts",
        options=[c.value for c in contacts],
        default=default_values,
        format_func=lambda v: _label_for(v, contacts),
    )
    if not selected_values:
        st.info("Select at least one contact.")
        return

    min_day = df_daily["llm_date"].min()
    max_day = df_daily["llm_date"].max()
    start, end = st.slider(
        "Date range",
        min_value=min_day.to_pydatetime(),
        max_value=max_day.to_pydatetime(),
        value=(min_day.to_pydatetime(), max_day.to_pydatetime()),
    )

    mask = (
        df_daily["wechat_id"].isin(selected_values)
        & (df_daily["llm_date"] >= pd.to_datetime(start))
        & (df_daily["llm_date"] <= pd.to_datetime(end))
    )
    sub = df_daily.loc[mask].copy()
    if sub.empty:
        st.info("No data in the selected window.")
        return

    # 聚合到 date+contact 级别
    group_cols = ["llm_date", "wechat_id", "contact_display_name"]
    agg = (
        sub.groupby(group_cols)
        .agg(
            chosen_ratio=("chosen_ratio", "mean"),
            edited_ratio_overall=("edited_ratio_overall", "mean"),
            edited_ratio_given_chosen=("edited_ratio_given_chosen", "mean"),
            total_candidates=("total_candidates", "sum"),
        )
        .reset_index()
    )

    st.write("Daily metrics (per contact):")
    st.dataframe(agg.sort_values(["llm_date", "wechat_id"]))

    # 画图
    if not agg.empty:
        pivot_chosen = agg.pivot_table(
            index="llm_date",
            columns="contact_display_name",
            values="chosen_ratio",
        )
        st.line_chart(pivot_chosen, height=260, use_container_width=True)

        pivot_edited = agg.pivot_table(
            index="llm_date",
            columns="contact_display_name",
            values="edited_ratio_given_chosen",
        )
        st.line_chart(pivot_edited, height=260, use_container_width=True)


def show_style_overall(db_path: str) -> None:
    st.subheader("🎭 Style profile metrics (overall)")

    df_style = load_view(db_path, "v_wechat_llm_metrics_style_overall")
    if df_style.empty:
        st.info("No style metrics yet.")
        return

    st.write("Aggregated by style_profile + prompt_version:")
    st.dataframe(
        df_style.sort_values(["style_label", "prompt_version"]),
        use_container_width=True,
        hide_index=True,
    )

    # 简单条形图：按 avg_len / chosen_ratio 看
    if "style_label" in df_style.columns:
        st.markdown("**Average candidate length by style**")
        st.bar_chart(
            df_style.set_index("style_label")["avg_len"],
            use_container_width=True,
            height=260,
        )

        st.markdown("**Chosen ratio by style**")
        st.bar_chart(
            df_style.set_index("style_label")["chosen_ratio"],
            use_container_width=True,
            height=260,
        )


def show_style_before_after(db_path: str) -> None:
    st.subheader("⏱ Style profile before / after update")

    engine = get_engine_cached(db_path)
    df = style_profile_before_after(engine)
    if df.empty:
        st.info("No before/after data – you may not have updated style profiles yet.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    pivot = df.pivot_table(
        index="label",
        columns="period",
        values=["chosen_ratio", "avg_len", "edited_ratio"],
    )
    st.write("Pivot by label / period:")
    st.dataframe(pivot, use_container_width=True)


def show_failover(db_path: str) -> None:
    st.subheader("⚠️ Failover events (LLM errors → heuristic)")

    df_fail = load_view(
        db_path,
        "v_wechat_llm_failover_daily",
        parse_dates=["llm_date"],
    )
    if df_fail.empty:
        st.info("No failover events recorded – good sign.")
        return

    st.dataframe(df_fail.sort_values(["llm_date", "wechat_id"]), use_container_width=True)

    pivot = df_fail.pivot_table(
        index="llm_date",
        columns="backend",
        values="fail_count",
        aggfunc="sum",
    )
    st.line_chart(pivot, use_container_width=True, height=260)


def show_contact_prompt_metrics(db_path: str) -> None:
    st.subheader("👤 Contact x prompt_version metrics")

    df = load_view(db_path, "v_wechat_llm_metrics_contact_overall")
    if df.empty:
        st.info("No contact/prompt metrics yet.")
        return

    st.dataframe(df.sort_values(["wechat_id", "prompt_version"]), use_container_width=True)

    contacts = sorted(df["contact_display_name"].dropna().unique())
    if not contacts:
        return

    selected = st.selectbox("Contact", contacts)
    sub = df[df["contact_display_name"] == selected]

    st.bar_chart(
        sub.set_index("prompt_version")["chosen_ratio"],
        use_container_width=True,
        height=260,
    )


# -------- main app --------


def main() -> None:
    st.set_page_config(
        page_title="WeChat LLM Dashboard",
        layout="wide",
    )

    st.title("WeChat LLM Dashboard")

    db_path = st.sidebar.text_input(
        "SQLite DB path",
        value=DEFAULT_DB_PATH,
        help="Path to the shared SQLite DB (wechat_assistant.db).",
    )
    if not db_path:
        st.stop()

    # initialize views/engine (cached)
    _ = get_engine_cached(db_path)

    tab_overview, tab_style, tab_contact, tab_fail = st.tabs(
        ["Overview", "Style profiles", "Contacts & prompts", "Failover"]
    )

    with tab_overview:
        show_chosen_vs_edited(db_path)

    with tab_style:
        show_style_overall(db_path)
        st.markdown("---")
        show_style_before_after(db_path)

    with tab_contact:
        show_contact_prompt_metrics(db_path)

    with tab_fail:
        show_failover(db_path)


if __name__ == "__main__":
    main()

"""Quick-start notebook-style helper for exploring LLM metrics.

Run this as a script or copy the cells into a notebook. It assumes your
SQLite file already contains the dashboard views created by
``analytics_memori_wechat.ensure_llm_dashboard_views``.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from analytics_memori_wechat import ensure_llm_dashboard_views, get_engine


DB_PATH = "wechat_assistant.db"


def main() -> None:
    engine = get_engine(f"sqlite:///{DB_PATH}")
    ensure_llm_dashboard_views(engine)

    daily = pd.read_sql("SELECT * FROM v_wechat_llm_metrics_daily", engine, parse_dates=["llm_date"])
    if daily.empty:
        print("No daily metrics yet; send a few messages first.")
        return

    contact_id = daily["wechat_id"].iloc[0]
    contact_df = daily[daily["wechat_id"] == contact_id].sort_values("llm_date")

    plt.plot(contact_df["llm_date"], contact_df["chosen_ratio"], marker="o")
    plt.title(f"Chosen ratio over time for {contact_id}")
    plt.xlabel("Day")
    plt.ylabel("Chosen ratio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

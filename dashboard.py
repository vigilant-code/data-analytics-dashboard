from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Google Play Store dashboard: six robust, interactive visualizations.
# Place this file next to googleplaystore.csv and (optionally) the reviews CSV.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
APP_CSV_CANDIDATES = [BASE_DIR / "googleplaystore.csv", BASE_DIR / "google-play-store-apps" / "googleplaystore.csv"]
REVIEWS_CSV_CANDIDATES = [BASE_DIR / "googleplaystore_user_reviews.csv", BASE_DIR / "google-play-store-apps" / "googleplaystore_user_reviews.csv"]
IST = ZoneInfo("Asia/Kolkata")

TASKS = {
    "Task 1 — Size vs rating": (17, 19),
    "Task 2 — Category sunburst": (18, 20),
    "Task 3 — Monthly installs heatmap": (18, 21),
    "Task 4 — Install trends": (16, 18),
    "Task 5 — Category performance heatmap": (15, 17),
    "Task 6 — Free vs paid radar": (13, 14),
}

st.set_page_config(page_title="Google Play Store Analytics", page_icon="📱", layout="wide")
st.title("📱 Google Play Store Data Analysis Dashboard")
st.caption("Six interactive views of the Google Play Store dataset · Time zone: India Standard Time (IST)")


def parse_numeric(value):
    """Parse values like '1,000+', '$4.99', '19M', and 'Varies with device'."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "").replace("$", "").replace("+", "")
    if not text or text.lower() in {"nan", "none", "varies with device", "not available"}:
        return np.nan
    multiplier = 1.0
    if text[-1:] in ("M", "m"):
        multiplier, text = 1_000_000.0, text[:-1]
    elif text[-1:] in ("K", "k"):
        multiplier, text = 1_000.0, text[:-1]
    elif text.endswith("%"):
        text = text[:-1]
    try:
        return float(text) * multiplier
    except (ValueError, TypeError):
        return np.nan


def find_file(candidates):
    return next((candidate for candidate in candidates if candidate.exists()), None)


@st.cache_data(show_spinner="Loading and cleaning the CSV data…")
def load_data():
    apps_path = find_file(APP_CSV_CANDIDATES)
    if apps_path is None:
        searched = "\n".join(f"• {path}" for path in APP_CSV_CANDIDATES)
        raise FileNotFoundError("Could not find googleplaystore.csv. Put it beside this Python file. Checked:\n" + searched)
    apps = pd.read_csv(apps_path)
    # Normalize common column names while preserving the original data.
    apps.columns = [str(col).strip() for col in apps.columns]
    required = {"App", "Category", "Rating", "Reviews", "Size", "Installs", "Type", "Price", "Content Rating", "Last Updated", "Android Ver"}
    missing = sorted(required - set(apps.columns))
    if missing:
        raise ValueError("The apps CSV is missing these columns: " + ", ".join(missing))

    apps = apps.copy()
    apps["App"] = apps["App"].fillna("Unknown app").astype(str)
    apps["Category"] = apps["Category"].fillna("Unknown").astype(str)
    apps["Rating"] = pd.to_numeric(apps["Rating"], errors="coerce")
    apps["Reviews_num"] = pd.to_numeric(apps["Reviews"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    apps["Installs_num"] = apps["Installs"].map(parse_numeric)
    apps["Size_num"] = apps["Size"].map(parse_numeric)
    apps["Price_num"] = apps["Price"].map(parse_numeric).fillna(0)
    apps["Type"] = apps["Type"].fillna("Free").astype(str).str.title()
    apps.loc[~apps["Type"].isin(["Free", "Paid"]), "Type"] = "Free"
    apps["Last Updated"] = pd.to_datetime(apps["Last Updated"], errors="coerce")
    apps["Month"] = apps["Last Updated"].dt.to_period("M").dt.to_timestamp()
    apps["Android Ver_num"] = apps["Android Ver"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    apps["Android Ver_num"] = pd.to_numeric(apps["Android Ver_num"], errors="coerce")
    reviews_path = find_file(REVIEWS_CSV_CANDIDATES)
    reviews = pd.DataFrame()
    if reviews_path is not None:
        try:
            reviews = pd.read_csv(reviews_path)
            reviews.columns = [str(col).strip() for col in reviews.columns]
        except Exception as exc:
            st.warning(f"Reviews file could not be read; sentiment-based filters will be skipped. Details: {exc}")
    return apps, reviews, apps_path.name, (reviews_path.name if reviews_path else None)


def window_info(task_name):
    start, end = TASKS[task_name]
    now = datetime.now(IST)
    allowed = start <= now.hour < end
    def clock(hour):
        return datetime(2000, 1, 1, hour).strftime("%I %p").lstrip("0")
    return allowed, f"{clock(start)}–{clock(end)} IST", now.strftime("%I:%M %p IST").lstrip("0")


def safe_plot(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key)


def empty_message(data, task):
    if data.empty:
        st.warning(f"No rows match the strict filters for {task}. Showing a broader fallback view where possible, so the chart is not silently blank.")
        return True
    return False


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return float(values.mean()) if values.notna().any() else np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def add_subjectivity(apps, reviews):
    """Return apps with average subjectivity when a compatible reviews file exists."""
    if reviews.empty or "App" not in reviews.columns:
        return apps, False
    col = next((c for c in reviews.columns if c.strip().lower() == "subjectivity"), None)
    if col is None:
        return apps, False
    sentiment = reviews[["App", col]].copy()
    sentiment[col] = pd.to_numeric(sentiment[col], errors="coerce")
    sentiment = sentiment.groupby("App", as_index=False)[col].mean().rename(columns={col: "Subjectivity"})
    return apps.merge(sentiment, on="App", how="left"), True


def task1(apps, reviews):
    st.subheader("Task 1 — App size vs rating")
    data = apps.dropna(subset=["Rating", "Installs_num", "Reviews_num", "Size_num"]).copy()
    strict = data[(data.Rating > 3.5) & (data.Installs_num > 50_000) & (data.Reviews_num > 500) & data.Size_num.between(10, 100)]
    strict = strict[~strict.App.str.contains("s", case=False, regex=False)]
    strict, has_subjectivity = add_subjectivity(strict, reviews)
    if has_subjectivity:
        strict = strict[strict.Subjectivity > 0.5]
    else:
        st.info("The reviews CSV with a Subjectivity column was not found; sentiment filtering is skipped for this chart.")
    if strict.empty:
        st.info("The full set of Task 1 filters returned no rows. Using the core numeric fields so the visualization remains available.")
        strict = data[(data.Rating > 3.5) & data.Size_num.between(1, 150)]
    if strict.empty:
        strict = data
    if strict.empty:
        st.error("The apps dataset has no rows with usable rating, size, installs, and review values.")
        return
    # Density heatmap encodes average installs in each size/rating bin.
    # Marginal histograms are intentionally omitted here because some Plotly versions
    # raise a ValueError when density_heatmap combines marginals with a color scale.
    fig = px.density_heatmap(
        strict, x="Size_num", y="Rating", z="Installs_num", histfunc="avg",
        nbinsx=24, nbinsy=18,
        color_continuous_scale="Viridis", title="App size vs rating (colour = average installs)",
        labels={"Size_num": "App size (MB)", "Rating": "Rating", "Installs_num": "Average installs"},
        hover_data={"App": True, "Category": True, "Reviews_num": ":,.0f"},
    )
    # Overlay outliers in installs using category-specific IQR.
    bounds = strict.groupby("Category")["Installs_num"].agg(
        lambda s: s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25))
    )
    outliers = strict[strict.apply(lambda r: r.Installs_num > bounds.get(r.Category, np.inf), axis=1)].nlargest(40, "Installs_num")
    if not outliers.empty:
        fig.add_trace(go.Scatter(x=outliers.Size_num, y=outliers.Rating, mode="markers+text",
            text=outliers.App.str.slice(0, 18), textposition="top center", name="Install outliers",
            marker=dict(color="#ff4fa3", size=9, line=dict(color="white", width=1)),
            customdata=np.stack([outliers.Category, outliers.Installs_num], axis=-1),
            hovertemplate="%{text}<br>Category: %{customdata[0]}<br>Installs: %{customdata[1]:,.0f}<extra></extra>"))
    safe_plot(fig, "task1_chart")
    st.caption(f"Rows plotted: {len(strict):,}")


def task2(apps):
    st.subheader("Task 2 — Category sunburst")
    data = apps.dropna(subset=["Rating", "Installs_num", "Reviews_num", "Size_num"]).copy()
    data = data[(data.Rating >= 4.0) & (data.Installs_num > 10_000) & (data.Reviews_num > 1_000) & data.Size_num.between(15, 80)]
    data = data[~data.App.str.contains(r"\d", regex=True)]
    data = data[~data.Category.str.startswith(("A", "C", "G", "S"), na=False)]
    if data.empty:
        st.info("Strict filters produced no rows; relaxing the app-name and category exclusions for the overview.")
        data = apps.dropna(subset=["Rating", "Installs_num", "Reviews_num"]).copy()
        data = data[(data.Rating >= 4.0) & (data.Installs_num > 10_000) & (data.Reviews_num > 100)]
    if data.empty:
        data = apps.dropna(subset=["Rating", "Installs_num"]).copy()
    if data.empty:
        st.error("There is no usable rating/install data for the sunburst.")
        return
    top_categories = data.groupby("Category").Installs_num.sum().nlargest(5).index
    data = data[data.Category.isin(top_categories)].copy()
    data["Rating Band"] = pd.cut(data.Rating, bins=[0, 4.2, 4.5, 4.7, 5.01], labels=["4.0–4.2", "4.2–4.5", "4.5–4.7", "4.7–5.0"], include_lowest=True, right=True).astype(str)
    data["Rating Band"] = data["Rating Band"].replace("nan", "Unrated")
    data["App Type"] = data["Type"].where(data["Type"].isin(["Free", "Paid"]), "Unknown")
    grouped = data.groupby(["Category", "App Type", "Rating Band"], observed=False).agg(
        Installs=("Installs_num", "sum"), Reviews=("Reviews_num", "sum"), WeightedRating=("Rating", lambda s: weighted_mean(s, data.loc[s.index, "Reviews_num"]))
    ).reset_index()
    # Plotly sunburst requires a hierarchy; app-level nodes are omitted to keep the plot readable.
    fig = px.sunburst(grouped, path=["Category", "App Type", "Rating Band"], values="Installs", color="WeightedRating",
        color_continuous_scale="RdYlGn", range_color=(3.5, 5), title="Installs by category, app type and rating band",
        hover_data={"Reviews": ":,.0f", "WeightedRating": ":.2f", "Installs": ":,.0f"})
    fig.update_traces(textinfo="label+percent parent", insidetextorientation="radial", marker=dict(line=dict(color="white", width=1)))
    safe_plot(fig, "task2_chart")
    st.caption("The supplied apps dataset does not contain a country column, so the hierarchy begins with Category rather than Country.")


def task3(apps, reviews):
    st.subheader("Task 3 — Monthly installs heatmap")
    data = apps.dropna(subset=["Month", "Installs_num", "Rating", "Reviews_num", "Size_num"]).copy()
    strict = data[(data.Rating >= 4) & (data.Installs_num > 10_000) & (data.Reviews_num > 500) & data.Size_num.between(15, 80)]
    strict = strict[strict.Category.str.startswith(("E", "C", "B"), na=False)]
    strict = strict[~strict.App.str.match(r"^[XYZ]", case=False) & ~strict.App.str.contains("s", case=False, regex=False)]
    strict, has_subjectivity = add_subjectivity(strict, reviews)
    if has_subjectivity:
        strict = strict[strict.Subjectivity > .5]
    if strict.empty:
        st.info("Strict filters returned no rows; showing monthly installs for the categories beginning E, C or B.")
        strict = data[data.Category.str.startswith(("E", "C", "B"), na=False)]
    if strict.empty:
        strict = data
    if strict.empty:
        st.error("No dated install data is available for the monthly heatmap.")
        return
    monthly = strict.groupby(["Category", "Month"], as_index=False).agg(Installs=("Installs_num", "sum"), Reviews=("Reviews_num", "sum"))
    top_categories = monthly.groupby("Category").Installs.sum().nlargest(5).index.tolist()
    selected = st.multiselect("Categories to display", options=sorted(monthly.Category.unique()), default=top_categories, key="task3_categories")
    if selected:
        monthly = monthly[monthly.Category.isin(selected)]
    pivot = monthly.pivot_table(index="Category", columns="Month", values="Installs", aggfunc="sum", fill_value=0)
    if pivot.empty:
        st.warning("Choose at least one category to display.")
        return
    # Use a continuous calendar range so missing months are shown as zero.
    full_months = pd.date_range(pivot.columns.min(), pivot.columns.max(), freq="MS")
    pivot = pivot.reindex(columns=full_months, fill_value=0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlGnBu", labels=dict(x="Month", y="Category", color="Installs"),
                    title="Monthly installs by category", x=[d.strftime("%Y-%m") for d in pivot.columns])
    safe_plot(fig, "task3_chart")
    growth = pivot.pct_change(axis=1).replace([np.inf, -np.inf], np.nan) * 100
    rolling = pivot.T.rolling(3, min_periods=1).mean().T
    with st.expander("Monthly growth and simple 3-month moving-average forecast"):
        st.write("Month-over-month growth (%)")
        st.dataframe(growth.round(1), use_container_width=True)
        st.write("Three-month moving average (baseline forecast proxy)")
        st.dataframe(rolling.round(0), use_container_width=True)


def task4(apps):
    st.subheader("Task 4 — Install trends and anomalies")
    data = apps.dropna(subset=["Month", "Installs_num", "Rating", "Reviews_num", "Size_num"]).copy()
    strict = data[(data.Rating >= 4.2) & (data.Reviews_num > 1_000) & data.Size_num.between(20, 80) & (data.Installs_num >= 10_000)]
    strict = strict[strict.Category.str.startswith(("T", "P", "B"), na=False)]
    if strict.empty:
        st.info("Strict filters returned no rows; using all categories beginning T, P or B with dated install data.")
        strict = data[data.Category.str.startswith(("T", "P", "B"), na=False)]
    if strict.empty:
        strict = data
    if strict.empty:
        st.error("No dated install data is available for the streamgraph.")
        return
    monthly = strict.groupby(["Month", "Category"], as_index=False).Installs_num.sum().rename(columns={"Installs_num": "Installs"})
    cats = monthly.groupby("Category").Installs.sum().nlargest(8).index
    monthly = monthly[monthly.Category.isin(cats)]
    wide = monthly.pivot(index="Month", columns="Category", values="Installs").fillna(0).sort_index()
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    wide = wide.reindex(full_index, fill_value=0)
    wide.index.name = "Month"
    mode = st.radio("Display", ["Monthly installs", "Cumulative installs", "Growth %"], horizontal=True, key="task4_mode")
    if mode == "Cumulative installs":
        plot_data = wide.cumsum()
    elif mode == "Growth %":
        plot_data = wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    else:
        plot_data = wide
    fig = go.Figure()
    for category in plot_data.columns:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[category], mode="lines", stackgroup="one" if mode != "Growth %" else None, name=str(category), hovertemplate=f"{category}<br>%{{x|%Y-%m}}<br>%{{y:,.2f}}<extra></extra>"))
    fig.update_layout(title=f"{mode} by category", xaxis_title="Month", yaxis_title=mode, hovermode="x unified")
    safe_plot(fig, "task4_chart")
    growth = wide.pct_change().replace([np.inf, -np.inf], np.nan) * 100
    totals = wide.sum(axis=1)
    z = (totals - totals.rolling(6, min_periods=3).mean()) / totals.rolling(6, min_periods=3).std().replace(0, np.nan)
    anomalies = (growth.abs().max(axis=1) > 25) | (z > 2)
    if anomalies.any():
        st.write("Potential anomalous months (flag for review; not proof of an error):")
        st.dataframe(pd.DataFrame({"Month": wide.index[anomalies].strftime("%Y-%m"), "Total installs": totals[anomalies].values, "Max absolute category growth (%)": growth.abs().max(axis=1)[anomalies].round(1).values}), use_container_width=True)
    else:
        st.caption("No months crossed the configured anomaly thresholds in this filtered view.")


def task5(apps):
    st.subheader("Task 5 — Clustered category performance heatmap")
    data = apps.dropna(subset=["Rating", "Reviews_num", "Installs_num", "Size_num", "Last Updated"]).copy()
    strict = data[(data.Rating >= 4) & (data.Size_num > 10) & (data.Installs_num >= 10_000) & (data.Reviews_num > 1_000) & (data["Last Updated"].dt.month == 1)]
    strict = strict[~strict.App.str.contains(r"\d", regex=True)]
    if strict.empty:
        st.info("The January-only and name filters returned no rows; relaxing those two restrictions to keep the heatmap useful.")
        strict = data[(data.Rating >= 4) & (data.Size_num > 10) & (data.Installs_num >= 10_000) & (data.Reviews_num > 1_000)]
    if strict.empty:
        st.info("Further relaxing numeric thresholds because no rows match the core filters.")
        strict = data
    if strict.empty:
        st.error("The dataset has no usable rows for category metrics.")
        return
    strict["Engagement"] = strict["Reviews_num"] / strict["Installs_num"].replace(0, np.nan)
    strict["UpdateYear"] = strict["Last Updated"].dt.year
    metrics = strict.groupby("Category").apply(lambda g: pd.Series({
        "Weighted rating": weighted_mean(g.Rating, g.Reviews_num),
        "Total reviews": g.Reviews_num.sum(),
        "Total installs": g.Installs_num.sum(),
        "Average size (MB)": g.Size_num.mean(),
        "Engagement rate": g.Engagement.mean() * 100,
        "Update frequency": g["Last Updated"].dt.to_period("M").nunique(),
    }), include_groups=False).dropna(how="all")
    if metrics.empty:
        st.error("Could not calculate category metrics from the available rows.")
        return
    metrics = metrics.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Bound extreme skew for display only; raw table remains available.
    raw = metrics.copy()
    normalized = (metrics - metrics.mean()) / metrics.std(ddof=0).replace(0, 1)
    normalized = normalized.replace([np.inf, -np.inf], 0).fillna(0)
    # Positive composite ranking based on normalized metric average; ranking is project-defined.
    score = normalized.mean(axis=1)
    order = score.sort_values(ascending=False).index
    metrics = metrics.loc[order].head(10)
    normalized = normalized.loc[metrics.index]
    score = score.loc[metrics.index]
    view = st.radio("Heatmap values", ["Normalized (Z-score)", "Raw metrics"], horizontal=True, key="task5_view")
    shown = normalized if view == "Normalized (Z-score)" else metrics
    fig = px.imshow(shown, aspect="auto", color_continuous_scale="RdBu_r" if view.startswith("Normalized") else "Viridis",
                    labels=dict(x="Metric", y="Category", color="Value"), title=f"Top categories by composite score — {view}", text_auto=".2f")
    fig.update_layout(xaxis_side="bottom")
    safe_plot(fig, "task5_chart")
    st.write("Composite score ranking (average of normalized metrics; higher is better under this chosen definition).")
    st.dataframe(pd.DataFrame({"Category": score.index, "Composite score": score.values}).reset_index(drop=True), use_container_width=True)
    with st.expander("View raw metrics"):
        st.dataframe(raw.loc[score.index].round(3), use_container_width=True)


def task6(apps):
    st.subheader("Task 6 — Free vs paid apps radar")
    data = apps.dropna(subset=["Rating", "Reviews_num", "Installs_num", "Size_num"]).copy()
    strict = data[(data.Installs_num >= 10_000) & (data.Reviews_num > 100) & (data.Size_num > 15) & (data["Android Ver_num"] > 4.0) & (data["Content Rating"].astype(str) == "Everyone") & (data.App.str.len() > 5)]
    if strict.empty:
        st.info("Strict eligibility filters returned no rows; relaxing Android/content/name constraints for the comparison.")
        strict = data[(data.Installs_num >= 10_000) & (data.Reviews_num > 100) & (data.Size_num > 15)]
    if strict.empty:
        strict = data
    if strict.empty:
        st.error("No usable data is available for the free-vs-paid radar chart.")
        return
    categories = strict.groupby("Category").Installs_num.sum().nlargest(5).index.tolist()
    strict = strict[strict.Category.isin(categories)].copy()
    strict["Revenue estimate"] = np.where(strict.Type.eq("Paid"), strict.Installs_num * strict.Price_num, 0.0)
    strict["Engagement rate"] = strict.Reviews_num / strict.Installs_num.replace(0, np.nan)
    cat_options = ["Overall"] + categories
    selected_cat = st.selectbox("Compare", cat_options, key="task6_category")
    subset = strict if selected_cat == "Overall" else strict[strict.Category == selected_cat]
    rows = []
    for app_type in ["Free", "Paid"]:
        group = subset[subset.Type == app_type]
        if group.empty:
            continue
        rows.append({
            "Type": app_type,
            "Average installs": group.Installs_num.mean(),
            "Weighted rating": weighted_mean(group.Rating, group.Reviews_num),
            "Total reviews": group.Reviews_num.sum(),
            "Average size": group.Size_num.mean(),
            "Revenue estimate": group["Revenue estimate"].sum(),
            "Engagement rate": group["Engagement rate"].mean() * 100,
        })
    summary = pd.DataFrame(rows).set_index("Type") if rows else pd.DataFrame()
    metrics = ["Average installs", "Weighted rating", "Total reviews", "Average size", "Revenue estimate", "Engagement rate"]
    if summary.empty:
        st.warning("Neither Free nor Paid rows remain after filtering.")
        return
    # Percentile-style min-max normalization across the two displayed app types.
    values = summary.reindex(columns=metrics).astype(float)
    normalized = values.copy()
    for metric in metrics:
        lo, hi = values[metric].min(), values[metric].max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            normalized[metric] = 50.0 if not pd.isna(lo) else 0.0
        else:
            normalized[metric] = (values[metric] - lo) / (hi - lo) * 100
    fig = go.Figure()
    for app_type in normalized.index:
        radial = normalized.loc[app_type, metrics].tolist()
        fig.add_trace(go.Scatterpolar(r=radial + [radial[0]], theta=metrics + [metrics[0]], fill="toself", name=app_type))
    fig.update_layout(title=f"Free vs paid — {selected_cat} (normalized scores)", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
    safe_plot(fig, "task6_chart")
    st.caption("Revenue is an estimate (installs × listed price), not verified earnings. A value of 50 is used when a metric is identical for both types.")
    st.dataframe(summary.T.rename_axis("Metric"), use_container_width=True)


try:
    df_apps, df_reviews, apps_filename, reviews_filename = load_data()
except Exception as exc:
    st.error(f"Could not load the dashboard data: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Apps loaded", f"{len(df_apps):,}")
col2.metric("Categories", f"{df_apps['Category'].nunique():,}")
col3.metric("Reviews records", f"{len(df_reviews):,}" if not df_reviews.empty else "Optional file missing")
st.caption(f"Apps file: {apps_filename} · Reviews file: {reviews_filename or 'not found (optional)'}")

selected_task = st.selectbox("Choose an analysis", list(TASKS.keys()), key="selected_task")
allowed, time_window, current_time = window_info(selected_task)
st.caption(f"Current time: {current_time} · Original task window: {time_window}")
# Time gates are optional because otherwise most users see no chart for much of the day.
enforce_time = st.checkbox("Enforce original time windows (hide chart outside its assigned slot)", value=False, help="Turn this on if you need the notebook's strict time-slot behaviour. Leave it off to inspect all six charts at any time.")
if enforce_time and not allowed:
    st.warning(f"{selected_task} is scheduled for {time_window}. Turn off the checkbox above to view it now.")
else:
    if not allowed:
        st.info(f"Outside the original time slot ({time_window}); displaying the chart because time-window enforcement is off.")
    try:
        if selected_task.startswith("Task 1"):
            task1(df_apps, df_reviews)
        elif selected_task.startswith("Task 2"):
            task2(df_apps)
        elif selected_task.startswith("Task 3"):
            task3(df_apps, df_reviews)
        elif selected_task.startswith("Task 4"):
            task4(df_apps)
        elif selected_task.startswith("Task 5"):
            task5(df_apps)
        else:
            task6(df_apps)
    except Exception as exc:
        st.error(f"{selected_task} failed: {type(exc).__name__}: {exc}")
        st.info("Please copy the full traceback from the terminal if the problem persists.")

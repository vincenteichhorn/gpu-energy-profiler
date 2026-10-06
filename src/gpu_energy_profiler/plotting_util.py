"""Shared plotting helpers for profiler visualizations."""

import plotly.express as px


def sample_colors(scale: str, item_count: int) -> list[str]:
    """Return evenly spaced colors for a collection.

    Args:
        scale: Plotly color scale name.
        item_count: Number of colors required.

    Returns:
        At least two colors sampled from the requested scale.
    """
    count = max(item_count, 2)
    return [
        str(color)
        for color in px.colors.sample_colorscale(
            scale, [index / (count - 1) for index in range(count)]
        )
    ]

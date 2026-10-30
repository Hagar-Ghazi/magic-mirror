import streamlit as st

from core.helpers import COLOR_PRESETS, build_ranges


class Spell:
    """A spell turns one BGR frame into a result plus its intermediate steps.

    process() must return (result_bgr, [(step_title, image), ...]).
    `state` is a dict the spell owns; it survives between frames so spells
    can remember backgrounds, drawings or previous frames.
    """

    name = ""
    icon = ""
    tagline = ""
    explain = ""

    def sidebar(self, ui, mode):
        return {}

    def process(self, frame, params, state):
        raise NotImplementedError


def action_button(ui, label, key):
    """A button whose clicks are counted, so a background thread can notice new clicks."""
    count_key = f"_action_{key}"
    st.session_state.setdefault(count_key, 0)
    if ui.button(label, key=f"btn_{key}", width="stretch"):
        st.session_state[count_key] += 1
    return st.session_state[count_key]


def take_action(state, params, name):
    """True once for every new click of the named action button."""
    count = params.get("actions", {}).get(name, 0)
    seen_key = f"_seen_{name}"
    if seen_key not in state:
        state[seen_key] = count
        return False
    if count != state[seen_key]:
        state[seen_key] = count
        return True
    return False


def color_controls(ui, key, default="Red"):
    presets = list(COLOR_PRESETS) + ["Custom"]
    preset = ui.selectbox("Target color", presets, index=presets.index(default), key=f"{key}_preset")
    hue = (0, 10)
    if preset == "Custom":
        hue = ui.slider("Hue range", 0, 179, (0, 10), key=f"{key}_hue")
    s_min = ui.slider("Min saturation", 0, 255, 110, key=f"{key}_s")
    v_min = ui.slider("Min brightness (V)", 0, 255, 60, key=f"{key}_v")
    return build_ranges(preset, hue, s_min, v_min)

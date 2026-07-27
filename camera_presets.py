AUTO_CAMERA_PRESET = "Auto Director"

CAMERA_PRESET_SPECS = {
    AUTO_CAMERA_PRESET: {
        "vertical_fraction": 1.0,
        "margin": None,
        "roll_degrees": 0.0,
        "allowed_border_clipping": (),
    },
    "Classic Full Shot": {
        "vertical_fraction": 1.0,
        "margin": 0.08,
        "roll_degrees": 0.0,
        "allowed_border_clipping": (),
    },
    "American / Cowboy Shot": {
        "vertical_fraction": 0.72,
        "margin": 0.08,
        "roll_degrees": 0.0,
        "allowed_border_clipping": ("bottom", "left", "right"),
    },
    "Medium Shot": {
        "vertical_fraction": 0.58,
        "margin": 0.08,
        "roll_degrees": 0.0,
        "allowed_border_clipping": ("bottom", "left", "right"),
    },
    "Medium Close-Up": {
        "vertical_fraction": 0.44,
        "margin": 0.08,
        "roll_degrees": 0.0,
        "allowed_border_clipping": ("bottom", "left", "right"),
    },
    "Classic Close-Up": {
        "vertical_fraction": 0.32,
        "margin": 0.08,
        "roll_degrees": 0.0,
        "allowed_border_clipping": ("bottom", "left", "right"),
    },
    "Dutch Left (10°)": {
        "vertical_fraction": 1.0,
        "margin": 0.08,
        "roll_degrees": -10.0,
        "allowed_border_clipping": (),
    },
    "Dutch Right (10°)": {
        "vertical_fraction": 1.0,
        "margin": 0.08,
        "roll_degrees": 10.0,
        "allowed_border_clipping": (),
    },
}

CAMERA_PRESETS = tuple(CAMERA_PRESET_SPECS)


def camera_preset_spec(name):
    try:
        return CAMERA_PRESET_SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported camera preset {name!r}; choose one of: {', '.join(CAMERA_PRESETS)}"
        ) from error

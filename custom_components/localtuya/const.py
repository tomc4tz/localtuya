"""Constants for localtuya integration."""

ATTR_CURRENT = "current"
ATTR_CURRENT_CONSUMPTION = "current_consumption"
ATTR_VOLTAGE = "voltage"

CONF_LOCAL_KEY = "local_key"
CONF_PROTOCOL_VERSION = "protocol_version"
CONF_DPS_STRINGS = "dps_strings"
CONF_PRODUCT_KEY = "product_key"
CONF_IS_GATEWAY = "is_gateway"
CONF_PARENT_GATEWAY = "parent_gateway"

# light
CONF_BRIGHTNESS_LOWER = "brightness_lower"
CONF_BRIGHTNESS_UPPER = "brightness_upper"
CONF_COLOR = "color"
CONF_COLOR_MODE = "color_mode"
CONF_COLOR_TEMP_MIN_KELVIN = "color_temp_min_kelvin"
CONF_COLOR_TEMP_MAX_KELVIN = "color_temp_max_kelvin"
CONF_COLOR_TEMP_REVERSE = "color_temp_reverse"
CONF_MUSIC_MODE = "music_mode"

# switch
CONF_CURRENT = "current"
CONF_CURRENT_CONSUMPTION = "current_consumption"
CONF_VOLTAGE = "voltage"

# cover
CONF_COMMANDS_SET = "commands_set"
CONF_POSITIONING_MODE = "positioning_mode"
CONF_CURRENT_POSITION_DP = "current_position_dp"
CONF_SET_POSITION_DP = "set_position_dp"
CONF_POSITION_INVERTED = "position_inverted"
CONF_SPAN_TIME = "span_time"

# fan
CONF_FAN_SPEED_CONTROL = "fan_speed_control"
CONF_FAN_OSCILLATING_CONTROL = "fan_oscillating_control"
CONF_FAN_SPEED_LOW = "fan_speed_low"
CONF_FAN_SPEED_MEDIUM = "fan_speed_medium"
CONF_FAN_SPEED_HIGH = "fan_speed_high"

# sensor
CONF_SCALING = "scaling"

DATA_DISCOVERY = "discovery"

DOMAIN = "localtuya"

# Platforms in this list must support config flows
PLATFORMS = [
    "binary_sensor",
    "cover",
    "fan",
    "light",
    "number",
    "select",
    "sensor",
    "switch",
]

# gateway & sub-device
GW_REQ_ADD = "request_add"
GW_REQ_REMOVE = "request_remove"
GW_REQ_STATUS = "request_status"
GW_REQ_SET_DP = "request_set_dp"
GW_REQ_SET_DPS = "request_set_dps"
GW_EVT_REQ_ACK = "request_acknowledged"
GW_EVT_STATUS_UPDATED = "event_status_updated"
GW_EVT_CONNECTED = "event_connected"
GW_EVT_DISCONNECTED = "event_disconnected"

TUYA_DEVICE = "tuya_device"

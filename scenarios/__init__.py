from . import saloon_64, saloon_64_layered
from .bedroom_1 import build_scenarios as build_bedroom_scenarios

SCENARIOS = {
    saloon_64.SCENARIO_NAME: saloon_64,
    saloon_64_layered.SCENARIO_NAME: saloon_64_layered,
}
SCENARIOS.update(build_bedroom_scenarios())

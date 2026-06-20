from engine.bicep_curl import BicepCurl
from engine.bench_press import BenchPress
from engine.tri_pushdown import TriPushdown
from engine.shoulder_press import ShoulderPress
from engine.lat_pulldown import LatPulldown
from engine.leg_extension import LegExtension

EXERCISES = {
    "bicep_curl":     BicepCurl,
    "bench_press":    BenchPress,
    "tri_pushdown":   TriPushdown,
    "shoulder_press": ShoulderPress,
    "lat_pulldown":   LatPulldown,
    "leg_extension":  LegExtension,
}
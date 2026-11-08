from .base import BaseBenchmark
from .mmbench import MMBenchBenchmark
from .pope import POPEBenchmark
from .scienceqa import ScienceQABenchmark
from .textvqa import TextVQABenchmark
from .docvqa import DocVQABenchmark
from .mmvet import MMVetBenchmark

BENCHMARKS = {
    "mmbench": MMBenchBenchmark,
    "pope": POPEBenchmark,
    "scienceqa": ScienceQABenchmark,
    "textvqa": TextVQABenchmark,
    "docvqa": DocVQABenchmark,
    "mmvet": MMVetBenchmark,
}

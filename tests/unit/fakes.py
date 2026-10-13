"""Stand-ins for the adapters, so services run without Docker, Hugging Face or a terminal."""

from collections.abc import Iterator

from host_your_own_llm.domain import ModelCheck, ModelSummary, ServiceState, StackConfig
from host_your_own_llm.errors import ModelError


class FakeCompose:
    def __init__(self, ps: list[ServiceState] | None = None, running: bool = False):
        self.calls: list[tuple] = []
        self._ps = ps or []
        self.running = running

    def run(self, *args: str, quiet: bool = False) -> str:
        self.calls.append(("run", *args))
        return ""

    def build(self, service: str) -> None:
        self.calls.append(("build", service))

    def up(self, service: str, wait: bool = False) -> None:
        self.calls.append(("up", service))

    def down(self, volumes: bool = False) -> None:
        self.calls.append(("down", volumes))

    def logs(self, service: str, tail: int = 40) -> None:
        self.calls.append(("logs", service))

    def ps(self) -> list[ServiceState]:
        return self._ps

    def is_running(self, service: str) -> bool:
        return self.running

    def remove_container(self, name: str) -> None:
        self.calls.append(("rm", name))


class FakeStore:
    def __init__(self, models: list[str] | None = None):
        self.models = list(models or [])
        self.calls: list[tuple] = []

    def start(self) -> None:
        self.calls.append(("start",))

    def list_models(self) -> list[str]:
        return list(self.models)

    def seed(self, model_id: str | None = None) -> None:
        self.calls.append(("seed", model_id))
        if model_id:
            self.models.append(model_id)

    def rm(self, model_id: str) -> None:
        self.calls.append(("rm", model_id))
        self.models.remove(model_id)


class FakeHub:
    def __init__(
        self,
        checks: dict[str, ModelCheck | ModelError] | None = None,
        tokens: dict[str, set[str]] | None = None,
        listing: list[ModelSummary] | ModelError | None = None,
    ):
        self.checks = checks or {}
        self.tokens = tokens or {}  # model id -> tokens that can access it
        self.listing = listing or []
        self.searches: list[tuple[str | None, bool]] = []

    def search_models(self, query: str | None = None, trending: bool = False) -> Iterator[ModelSummary]:
        self.searches.append((query, trending))
        if isinstance(self.listing, ModelError):
            raise self.listing
        yield from (m for m in self.listing if not query or query in m.model_id)

    def check_model(self, model_id: str, token: str | None) -> ModelCheck:
        result = self.checks[model_id]
        if isinstance(result, ModelError):
            raise result
        return result

    def has_access(self, model_id: str, token: str) -> bool:
        return token in self.tokens.get(model_id, set())


class FakePrompter:
    """Answers questions from a queue. An empty answer takes the question's default."""

    def __init__(self, *answers: object, tty: bool = True):
        self.answers = list(answers)
        self.questions: list[str] = []
        self.tty = tty

    def _next(self, question: str) -> object:
        self.questions.append(question)
        if not self.answers:
            raise AssertionError(f"unexpected question: {question}")
        return self.answers.pop(0)

    def confirm(self, question: str, default: bool) -> bool:
        answer = self._next(question)
        return default if answer == "" else bool(answer)

    def ask(self, question: str, default: str | None = None) -> str:
        answer = self._next(question)
        return default if answer == "" and default is not None else str(answer)

    def ask_int(self, question: str, default: int) -> int:
        answer = self._next(question)
        return default if answer == "" else int(answer)

    def ask_secret(self, question: str) -> str | None:
        if not self.tty:
            return None
        return str(self._next(question))


class FakeVetter:
    """Accepts every model with the token it is given, unless told to reject."""

    def __init__(self, token: str | None = "", reject: bool = False):
        self.token = token
        self.reject = reject
        self.vetted: list[str] = []

    def gpu(self):
        return None

    def vet(self, model_id: str, token: str) -> str | None:
        self.vetted.append(model_id)
        if self.reject:
            return None
        return self.token if self.token is not None else token


class FakeStack:
    def __init__(self, running: bool = False):
        self.running = running
        self.started: list[StackConfig] = []

    def vllm_running(self) -> bool:
        return self.running

    def start_vllm(self, config: StackConfig) -> None:
        self.started.append(config)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
